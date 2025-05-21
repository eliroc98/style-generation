from pathlib import Path
import itertools
import warnings
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from daam import trace, set_seed

warnings.filterwarnings("ignore", module="daam.utils")
warnings.filterwarnings("ignore", module="huggingface_hub.file_download")

# supported models:
#   - 'runwayml/stable-diffusion-v1-5',
#   - 'stabilityai/stable-diffusion-2-base',
#   - 'stabilityai/stable-diffusion-2',
#   - 'stabilityai/stable-diffusion-2-1-base',
#   - 'stabilityai/stable-diffusion-2-1',
#   - 'stabilityai/stable-diffusion-xl-base-1.0'

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = (
    "cuda:1"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_DIR = OUTPUT_DIR / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

ENTITIES_DIR = Path("entities")

with open(ENTITIES_DIR / "objects.txt", "r", encoding="utf-8") as f:
    CONTENTS = f.read().splitlines()

with open(ENTITIES_DIR / "movements.txt", "r", encoding="utf-8") as f:
    MOVEMENTS = [style.split()[1].replace("_", " ") for style in f.read().splitlines()]

with open(ENTITIES_DIR / "artists.txt", "r", encoding="utf-8") as f:
    ARTISTS = [artist.split()[1].replace("_", " ") for artist in f.read().splitlines()]

STYLES = MOVEMENTS + ARTISTS

CONTENTS = CONTENTS[:10]
STYLES = STYLES[:10]
MEDIUMS = ["painting"]  # TODO: expand

PROMPT_TEMPLATES = [
    # "a <MEDIUM> of a <CONTENT> by <STYLE>",
    "a <MEDIUM> of a <CONTENT> in the <STYLE> style",
    "a <STYLE> <MEDIUM> of a <CONTENT>",
    # "<CONTENT> <STYLE>",
    # "<CONTENT> . <STYLE>",
    # "<CONTENT> in the style of <STYLE>",
    "a <CONTENT> in the <STYLE> style",
    "a <CONTENT> with <STYLE> style",
]

IOU_THRESHOLDS_TO_TEST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@torch.no_grad()
def setup_model():
    if "xl" in MODEL_ID:
        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            use_auth_token=True,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, use_auth_token=True)
    pipe = pipe.to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def iou(a, b, t: float, use_quantile: bool) -> float:
    """
    Calculates Intersection over Union (IoU) and support metrics.

    Returns:
        tuple: (iou_score, support_a, support_b, support_intersection, support_union)
    """
    a = a.float().to(DEVICE)
    b = b.float().to(DEVICE)
    n = a.numel()

    if use_quantile:
        threshold_a = a.quantile(t)
        threshold_b = b.quantile(t)
        bin_a = a >= threshold_a
        bin_b = b >= threshold_b
    else:
        threshold_a = t
        threshold_b = t
        bin_a = a >= threshold_a
        bin_b = b >= threshold_b

    intersection_map = bin_a & bin_b
    union_map = bin_a | bin_b

    intersection = intersection_map.float().sum()
    union = union_map.float().sum()

    iou_score = 0.0 if union < 1e-6 else (intersection / union).item()

    support_a = bin_a.float().sum().item()
    support_b = bin_b.float().sum().item()
    support_intersection = intersection.item()
    support_union = union.item()

    return (
        iou_score,
        support_a / n,
        support_b / n,
        support_intersection / n,
        support_union / n,
    )


def iou_baseline(
    word_heatmaps: list,
    content_idx: int,
    style_idx: int,
    content_style_only: bool,
    t: float,
    use_quantile: bool,
) -> float:
    iou_scores = []
    for wi, wj in itertools.combinations(range(len(word_heatmaps)), r=2):
        if (wi == content_idx and wj == style_idx) or (
            wi == style_idx and wj == content_idx
        ):
            continue
        if content_style_only and not any(
            x for x in [wi, wj] if x in [content_idx, style_idx]
        ):
            continue
        iou_scores.append(
            iou(word_heatmaps[wi], word_heatmaps[wj], t=t, use_quantile=use_quantile)
        )
    iou_baseline_mean = np.mean([iou_scores[i][0] for i in range(len(iou_scores))])
    iou_baseline_std = np.std([iou_scores[i][0] for i in range(len(iou_scores))])
    sup_a_baseline = np.mean([iou_scores[i][1] for i in range(len(iou_scores))])
    sup_b_baseline = np.mean([iou_scores[i][2] for i in range(len(iou_scores))])
    sup_intersection_baseline = np.mean(
        [iou_scores[i][3] for i in range(len(iou_scores))]
    )
    sup_union_baseline = np.mean([iou_scores[i][4] for i in range(len(iou_scores))])
    return (
        iou_baseline_mean,
        iou_baseline_std,
        sup_a_baseline,
        sup_b_baseline,
        sup_intersection_baseline,
        sup_union_baseline,
    )


def generate_prompts():
    """
    Generates all unique prompt combinations from the global lists of
    PROMPT_TEMPLATES, CONTENTS, STYLES, MEDIUMS, and VERBS.

    Returns:
        list: A list of dictionaries, where each dictionary contains:
            - "text": The generated prompt string.
            - "content_word": The full content string.
            - "style_word": The full style string.
    """
    all_prompts_info = []
    prompt_combinations = list(
        itertools.product(*[PROMPT_TEMPLATES, MEDIUMS, CONTENTS, STYLES])
    )

    for prompt_combination in prompt_combinations:
        template, medium, content, style = prompt_combination
        prompt_text = template.replace("<MEDIUM>", medium)
        prompt_text = prompt_text.replace("<CONTENT>", content)
        prompt_text = prompt_text.replace("<STYLE>", style)
        prompt_info = {
            "template": template,
            "text": prompt_text,
            "content_word": content,
            "style_word": style,
            "medium_word": medium,
        }
        all_prompts_info.append(prompt_info)

    return all_prompts_info


def save_image_visualization(pipe, prompt, content_word, style_word, seed, img_dir):
    """
    Saves the image visualization for the given prompt and words.

    Args:
        pipe (DiffusionPipeline): The diffusion pipeline.
        prompt (str): The prompt string.
        content_word (str): The content word.
        style_word (str): The style word.
        seed (int): The random seed.
        img_dir (Path): The directory to save the image.
    """
    gen = set_seed(seed)
    with torch.no_grad(), trace(pipe) as tc:
        out = pipe(prompt, num_inference_steps=30, generator=gen)
        original_image = out.images[0]
        global_heat_map = tc.compute_global_heat_map()
        content_map = global_heat_map.compute_word_heat_map(content_word)
        style_map = global_heat_map.compute_word_heat_map(style_word)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        content_map.plot_overlay(out.images[0], ax=axes[1])
        axes[1].set_title(f"{content_word}")
        axes[1].axis("off")

        style_map.plot_overlay(out.images[0], ax=axes[2])
        axes[2].set_title(f"{style_word}")
        axes[2].axis("off")

        plt.suptitle(f"{prompt}", y=0.075)
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        fig.savefig(img_dir / f"img_{seed}_overlay.png")

        plt.close()


pipe = setup_model()
prompts = generate_prompts()
results = []

for i, prompt_info in enumerate(tqdm(prompts)):
    template = prompt_info["template"]
    prompt = prompt_info["text"]
    content_word = prompt_info["content_word"]
    style_word = prompt_info["style_word"]
    gen = set_seed(i)

    parts = template.split()
    words = []
    for part in parts:
        if part == "<CONTENT>":
            words.append(content_word)
        elif part == "<STYLE>":
            words.append(style_word)
        elif part == "<MEDIUM>":
            words.append(prompt_info["medium_word"])
        else:
            words.append(part)

    tokenized_prompt = [w.replace("</w>", "") for w in pipe.tokenizer.tokenize(prompt)]
    token_indexes = []
    start_idx = 0
    for word in words:
        normalized_word = word.replace(" ", "").lower()
        for n in range(1, len(tokenized_prompt) - start_idx + 1):
            current_segment = tokenized_prompt[start_idx : start_idx + n]
            if "".join(current_segment).lower() == normalized_word:
                token_indexes.append(start_idx)
                start_idx += n
                break

    with torch.no_grad(), trace(pipe) as tc:
        out = pipe(prompt, num_inference_steps=30, generator=gen)
        image = out.images[0]
        global_heat_map = tc.compute_global_heat_map()

        word_heatmaps = [
            global_heat_map.compute_word_heat_map(
                word=word, word_idx=token_idx
            ).expand_as(image=image)
            for word, token_idx in zip(words, token_indexes)
        ]

        content_idx = words.index(content_word)
        style_idx = words.index(style_word)

        content_map = word_heatmaps[content_idx]
        style_map = word_heatmaps[style_idx]

        for t_val in IOU_THRESHOLDS_TO_TEST:
            for use_quantile in [False, True]:
                (
                    iou_baseline_mean,
                    iou_baseline_std,
                    sup_a_baseline,
                    sup_b_baseline,
                    sup_intersection_baseline,
                    sup_union_baseline,
                ) = iou_baseline(
                    word_heatmaps,
                    content_idx,
                    style_idx,
                    content_style_only=False,
                    t=t_val,
                    use_quantile=use_quantile,
                )
                (
                    iou_baseline_cs_mean,
                    iou_baseline_cs_std,
                    sup_a_baseline_cs,
                    sup_b_baseline_cs,
                    sup_intersection_baseline_cs,
                    sup_union_baseline_cs,
                ) = iou_baseline(
                    word_heatmaps,
                    content_idx,
                    style_idx,
                    content_style_only=True,
                    t=t_val,
                    use_quantile=use_quantile,
                )
                iou_score, sup_content, sup_style, sup_intersect, sup_union = iou(
                    content_map, style_map, t=t_val, use_quantile=use_quantile
                )
                results.append(
                    {
                        "prompt": prompt,
                        "template": template,
                        "content_word": content_word,
                        "style_word": style_word,
                        "seed": i,
                        "use_quantile": use_quantile,
                        "iou_threshold": t_val,
                        "iou_baseline_mean": iou_baseline_mean,
                        "iou_baseline_std": iou_baseline_std,
                        "support_a_baseline": sup_a_baseline,
                        "support_b_baseline": sup_b_baseline,
                        "support_intersection_baseline": sup_intersection_baseline,
                        "support_union_baseline": sup_union_baseline,
                        "iou_baseline_cs_mean": iou_baseline_cs_mean,
                        "iou_baseline_cs_std": iou_baseline_cs_std,
                        "support_a_baseline_cs": sup_a_baseline_cs,
                        "support_b_baseline_cs": sup_b_baseline_cs,
                        "support_intersection_baseline_cs": sup_intersection_baseline_cs,
                        "support_union_baseline_cs": sup_union_baseline_cs,
                        "iou_score": iou_score,
                        "support_content": sup_content,
                        "support_style": sup_style,
                        "support_intersection": sup_intersect,
                        "support_union": sup_union,
                    }
                )

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()

# Save results to CSV
csv_path = OUTPUT_DIR / "content_style_iou_results.csv"
results_df = pd.DataFrame(results)
results_df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")

# Save relevant images
N_RELEVANT = 10
IOU_THRESHOLD_CONFIG = 0.4
USE_QUANTILE_CONFIG = False

results_df["delta_iou"] = results_df["iou_baseline_mean"] - results_df["iou_score"]
results_to_plot = (
    results_df.loc[
        (results_df["iou_threshold"] == IOU_THRESHOLD_CONFIG)
        & (results_df["use_quantile"] == USE_QUANTILE_CONFIG)
    ]
    .groupby(["prompt", "content_word", "style_word", "seed"])
    .agg({"delta_iou": "mean"})
    .sort_values(by="delta_iou", ascending=False)
    .reset_index()
    .dropna()
)

for _, row in results_to_plot.head(N_RELEVANT).iterrows():
    print(row["seed"], row["prompt"])
    TOP_DIR = (
        IMAGE_DIR
        / (
            str(IOU_THRESHOLD_CONFIG)
            + "_"
            + str("quantile" if USE_QUANTILE_CONFIG else "threshold")
        )
        / "top"
    )
    TOP_DIR.mkdir(parents=True, exist_ok=True)
    save_image_visualization(
        pipe,
        row["prompt"],
        row["content_word"],
        row["style_word"],
        row["seed"],
        TOP_DIR,
    )

for _, row in results_to_plot.tail(N_RELEVANT).iterrows():
    print(row["seed"], row["prompt"])
    BOTTOM_DIR = (
        IMAGE_DIR
        / (
            str(IOU_THRESHOLD_CONFIG)
            + "_"
            + str("quantile" if USE_QUANTILE_CONFIG else "threshold")
        )
        / "bottom"
    )
    BOTTOM_DIR.mkdir(parents=True, exist_ok=True)
    save_image_visualization(
        pipe,
        row["prompt"],
        row["content_word"],
        row["style_word"],
        row["seed"],
        BOTTOM_DIR,
    )
