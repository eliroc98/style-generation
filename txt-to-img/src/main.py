"""Main script to run the IoU experiment for content and style in diffusion models."""

import warnings
import pandas as pd
from tqdm import tqdm
import torch
from daam import trace, set_seed

# Import from our refactored modules
import config
from data_utils import generate_prompts, get_token_indices_for_template_words
from model_utils import setup_diffusion_model
from analysis_utils import (
    calculate_iou_metrics,
    calculate_iou_baseline,
    save_image_with_heatmaps,
)

# Apply warnings filters at the start of the execution
warnings.filterwarnings("ignore", module="daam.utils")
warnings.filterwarnings("ignore", module="huggingface_hub.file_download")


def run_experiment():
    """
    Main function to run the IoU experiment.
    """
    # 1. Setup Model
    pipe = setup_diffusion_model(config.MODEL_ID, config.DEVICE)

    # 2. Generate Prompts
    prompts_info_list = generate_prompts(
        config.PROMPT_TEMPLATES, config.MEDIUMS, config.CONTENTS, config.STYLES
    )

    results = []

    # 3. Main Processing Loop
    current_seed = config.SEED_START_VALUE
    for prompt_info in tqdm(prompts_info_list, desc="Processing prompts"):
        gen = set_seed(current_seed)  # Set seed for reproducibility for this prompt

        template = prompt_info["template"]
        prompt_text = prompt_info["text"]
        content_word = prompt_info["content_word"]
        style_word = prompt_info["style_word"]
        medium_word = prompt_info["medium_word"]

        resolved_words, token_indices, content_idx, style_idx, _ = (
            get_token_indices_for_template_words(
                pipe.tokenizer,
                prompt_text,
                template,
                content_word,
                style_word,
                medium_word,
            )
        )

        with torch.no_grad(), trace(pipe) as tc:
            out = pipe(
                prompt_text,
                num_inference_steps=config.NUM_INFERENCE_STEPS,
                generator=gen,
            )
            image = out.images[0]
            global_heat_map = tc.compute_global_heat_map()

            word_heatmaps = []
            for word, token_idx in zip(resolved_words, token_indices):
                heatmap = global_heat_map.compute_word_heat_map(
                    word=word, word_idx=token_idx
                ).expand_as(image=image)
                word_heatmaps.append(heatmap)

            content_map = word_heatmaps[content_idx]
            style_map = word_heatmaps[style_idx]

            for t_val in config.IOU_THRESHOLDS_TO_TEST:
                for use_quantile_flag in [False, True]:
                    (
                        iou_bl_mean,
                        iou_bl_std,
                        sup_a_bl,
                        sup_b_bl,
                        sup_int_bl,
                        sup_union_bl,
                    ) = calculate_iou_baseline(
                        word_heatmaps,
                        content_idx,
                        style_idx,
                        content_style_only=False,
                        threshold=t_val,
                        use_quantile=use_quantile_flag,
                        device=config.DEVICE,
                    )

                    (
                        iou_bl_cs_mean,
                        iou_bl_cs_std,
                        sup_a_bl_cs,
                        sup_b_bl_cs,
                        sup_int_bl_cs,
                        sup_union_bl_cs,
                    ) = calculate_iou_baseline(
                        word_heatmaps,
                        content_idx,
                        style_idx,
                        content_style_only=True,
                        threshold=t_val,
                        use_quantile=use_quantile_flag,
                        device=config.DEVICE,
                    )

                    iou_score, sup_content, sup_style, sup_intersect, sup_union = (
                        calculate_iou_metrics(
                            content_map,
                            style_map,
                            t_val,
                            use_quantile_flag,
                            config.DEVICE,
                        )
                    )

                    results.append(
                        {
                            "prompt": prompt_text,
                            "template": template,
                            "content_word": content_word,
                            "style_word": style_word,
                            "seed": current_seed,
                            "use_quantile": use_quantile_flag,
                            "iou_threshold": t_val,
                            "iou_baseline_mean": iou_bl_mean,
                            "iou_baseline_std": iou_bl_std,
                            "support_a_baseline": sup_a_bl,
                            "support_b_baseline": sup_b_bl,
                            "support_intersection_baseline": sup_int_bl,
                            "support_union_baseline": sup_union_bl,
                            "iou_baseline_cs_mean": iou_bl_cs_mean,
                            "iou_baseline_cs_std": iou_bl_cs_std,
                            "support_a_baseline_cs": sup_a_bl_cs,
                            "support_b_baseline_cs": sup_b_bl_cs,
                            "support_intersection_baseline_cs": sup_int_bl_cs,
                            "support_union_baseline_cs": sup_union_bl_cs,
                            "iou_score": iou_score,
                            "support_content": sup_content,
                            "support_style": sup_style,
                            "support_intersection": sup_intersect,
                            "support_union": sup_union,
                        }
                    )

        # Memory cleanup
        if config.DEVICE.startswith("cuda"):
            torch.cuda.empty_cache()
        elif config.DEVICE == "mps":
            torch.mps.empty_cache()

        current_seed += 1

    # 4. Save Results
    results_df = pd.DataFrame(results)
    csv_path = config.OUTPUT_DIR / "content_style_iou_full_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # 5. Save Relevant Image Visualizations
    results_df["delta_iou"] = results_df["iou_baseline_mean"] - results_df["iou_score"]

    for iou_threshold_for_vis, use_quantile_for_vis in zip(
        config.DEFAULT_IOU_THRESHOLD_FOR_VIS, config.DEFAULT_USE_QUANTILE_FOR_VIS
    ):
        # Filter for the specific configuration for visualization
        plot_config_df = results_df.loc[
            (results_df["iou_threshold"] == iou_threshold_for_vis)
            & (results_df["use_quantile"] == use_quantile_for_vis)
        ]

        # Group by unique image generation settings and average delta_iou
        results_to_plot = (
            plot_config_df.groupby(
                ["prompt", "content_word", "style_word", "seed"], as_index=False
            )
            .agg({"delta_iou": "mean"})
            .sort_values(by="delta_iou", ascending=False)
            .dropna()
        )

        # Top N images
        top_dir_name = (
            f"{iou_threshold_for_vis}_"
            + f"{'quantile' if use_quantile_for_vis else 'threshold'}"
            + "/top"
        )
        TOP_DIR = config.IMAGE_DIR / top_dir_name
        TOP_DIR.mkdir(parents=True, exist_ok=True)

        for _, row in results_to_plot.head(config.N_RELEVANT_IMAGES_TO_SAVE).iterrows():
            save_image_with_heatmaps(
                pipe,
                row["prompt"],
                row["content_word"],
                row["style_word"],
                int(row["seed"]),
                TOP_DIR,
            )

        # Bottom N images
        bottom_dir_name = (
            f"{iou_threshold_for_vis}_"
            + f"{'quantile' if use_quantile_for_vis else 'threshold'}"
            + "/bottom"
        )
        BOTTOM_DIR = config.IMAGE_DIR / bottom_dir_name
        BOTTOM_DIR.mkdir(parents=True, exist_ok=True)

        for _, row in results_to_plot.tail(config.N_RELEVANT_IMAGES_TO_SAVE).iterrows():
            save_image_with_heatmaps(
                pipe,
                row["prompt"],
                row["content_word"],
                row["style_word"],
                int(row["seed"]),
                BOTTOM_DIR,
            )


if __name__ == "__main__":
    run_experiment()
