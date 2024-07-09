import os
import time
from pathlib import Path

from config.config_parsers.embedder_parsers import DINOv2InferConfig
from engine.embedder.dinov2.dinov2 import DINOv2Inference
from engine.embedder.dinov2.dinov2_dataset import DINOv2SegmentationLabeledRasterCocoDataset


if __name__ == '__main__':
    input_path = 'C:/Users/Hugo/Documents/XPrize/infer/20240521_zf2100ha_highres_m3m_rgb_2_clahe_no_black_adjustedL_adjustedA1536_adjustedB1536/20240521_zf2100ha_highres_m3m_rgb_clahe_final'
    metric = 'cosine'
    image_size_center_crop_pad = None
    reduce_algo = 'umap'
    perplexity = 500
    min_cluster_size = 5
    n_components = 5
    use_cls_token = True
    now = time.time()
    output_dir = f"./dbscan_plots_dinov2/{reduce_algo}_{metric}_{n_components}_{image_size_center_crop_pad}_{min_cluster_size}_{perplexity}_{now}"
    os.makedirs(output_dir, exist_ok=True)

    size = 'base'

    dataset = DINOv2SegmentationLabeledRasterCocoDataset(
        root_path=[
            Path(input_path),
        ],
        fold='infer',
        image_size_center_crop_pad=image_size_center_crop_pad
    )

    config = DINOv2InferConfig(
        size=size,
        batch_size=1,
        instance_segmentation=False,
        mean_std_descriptor='imagenet'
    )

    dinov2 = DINOv2Inference(
        config=config,

    )

    embeddings_df = dinov2.infer_on_segmentation_dataset(
        dataset=dataset,
        average_non_masked_patches=not use_cls_token
    )

    embeddings_df.to_pickle(f'embeddings_df_clahe1536_{use_cls_token}_{image_size_center_crop_pad}_{now}.pkl')
