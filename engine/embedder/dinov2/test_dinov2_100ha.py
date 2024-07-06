import os
import time
from pathlib import Path
from typing import List

import albumentations
import numpy as np
import pandas as pd
import rasterio
import umap
from geodataset.dataset import SegmentationLabeledRasterCocoDataset
from geodataset.dataset.base_dataset import BaseLabeledRasterCocoDataset
from geodataset.utils import rle_segmentation_to_mask
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config.config_parsers.embedder_parsers import DINOv2InferConfig
from engine.embedder.dinov2.dinov2 import DINOv2Inference


class DINOv2SegmentationLabeledRasterCocoDataset(BaseLabeledRasterCocoDataset):
    def __init__(self, fold: str, root_path: Path or List[Path],
                 transform: albumentations.core.composition.Compose = None,
                 image_size_center_crop_pad: int = None):
        super().__init__(fold=fold, root_path=root_path, transform=transform)

        self.image_size_center_crop_pad = image_size_center_crop_pad

    def __getitem__(self, idx: int):
        """
        Retrieves a tile and its segmentations/masks by index, applying any specified transformations.

        Parameters:
        - idx: int, the index of the tile to retrieve.

        Returns:
        - A tuple containing the transformed tile and its segmentations/masks.
        """
        tile_info = self.tiles[idx]

        with rasterio.open(tile_info['path']) as tile_file:
            tile = tile_file.read([1, 2, 3])  # Reading the first three bands

        labels = tile_info['labels']
        masks = []

        for label in labels:
            if 'segmentation' in label:
                segmentation = label['segmentation']
                if ('is_rle_format' in label and label['is_rle_format']) or isinstance(segmentation, dict):
                    # RLE format
                    mask = rle_segmentation_to_mask(segmentation)
                else:
                    raise NotImplementedError("Please make sure that the masks are encoded using RLE.")

                masks.append(mask)

        category_ids = np.array([0 if label['category_id'] is None else label['category_id']
                                 for label in labels])

        if self.transform:
            transformed = self.transform(image=tile.transpose((1, 2, 0)),
                                         mask=np.stack(masks, axis=0),
                                         labels=category_ids)
            transformed_image = transformed['image'].transpose((2, 0, 1))
            transformed_masks = [mask for mask in transformed['mask']]
            transformed_category_ids = transformed['labels']
        else:
            transformed_image = tile
            transformed_masks = masks
            transformed_category_ids = category_ids

        transformed_image_not_cropped = transformed_image / 255

        if self.image_size_center_crop_pad:
            image_size = self.image_size_center_crop_pad
            if transformed_image.shape[1] > image_size:
                # crop
                data_center = int(transformed_image.shape[1] / 2)
                data = transformed_image[:,
                       data_center - image_size // 2:data_center + image_size // 2,
                       data_center - image_size // 2:data_center + image_size // 2,
                       ]
                masks = [mask[
                         data_center - image_size // 2:data_center + image_size // 2,
                         data_center - image_size // 2:data_center + image_size // 2,
                         ] for mask in masks]
                transformed_image = data
                transformed_masks = masks
            elif transformed_image.shape[1] < image_size:
                # pad
                padding = (image_size - transformed_image.shape[1]) // 2
                padded_data = np.zeros((transformed_image.shape[0], image_size, image_size),
                                       dtype=transformed_image.dtype)
                padded_data[:, padding:padding + transformed_image.shape[1], padding:padding + transformed_image.shape[1]] = transformed_image
                data = padded_data
                masks = []
                for transformed_mask in transformed_masks:
                    padded_mask = np.zeros((image_size, image_size), dtype=transformed_mask.dtype)
                    padded_mask[padding:padding + transformed_mask.shape[0], padding:padding + transformed_mask.shape[0]] = transformed_mask
                    masks.append(padded_mask)
                transformed_image = data
                transformed_masks = masks

        transformed_image = transformed_image / 255  # normalizing
        area = np.array([np.sum(mask) for mask in masks])
        # suppose all instances are not crowd
        iscrowd = np.zeros((len(transformed_masks),))
        # get tile id
        image_id = np.array([idx])
        transformed_masks = {'masks': transformed_masks, 'labels': transformed_category_ids,
                             'area': area, 'iscrowd': iscrowd, 'image_id': image_id}

        #
        # # display transformed_image_not_cropped and transformed_image side to side
        # fig, ax = plt.subplots(1, 2, figsize=(15, 15))
        # ax[0].imshow(transformed_image_not_cropped.transpose(1, 2, 0))
        # ax[1].imshow(transformed_image.transpose(1, 2, 0))
        # plt.show()

        return transformed_image, transformed_masks


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
