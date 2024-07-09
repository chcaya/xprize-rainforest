import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from einops import einops
from geodataset.utils import rle_segmentation_to_mask, mask_to_polygon, tiles_polygons_gdf_to_crs_gdf
from scipy import sparse
from skimage.measure import block_reduce
from tqdm import tqdm

from geodataset.dataset import SegmentationLabeledRasterCocoDataset

from config.config_parsers.embedder_parsers import DINOv2InferConfig
from engine.embedder.dinov2.dinov2_dataset import DINOv2SegmentationLabeledRasterCocoDataset
from engine.embedder.utils import apply_pca_to_images, IMAGENET_MEAN, IMAGENET_STD, FOREST_QPEB_MEAN, FOREST_QPEB_STD
from engine.utils.utils import collate_fn_segmentation


class DINOv2Preprocessor:
    def __init__(self, vit_patch_size: int, normalize: bool, instance_normalization: bool, mean_std_descriptor: str = None):
        self.vit_patch_size = vit_patch_size
        self.normalize = normalize
        self.instance_normalization = instance_normalization
        self.mean_std_descriptor = mean_std_descriptor

        assert self.mean_std_descriptor is None or self.mean_std_descriptor in ['imagenet', 'forest_qpeb']

    def _get_pad(self, size):
        new_size = math.ceil(size / self.vit_patch_size) * self.vit_patch_size
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    def preprocess(self, x: torch.Tensor):
        if self.normalize:
            if self.instance_normalization:
                mean = x.mean(dim=(2, 3), keepdim=True)
                var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
                x = tvf.normalize(x, mean=list(mean[0,:,0,0]), std=list(var[0,:,0,0]))
            else:
                if self.mean_std_descriptor == 'imagenet':
                    x = tvf.normalize(x, mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD))
                elif self.mean_std_descriptor == 'forest_qpeb':
                    x = tvf.normalize(x, mean=list(FOREST_QPEB_MEAN), std=list(FOREST_QPEB_STD))
                else:
                    raise ValueError("Invalid mean_std_descriptor value. Valid values are ['imagenet', 'forest_qpeb']")

        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        x = F.pad(x, pads)
        num_h_patches, num_w_patches = x.shape[2] // self.vit_patch_size, x.shape[3] // self.vit_patch_size
        return x, pads, num_h_patches, num_w_patches

    @staticmethod
    def postprocess(output, num_h_patches: int, num_w_patches: int):
        output = einops.rearrange(
                output, "b (h w) c -> b h w c", h=num_h_patches, w=num_w_patches
        )
        return output


class DINOv2Inference:
    SUPPORTED_SIZES = ['small', 'base', 'large', 'giant']
    EMBEDDING_SIZES = {
        'small': 384,
        'base': 768,
        'large': 1024,
        'giant': 1536
    }

    def __init__(self,
                 size: str,
                 normalize: bool,
                 instance_segmentation: bool,
                 mean_std_descriptor: str = None,):

        self.size = size
        self.normalize = normalize
        self.instance_segmentation = instance_segmentation
        self.mean_std_descriptor = mean_std_descriptor

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vit_patch_size = 14
        self.model = self._load_model()
        self.preprocessor = DINOv2Preprocessor(self.vit_patch_size, self.normalize, self.instance_segmentation, self.mean_std_descriptor)

    def _load_model(self):
        assert self.size in self.SUPPORTED_SIZES, \
            f"Invalid DINOv2 model size: \'{self.size}\'. Valid value are {self.SUPPORTED_SIZES}."

        model_name = f"dinov2_vit{self.size[0]}{self.vit_patch_size}_reg"

        return torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True).to(self.device)

    def __call__(self, x: torch.Tensor, average_non_masked_patches: bool):
        with torch.inference_mode():
            pp_x, pads, num_h_patches, num_w_patches = self.preprocessor.preprocess(x)
            pp_x = pp_x.to(self.device)

            output = self.model(pp_x, is_training=True)

            if average_non_masked_patches:
                output = output['x_norm_patchtokens']
                output_pp = self.preprocessor.postprocess(
                    output,
                    num_h_patches=num_h_patches,
                    num_w_patches=num_w_patches
                )
            else:
                output = output['x_norm_clstoken']
                output_pp = output

            return output_pp.cpu().numpy(), pads

    def infer_on_segmentation_dataset(self,
                                      dataset: DINOv2SegmentationLabeledRasterCocoDataset,
                                      average_non_masked_patches: bool,
                                      batch_size: int):
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  num_workers=3,
                                                  collate_fn=collate_fn_segmentation)

        dfs = []
        tqdm_dataset = tqdm(data_loader, desc="Inferring DINOv2...")
        for i, x in enumerate(tqdm_dataset):
            images = x[0]
            labels = x[1]

            embeddings, image_pads = self(images, average_non_masked_patches=average_non_masked_patches)
            for j, label in enumerate(labels):
                embedding = embeddings[j]
                image_masks = label['masks']
                image_masks = image_masks.cpu().detach().numpy()

                masks = np.stack(image_masks, axis=0)

                # Applying padding to the masks
                masks_pads = ((0, 0), (image_pads[0], image_pads[1]), (image_pads[2], image_pads[3]))
                image_masks = np.pad(masks, masks_pads, mode='constant', constant_values=0)

                down_sampled_masks = block_reduce(
                    image_masks,
                    block_size=(1, self.vit_patch_size, self.vit_patch_size),
                    func=np.mean
                )

                if average_non_masked_patches:
                    down_sampled_masks_patches_embeddings = embedding * down_sampled_masks[:, :, :, np.newaxis]
                    down_sampled_masks_embeddings = np.sum(down_sampled_masks_patches_embeddings, axis=(1, 2))

                    non_zero_mask = down_sampled_masks > 0
                    non_zero_count = np.sum(non_zero_mask, axis=(1, 2))
                    non_zero_count = np.where(non_zero_count == 0, 1, non_zero_count)
                    non_zero_patches_mean = down_sampled_masks_embeddings / non_zero_count[:, np.newaxis]
                    embedding = non_zero_patches_mean

                df = pd.DataFrame({
                    'labels': label['labels'].cpu().numpy().tolist(),
                    'geometry': label['labels_polygons'],
                    'area': label['area'].cpu().numpy().tolist(),
                    'iscrowd': label['iscrowd'].cpu().numpy().tolist(),
                    'image_id': [int(label['image_id'][0])] * len(label['labels']),
                    'tile_path': [dataset.tiles[int(label['image_id'][0])]['path']] * len(label['labels'].cpu().numpy()),
                    'embeddings': [embedding.flatten().tolist()],
                    'down_sampled_masks': down_sampled_masks.tolist()
                })
                dfs.append(df)

        final_gdf = gpd.GeoDataFrame(pd.concat(dfs))
        final_gdf_crs = tiles_polygons_gdf_to_crs_gdf(final_gdf)
        final_gdf_crs.set_geometry('geometry')
        final_gdf_crs['embeddings'] = final_gdf_crs['embeddings'].apply(lambda e: str(e))
        print("Done.")

        return final_gdf_crs


def infer_dinov2(data_roots: str,
                 image_size_center_crop_pad: int,
                 size: str,
                 use_cls_token: bool):

    dataset = DINOv2SegmentationLabeledRasterCocoDataset(
        root_path=data_roots,
        fold='infer',
        image_size_center_crop_pad=image_size_center_crop_pad
    )

    dinov2 = DINOv2Inference(
        size=size,
        normalize=False,
        instance_segmentation=False
    )

    embeddings_df = dinov2.infer_on_segmentation_dataset(
        dataset=dataset,
        average_non_masked_patches=not use_cls_token,
        batch_size=1
    )

    return embeddings_df
