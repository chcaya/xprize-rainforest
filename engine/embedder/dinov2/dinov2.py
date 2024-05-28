import itertools
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from einops import einops
from scipy import sparse
from skimage.measure import block_reduce
from tqdm import tqdm

from geodataset.dataset import SegmentationLabeledRasterCocoDataset

from config.config_parsers.embedder_parsers import DINOv2InferConfig
from engine.embedder.utils import apply_pca_to_images
from engine.utils.utils import collate_fn_segmentation


class DINOv2Preprocessor:
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, vit_patch_size: int):
        self.vit_patch_size = vit_patch_size

    def _get_pad(self, size):
        new_size = math.ceil(size / self.vit_patch_size) * self.vit_patch_size
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    def preprocess(self, x: torch.Tensor):
        x = tvf.normalize(x, mean=list(self.MEAN), std=list(self.STD))
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

    def __init__(self, config: DINOv2InferConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vit_patch_size = 14        # TODO put in config?
        self.model = self._load_model()
        self.preprocessor = DINOv2Preprocessor(self.vit_patch_size)

    def _load_model(self):
        assert self.config.size in self.SUPPORTED_SIZES, \
            f"Invalid DINOv2 model size: \'{self.config.size}\'. Valid value are {self.SUPPORTED_SIZES}."

        model_name = f"dinov2_vit{self.config.size[0]}{self.vit_patch_size}_reg"

        return torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True).to(self.device)

    def __call__(self, x: torch.Tensor):
        with torch.inference_mode():
            pp_x, pads, num_h_patches, num_w_patches = self.preprocessor.preprocess(x)
            pp_x = pp_x.to(self.device)

            output = self.model(pp_x, is_training=True)
            output = output['x_norm_patchtokens']

            output_pp = self.preprocessor.postprocess(
                output,
                num_h_patches=num_h_patches,
                num_w_patches=num_w_patches
            )

            output_pp = output_pp.cpu().detach().numpy()

            return output_pp, pads

    def infer_on_segmentation_dataset(self, dataset: SegmentationLabeledRasterCocoDataset):
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.config.batch_size,
                                                  num_workers=3,
                                                  collate_fn=collate_fn_segmentation)

        dfs = []
        embeddings_list = []
        down_sampled_masks_list = []
        tqdm_dataset = tqdm(data_loader, desc="Inferring DINOv2...")
        for i, x in enumerate(tqdm_dataset):
            images = x[0]
            labels = x[1]

            embeddings, image_pads = self(images)
            for j, label in enumerate(labels):
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

                down_sampled_masks_list.append(down_sampled_masks)
                embeddings_list.append(embeddings[j])

                df = pd.DataFrame({
                    'labels': label['labels'].numpy().tolist(),
                    'area': label['area'].numpy().tolist(),
                    'iscrowd': label['iscrowd'].numpy().tolist(),
                    'image_id': [int(label['image_id'][0])] * len(label['labels']),
                    'tiles_paths': [dataset.tiles[int(label['image_id'][0])]['path']] * len(label['labels']),
                })
                dfs.append(df)

        stacked_embeddings = np.stack(embeddings_list, axis=0)

        # if self.config.use_pca:
        #     stacked_embeddings = apply_pca_to_images(
        #         stacked_embeddings,
        #         pca_model_path=self.config.pca_model_path,
        #         n_patches=self.config.pca_n_patches,
        #         n_features=self.config.pca_n_features
        #     )

        for df, down_sampled_masks, reduced_embeddings in tqdm(
                zip(dfs, down_sampled_masks_list, stacked_embeddings),
                desc="Computing embeddings for each image masks...",
                total=len(dfs)
                ):                              # TODO this is really slow, should be parallelized
            down_sampled_masks_patches_embeddings = reduced_embeddings * down_sampled_masks[:, :, :, np.newaxis]
            down_sampled_masks_embeddings = np.sum(down_sampled_masks_patches_embeddings, axis=(1, 2))

            non_zero_mask = down_sampled_masks > 0
            non_zero_count = np.sum(non_zero_mask, axis=(1, 2))
            non_zero_count = np.where(non_zero_count == 0, 1, non_zero_count)
            non_zero_patches_mean = down_sampled_masks_embeddings / non_zero_count[:, np.newaxis]

            df['embeddings'] = non_zero_patches_mean.tolist()
            df['down_sampled_masks'] = down_sampled_masks.tolist()

        final_df = pd.concat(dfs)

        print("Done.")

        return final_df
