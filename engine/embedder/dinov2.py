import itertools
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from einops import einops
from matplotlib import pyplot as plt
from skimage.measure import block_reduce
from sklearn.decomposition import PCA
from tqdm import tqdm

from geodataset.dataset import SegmentationLabeledRasterCocoDataset

from config.config_parsers.embedder_parsers import DINOv2InferConfig
from engine.embedder.utils import apply_pca_to_images


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

    def preprocess(self, x: np.ndarray):
        x = torch.Tensor(x).unsqueeze(0)  # instead of unsqueeze, add support for batches
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

    def __call__(self, x: np.ndarray):
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
        dfs = []
        down_sampled_masks_list = []
        embeddings_list = []
        tqdm_dataset = tqdm(dataset, desc="Inferring DINOv2...")
        for i, x in enumerate(tqdm_dataset):
            image = x[0]
            labels = x[1]

            embeddings, pads = self(image)

            masks = labels['masks']
            masks = np.stack(masks, axis=0)
            # Applying padding to the masks
            pads = ((0, 0), (pads[0], pads[1]), (pads[2], pads[3]))
            masks = np.pad(masks, pads, mode='constant', constant_values=0)

            down_sampled_masks = block_reduce(
                masks,
                block_size=(1, self.vit_patch_size, self.vit_patch_size),
                func=np.mean
            )

            df = pd.DataFrame({'labels': labels['labels'],
                               'area': labels['area'],
                               'iscrowd': labels['iscrowd']})
            df['image_id'] = labels['image_id'][0]
            df['image_path'] = dataset.tiles[labels['image_id'][0]]['path']

            dfs.append(df)
            embeddings_list.append(embeddings.squeeze(0))
            down_sampled_masks_list.append(down_sampled_masks)

        stacked_embeddings = np.stack(embeddings_list, axis=0)
        reduced_stacked_embeddings = apply_pca_to_images(
            stacked_embeddings,
            n_patches=self.config.pca_n_patches,
            n_features=self.config.pca_n_features
        )

        print("Computing embeddings for each mask...")
        for df, down_sampled_masks, reduced_embeddings in zip(dfs, down_sampled_masks_list, reduced_stacked_embeddings):
            down_sampled_masks_patches_embeddings = reduced_embeddings * down_sampled_masks[:, :, :, np.newaxis]
            down_sampled_masks_embeddings = np.sum(down_sampled_masks_patches_embeddings, axis=(1, 2))

            non_zero_mask = down_sampled_masks > 0
            non_zero_count = np.sum(non_zero_mask, axis=(1, 2))
            non_zero_count = np.where(non_zero_count == 0, 1, non_zero_count)
            non_zero_patches_mean = down_sampled_masks_embeddings / non_zero_count[:, np.newaxis]

            df['embeddings'] = non_zero_patches_mean.tolist()

        final_df = pd.concat(dfs)

        print("Done.")

        return final_df
