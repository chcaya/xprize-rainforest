import itertools
import math

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvf
from einops import einops

from config.config_parsers.embedder_parsers import DINOv2InferConfig


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

    def preprocess(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        x = tvf.normalize(x, mean=list(self.MEAN), std=list(self.STD))
        output = F.pad(x, pads)
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

    def __call__(self, x):
        with torch.no_grad():
            pp_x = self.preprocessor.preprocess(x)
            pp_x = pp_x.to(self.device)

            # TODO postprocess here by removing the padding added by the preprocessor

            return self.model(pp_x, mode='encoder')
