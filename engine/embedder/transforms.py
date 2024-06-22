import albumentations as A
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np

from engine.embedder.siamese.torch_tranforms import (
    RandomChannelDropout, RandomSpectralTransform, GeometricDistortions,
    RandomNoiseInjection, PatchSwapping, ScaleChange, Solarization,
    HorizontalFlip, RandomRotationTranslation, RandomCutoutErasing, MultiCrop
)


class AlbumentationsTorchWrapper(ImageOnlyTransform):
    """
    Note: this currently supports 3 and 4 channel imagery but not more channels at present,
     it can be extended in the future if needed
    """

    def __init__(self, torch_transform, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.torch_transform = torch_transform

    def apply(self, img, **params):

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.transpose(2, 0, 1))  # Reorder numpy dimensions to CxHxW for torch

        img = self.torch_transform(img)

        if img.dim() == 3:
            img = img.permute(1, 2, 0)  # Convert CxHxW back to HxWxC

        return img.detach().cpu().numpy()  # Ensure it is on CPU and detach from computation graph


A_Torch_Wrapper = AlbumentationsTorchWrapper

embedder_transforms = [
    # drop permute layers
    A_Torch_Wrapper(RandomChannelDropout(drop_prob=0.02)),
    # A_Torch_Wrapper(RandomSpectralTransform()),

    # flips, standard image transforms
    A.HorizontalFlip(),
    A.VerticalFlip(),
    # A.ShiftScaleRotate(p=0.2), #can put boxes out of the image and then the training crashes on an image without boxes
    A.RandomBrightnessContrast(p=0.1),
    A.HueSaturationValue(hue_shift_limit=10, p=0.1),
    A.RGBShift(p=0.1),
    A.RandomGamma(p=0.1),
    A.Blur(p=0.1),
    A.GaussNoise(p=0.1),  # Equivalent to RandomNoiseInjection
    # A.Resize(256, 256, p=1.0),  # Assuming ScaleChange is a resize
    # useful for geospatial
    # A_Torch_Wrapper(Solarization(threshold=128)),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.2),  # For RandomRotationTranslation
    A.Affine(shear={'x': (-15, 15), 'y': (-15, 15)}, p=0.5),  # Adding shear transformation
    # A_Torch_Wrapper(PatchSwapping()),     # causes errors
    A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.2),  # Similar to RandomCutoutErasing

    A.ToGray(p=0.02),
    A.ToSepia(p=0.02),
    # A_Torch_Wrapper(MultiCrop())

]
