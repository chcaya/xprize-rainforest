import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
import numpy as np
from PIL import Image
from .torch_tranforms import (
    RandomChannelDropout, RandomSpectralTransform, GeometricDistortions,
    RandomNoiseInjection, PatchSwapping, ScaleChange, Solarization,
    HorizontalFlip, RandomRotationTranslation, RandomCutoutErasing, MultiCrop
)

# Wrapper class to use PyTorch transforms in Albumentations
class AlbumentationsTorchWrapper(ImageOnlyTransform):
    def __init__(self, torch_transform, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.torch_transform = torch_transform

    def apply(self, img, **params):
        img = Image.fromarray(img)  # Convert numpy array to PIL Image
        img = self.torch_transform(img)  # Apply torch transform
        return np.array(img)  # Convert PIL Image back to numpy array

A_Torch_Wrapper = AlbumentationsTorchWrapper
detector_transforms = [
            # drop permute layers
            A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.1),  # Equivalent to RandomChannelDropout
            A_Torch_Wrapper(RandomSpectralTransform()),

            # flips, standard image transforms
            A.HorizontalFlip(),
            A.VerticalFlip(),
            # A.ShiftScaleRotate(p=0.2),        # this can put boxes out of the image and then the training crashes on an image without boxes
            A.RandomBrightnessContrast(p=0.1),
            A.HueSaturationValue(hue_shift_limit=10, p=0.1),
            A.RGBShift(p=0.1),
            A.RandomGamma(p=0.1),
            A.Blur(p=0.1),
            A.GaussNoise(p=0.1),  # Equivalent to RandomNoiseInjection
            A.Resize(256, 256, p=1.0),  # Assuming ScaleChange is a resize
            # useful for geospatial
            A.functional.solarize(threshold=128),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.2),  # For RandomRotationTranslation
            A.Affine(shear={'x': (-15, 15), 'y': (-15, 15)}, p=0.5),  # Adding shear transformation
            A_Torch_Wrapper(PatchSwapping()),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.2),  # Similar to RandomCutoutErasing

            A.ToGray(p=0.02),
            A.ToSepia(p=0.02),
            # A_Torch_Wrapper(MultiCrop())

]
