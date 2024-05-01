import torch
import torchvision.transforms as T
import random


class RandomChannelDropout(torch.nn.Module):
    """
    Randomly drops entire channels (i.e., sets them to zero) from a given image tensor. This transform
    is useful for encouraging the model to learn robust features that do not rely on the presence of all channels.

    Parameters:
    - drop_prob (float): Probability of dropping any single channel.
    """

    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # x shape should be (C, H, W) where C is the number of channels
        for channel_idx in range(x.shape[0]):
            if random.random() < self.drop_prob:
                x[channel_idx] = 0  # Set the entire channel to zero
        return x


class RandomSpectralTransform(torch.nn.Module):
    """
    Applies a random spectral transformation by selecting random channels within a given range.
    """

    def __init__(self, band_range=(0, 10)):
        super().__init__()
        self.band_range = band_range

    def forward(self, x):
        start_band, end_band = self.band_range
        num_bands = x.shape[0]
        if num_bands > end_band:
            indices = torch.randperm(num_bands)[:end_band]
            return x[indices]
        return x


class GeometricDistortions(torch.nn.Module):
    """
    Applies random affine transformations including rotation, translation, scaling, and shearing.
    """

    def __init__(self, degrees=10, translate=(0.1, 0.1), scale_ranges=(0.9, 1.1), shear=5):
        super().__init__()
        self.degrees = degrees
        self.translate = translate
        self.scale_ranges = scale_ranges
        self.shear = shear

    def forward(self, x):
        angle = random.uniform(-self.degrees, self.degrees)
        translate = (random.uniform(-self.translate[0] * x.size(2), self.translate[0] * x.size(2)),
                     random.uniform(-self.translate[1] * x.size(1), self.translate[1] * x.size(1)))
        scale = random.uniform(*self.scale_ranges)
        shear = random.uniform(-self.shear, self.shear)
        return T.functional.affine(x, angle=angle, translate=translate, scale=scale, shear=shear,
                                   interpolation=T.InterpolationMode.BILINEAR)


class RandomNoiseInjection(torch.nn.Module):
    """
    Injects random noise into the image to simulate sensor noise and enhance robustness.
    """

    def __init__(self, noise_level=0.05):
        super().__init__()
        self.noise_level = noise_level

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise_level
        return x + noise


class PatchSwapping(torch.nn.Module):
    """
    Randomly swaps patches from within the same image or from geographically close images
    to help the model learn spatial relationships and context.
    """

    def __init__(self, swap_size=50):
        super().__init__()
        self.swap_size = swap_size

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]
        x1, y1 = random.randint(0, H - self.swap_size), random.randint(0, W - self.swap_size)
        x2, y2 = random.randint(0, H - self.swap_size), random.randint(0, W - self.swap_size)
        patch = x[:, x1:x1 + self.swap_size, y1:y1 + self.swap_size].clone()
        x[:, x1:x1 + self.swap_size, y1:y1 + self.swap_size] = x[:, x2:x2 + self.swap_size, y2:y2 + self.swap_size]
        x[:, x2:x2 + self.swap_size, y2:y2 + self.swap_size] = patch
        return x


class ScaleChange(torch.nn.Module):
    """
    Changes the scale of the image randomly within a specified range.
    """

    def __init__(self, scale_range=(0.8, 1.2)):
        super().__init__()
        self.scale_range = scale_range

    def forward(self, x):
        scale_factor = random.uniform(*self.scale_range)
        return T.functional.resize(x, [int(x.size(1) * scale_factor), int(x.size(2) * scale_factor)])


class Solarization(torch.nn.Module):
    """
    Applies solarization effect to the image.
    """

    def __init__(self, threshold=128):
        super().__init__()
        self.threshold = threshold

    def forward(self, img):
        return torch.where(img > self.threshold, 255 - img, img)


class HorizontalFlip(torch.nn.Module):
    """
    Flips the image horizontally.
    """

    def forward(self, img):
        return T.functional.hflip(img)


class RandomRotationTranslation(torch.nn.Module):
    """
    Applies random rotation and translation to the image.
    """

    def __init__(self, degrees=10, translate=(0.1, 0.1)):
        super().__init__()
        self.degrees = degrees
        self.translate = translate

    def forward(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        trans_x = random.uniform(-self.translate[0] * img.size(-1), self.translate[0] * img.size(-1))
        trans_y = random.uniform(-self.translate[1] * img.size(-2), self.translate[1] * img.size(-2))
        return T.functional.affine(img, angle=angle, translate=(trans_x, trans_y), scale=1, shear=0)


class RandomCutoutErasing(torch.nn.Module):
    """
    Randomly erases a portion of the image.
    """

    def __init__(self, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        super().__init__()
        self.scale = scale
        self.ratio = ratio

    def forward(self, img):
        i = random.randint(0, img.size(1) - 1)
        j = random.randint(0, img.size(2) - 1)
        h = int(random.uniform(self.scale[0], self.scale[1]) * img.size(1))
        w = int(random.uniform(self.scale[0], self.scale[1]) * img.size(2))
        return T.functional.erase(img, i, j, h, w, v=random.uniform(0, 1), inplace=False)


class MultiCrop(torch.nn.Module):
    """
    Creates multiple crops of the same image at different scales.
    """

    def __init__(self, large_crop_size, num_large_crops, small_crop_size, num_small_crops):
        super().__init__()
        self.large_crop_size = large_crop_size
        self.num_large_crops = num_large_crops
        self.small_crop_size = small_crop_size
        self.num_small_crops = num_small_crops

    def forward(self, img):
        crops = []
        for _ in range(self.num_large_crops):
            crops.append(T.functional.resize(T.functional.center_crop(img, min(img.size()) - 1), self.large_crop_size))
        for _ in range(self.num_small_crops):
            crops.append(T.functional.resize(T.functional.center_crop(img, min(img.size()) // 2), self.small_crop_size))
        return crops


# Compose all transformations, this is an example list
all_transforms = T.Compose([
    RandomChannelDropout(drop_prob=0.05),
    Solarization(),
    HorizontalFlip(),
    RandomRotationTranslation(),
    RandomCutoutErasing(),
    MultiCrop(large_crop_size=224, num_large_crops=2, small_crop_size=96, num_small_crops=4)
])
