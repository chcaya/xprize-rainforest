import unittest
import numpy as np
from PIL import Image
import torch
from albumentations import Compose

from engine.embedder.siamese.transforms import AlbumentationsTorchWrapper, detector_transforms  # Update 'your_module' to your actual module name
from engine.embedder.siamese.torch_tranforms import (
    RandomChannelDropout, RandomSpectralTransform, GeometricDistortions,
    RandomNoiseInjection, PatchSwapping, ScaleChange, Solarization,
    HorizontalFlip, RandomRotationTranslation, RandomCutoutErasing, MultiCrop
)


class TestAlbumentationsTransforms(unittest.TestCase):

    def setUp(self):
        # Create a sample tensor (3-channel image)
        self.sample_image = torch.rand(3, 224, 224)  # Simulate a 3-channel RGB image with dimensions 224x224
        self.sample_image_numpy = self.sample_image.permute(1, 2, 0).detach().cpu().numpy()  # Simulate a 3-channel RGB image with dimensions 224x224

        self.pipeline_transforms = detector_transforms

    # Test to ensure the wrapper maintains image dimensions and type
    def test_wrapper_maintains_image_properties(self):
        torch_transform = RandomChannelDropout()
        wrapped_transform = AlbumentationsTorchWrapper(torch_transform, p=1.0)
        transformed_image = wrapped_transform(image=self.sample_image)
        transformed_image_torch = torch.from_numpy(transformed_image['image']).permute(2, 0, 1)

        assert transformed_image_torch.shape == self.sample_image.shape

    # Test to ensure the transformation logic is applied correctly
    def test_wrapper_applies_transformation_correctly(self):
        torch_transform = RandomChannelDropout(drop_prob=1.0) # drop all channels
        wrapped_transform = AlbumentationsTorchWrapper(torch_transform, p=1.0)
        transformed_image = wrapped_transform(image=self.sample_image)
        # Verify the transformation logic: check if colors are inverted
        assert np.array_equal(np.zeros_like(transformed_image['image']), transformed_image['image'])

    def test_wrapper_applies_transformation_correctly(self):
        torch_transform = RandomChannelDropout(drop_prob=1.0) # drop all channels
        wrapped_transform = AlbumentationsTorchWrapper(torch_transform, p=1.0)
        transformed_image = wrapped_transform(image=self.sample_image)
        # Verify the transformation logic: check if colors are inverted
        assert np.array_equal(np.zeros_like(transformed_image['image']), transformed_image['image'])

    def test_composed_pipeline_transforms_applied_correctly(self):
        wrapped_transform = Compose(self.pipeline_transforms)
        transformed_image = wrapped_transform(image = self.sample_image_numpy)
        # Verify the transformation logic: check if colors are inverted
        transformed_image_torch = torch.from_numpy(transformed_image['image']).permute(2, 0, 1)
        assert transformed_image_torch.shape == self.sample_image.shape
