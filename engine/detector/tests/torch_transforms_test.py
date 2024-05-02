import unittest
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/daoud/PycharmProjects/ssl_transforms/xprize-rainforest/')

from engine.detector.torch_tranforms import *

# Mock the random module for reproducibility
random.seed(0)
torch.manual_seed(0)




class TestCustomAlbumentationsWrapper(unittest.TestCase):
    def setUp(self):
        # Define a simple custom torch transform
        self.custom_transform = CustomTorchTransform(params)

        # Wrap the custom transform for Albumentations
        self.albumentations_wrapper = CustomAlbumentationsWrapper(self.custom_transform)

        # Create Albumentations pipeline with your custom transform
        self.transform = A.Compose([
            self.albumentations_wrapper,
            # Add more Albumentations transforms here if needed
        ])

    def test_transform_output(self):
        # Define a sample input image
        input_image = torch.rand(3, 256, 256)  # Example shape, adjust as per your input

        # Apply the transformation pipeline
        transformed_data = self.transform(image=input_image)

        # Extract the transformed image
        transformed_image = transformed_data['image']

        # Assert that the transformed image has the correct shape
        expected_shape = (3, 256, 256)  # Adjust as per your transformation logic
        self.assertEqual(transformed_image.shape, expected_shape)

    def test_transform_consistency(self):
        # Define a sample input image
        input_image = torch.rand(3, 256, 256)  # Example shape, adjust as per your input

        # Apply the transformation pipeline twice to check consistency
        transformed_data_1 = self.transform(image=input_image)
        transformed_data_2 = self.transform(image=input_image)

        # Extract the transformed images
        transformed_image_1 = transformed_data_1['image']
        transformed_image_2 = transformed_data_2['image']

        # Assert that the transformed images are identical (pipeline consistency)
        torch.testing.assert_allclose(transformed_image_1, transformed_image_2)


class TestImageTransforms(unittest.TestCase):

    def setUp(self):
        # Create a sample tensor (3-channel image)
        self.sample_image = torch.rand(3, 224, 224)  # Simulate a 3-channel RGB image with dimensions 224x224

    def visualize_transformation(self, original_image, transformed_image):
        """Visualize original and transformed images side by side."""
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image.permute(1, 2, 0))
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(transformed_image.permute(1, 2, 0))
        plt.title('Transformed Image')
        plt.show()

    def test_random_channel_dropout(self):
        # Instantiate the transform
        transform = RandomChannelDropout(drop_prob=1)  # Set drop_prob to 1 to ensure all channels are dropped
        # Apply the transform
        transformed_image = transform(self.sample_image)
        # Assert that all channels are zero
        self.assertTrue(torch.equal(transformed_image, torch.zeros_like(self.sample_image)))

        transform = RandomChannelDropout(drop_prob=0)  # Set drop_prob to 0 to ensure no channels are dropped
        transformed_image = transform(self.sample_image)
        self.assertTrue(torch.equal(transformed_image, self.sample_image))

    def test_random_spectral_transform(self):
        transform = RandomSpectralTransform(band_range=(0, 1))
        transformed_image = transform(self.sample_image)
        # Check if the transformed image has only up to 1 channel
        self.assertTrue(transformed_image.shape[0] <= 1)

    def test_geometric_distortions_identity(self):
        identity_transform = GeometricDistortions(degrees=0, translate=(0, 0), scale_ranges=(1, 1), shear=0)
        transformed_image = identity_transform(self.sample_image)
        # Check if the image remains unchanged when parameters are set to neutral
        self.assertTrue(torch.allclose(transformed_image, self.sample_image, atol=1e-4))
        self.assertEqual(transformed_image.shape, self.sample_image.shape)

    def test_geometric_distortions(self):
        transform = GeometricDistortions(degrees=2, translate=(0, 0), scale_ranges=(1, 1), shear=0)
        transformed_image = transform(self.sample_image)

        self.assertEqual(transformed_image.shape, self.sample_image.shape)
        self.assertFalse(torch.equal(transformed_image, self.sample_image))

    def test_random_noise_injection_identity(self):
        identity_transform = RandomNoiseInjection(noise_level=0)
        transformed_image = identity_transform(self.sample_image)
        # Check if the image remains unchanged with noise_level 0
        self.assertTrue(torch.equal(transformed_image, self.sample_image))

    def test_random_noise_injection(self):
        transform = RandomNoiseInjection(noise_level=0.2)
        transformed_image = transform(self.sample_image)
        # Check if the image remains unchanged with noise_level 0
        self.assertFalse(torch.equal(transformed_image, self.sample_image))

    def test_patch_swapping(self):
        transform = PatchSwapping(swap_size=50)
        transformed_image = transform(self.sample_image.clone())  # Use a clone to prevent in-place modifications
        # Check if the image is not exactly the same (since a patch swap should have occurred)
        self.assertFalse(torch.equal(transformed_image, self.sample_image))

    def test_scale_change(self):
        transform = ScaleChange(scale_range=(1, 1))  # No scaling applied
        transformed_image = transform(self.sample_image)
        # Check dimensions are unchanged
        self.assertEqual(transformed_image.shape, self.sample_image.shape)

    def test_solarization(self):
        transform = Solarization(threshold=0.5)  # A middle value where some pixels should change
        transformed_image = transform(self.sample_image)
        # Check if the transformation is applied correctly
        self.assertFalse(torch.equal(transformed_image, self.sample_image))

    def test_horizontal_flip(self):
        transform = HorizontalFlip()
        transformed_image = transform(self.sample_image)
        # Check if flipped correctly
        self.assertTrue(torch.equal(transformed_image, TF.hflip(self.sample_image)))

    def test_random_rotation_translation(self):
        transform = RandomRotationTranslation(degrees=0, translate=(0, 0))
        transformed_image = transform(self.sample_image)
        # Check if image remains unchanged with no rotation or translation
        self.assertTrue(torch.equal(transformed_image, self.sample_image))

    def test_random_cutout_erasing(self):
        transform = RandomCutoutErasing(scale=(0, 0), ratio=(1, 1))  # No erasing applied
        transformed_image = transform(self.sample_image)
        # Check if image remains unchanged
        self.assertTrue(torch.equal(transformed_image, self.sample_image))

    def test_multi_crop(self):
        transform = MultiCrop(large_crop_size=224, num_large_crops=1, small_crop_size=96, num_small_crops=1)
        transformed_images = transform(self.sample_image)
        # Check if the correct number of crops are returned
        self.assertEqual(len(transformed_images), 2)


if __name__ == '__main__':
    unittest.main()
