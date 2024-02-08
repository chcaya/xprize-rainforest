import re
from typing import List

import os
import pickle

import albumentations
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class TilesObjectDetectionDataset(Dataset):
    BACKGROUND_CLASS_ID = 0

    def __init__(self,
                 datasets_configs: List[dict],
                 transform: albumentations.Compose):
        """
        Args:
            datasets_configs (list of dicts): Each dict contains 'name' and 'data_root' keys for the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.samples = []  # Store tuples of (image_path, label_path, dataset_name)

        for dataset in datasets_configs:
            name = dataset['name']
            data_root = dataset['data_root']
            tiles_dir = os.path.join(data_root, 'tiles')
            labels_dir = os.path.join(data_root, 'labels')

            # Check if 'tiles' and 'labels' subfolders exist
            if not os.path.exists(tiles_dir) or not os.path.exists(labels_dir):
                raise FileNotFoundError(f"Missing 'tiles' or 'labels' directories in {data_root}")

            # List all files in the 'tiles' directory
            tile_files = [f for f in os.listdir(tiles_dir) if os.path.isfile(os.path.join(tiles_dir, f))]

            for tile_file in tile_files:
                tile_coords = re.search(r"(\d+_\d+)", tile_file)
                tile_path = os.path.join(tiles_dir, tile_file)
                label_filename = 'labels_' + tile_coords.group() + '.pkl'
                label_path = os.path.join(labels_dir, label_filename)
                self.samples.append((tile_path, label_path, name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path, dataset_name = self.samples[idx]

        # Load the image using PIL and convert it to RGB
        image = Image.open(img_path).convert("RGB")
        # Convert the PIL Image to a Float NumPy array for Albumentations
        image = np.array(image)

        # Load the bounding boxes
        with open(label_path, 'rb') as f:
            boxes = pickle.load(f)

        boxes = np.array(boxes, dtype=np.float32)
        labels = [self.BACKGROUND_CLASS_ID + 1] * len(boxes)

        # Apply transformations
        if self.transform:
            # This assumes self.transform is an Albumentations composition including ToTensorV2()
            transformed = self.transform(image=image,
                                         bboxes=boxes,
                                         labels=labels)  # Update labels accordingly
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']

        image = torch.Tensor(image).float() / 255
        image = image.permute(2, 0, 1)
        boxes = torch.Tensor(boxes)
        labels = torch.Tensor(labels).long()

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4))

        sample = {'image': image, 'boxes': boxes, 'labels': labels, 'dataset_name': dataset_name}

        return sample
