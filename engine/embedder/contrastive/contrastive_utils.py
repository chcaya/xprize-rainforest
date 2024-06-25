import numpy as np
import torch
import torch.nn.functional as F
import heapq

from torch import nn

from engine.embedder.contrastive.contrastive_model import XPrizeTreeEmbedder, XPrizeTreeEmbedder2

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# QPEB = Quebec Panama Equator Brazil
FOREST_QPEB_MEAN = np.array([0.463, 0.537, 0.363])
FOREST_QPEB_STD = np.array([0.207, 0.206, 0.162])


def scale_values(values, old_min, old_max, new_min, new_max):
    scaled_values = []
    for value in values:
        scaled_value = ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        scaled_values.append(scaled_value)
    return scaled_values


def normalize(data: np.array, mean: np.array, std: np.array):
    for channel in range(data.shape[0]):
        data[channel] = (data[channel] - mean[channel]) / std[channel]

    return data


def save_model(model, checkpoint_output_file):
    if isinstance(model, nn.DataParallel):
        # Save the original model which is wrapped inside `.module`
        actual_model = model.module
    else:
        # Directly save the model
        actual_model = model

    if isinstance(actual_model, XPrizeTreeEmbedder):
        torch.save(actual_model.state_dict(), checkpoint_output_file)
    elif isinstance(actual_model, XPrizeTreeEmbedder2):
        actual_model.save(checkpoint_output_file)
    else:
        raise NotImplementedError(f"Model type not supported for saving: {type(actual_model)}.")


def pad_and_stack_images(images):
    # check if all images have same sizes
    if len(set([img.shape[1:] for img in images])) == 1:
        return torch.stack(images)

    # Find the maximum height and width
    max_height = max([img.shape[1] for img in images])
    max_width = max([img.shape[2] for img in images])

    # Pad each image to the maximum height and width
    padded_images = []
    for img in images:
        height_diff = max_height - img.shape[1]
        width_diff = max_width - img.shape[2]

        # Calculate padding: (left, right, top, bottom)
        padding = (width_diff // 2, width_diff - (width_diff // 2), height_diff // 2, height_diff - (height_diff // 2))
        padded_img = F.pad(img, padding)
        padded_images.append(padded_img)

    return torch.stack(padded_images)


def contrastive_collate_fn(batch):
    images = [torch.Tensor(b[0]) for b in batch]
    months = torch.Tensor([b[1] for b in batch]).long()
    days = torch.Tensor([b[2] for b in batch]).long()
    labels_ids = torch.Tensor([b[3] for b in batch])
    labels = [b[4] for b in batch]
    families_ids = torch.Tensor([b[5] for b in batch])
    families = [b[6] for b in batch]

    images_padded = pad_and_stack_images(images)

    return images_padded, months, days, labels_ids, labels, families_ids, families

