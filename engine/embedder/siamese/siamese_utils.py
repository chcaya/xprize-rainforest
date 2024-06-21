import numpy as np
import torch
import torch.nn.functional as F
import heapq

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# QPEB = Quebec Panama Equator Brazil
FOREST_QPEB_MEAN = np.array([0.463, 0.537, 0.363])
FOREST_QPEB_STD = np.array([0.207, 0.206, 0.162])


class LimitedSizeHeap:
    def __init__(self, max_size):
        self.max_size = max_size
        self.data = []

    def add(self, value):
        if len(self.data) < self.max_size:
            heapq.heappush(self.data, value)
        else:
            heapq.heappushpop(self.data, value)

    def get_max_value(self):
        return self.data[0]

    def get_data(self):
        return sorted(self.data)


# Example usage
def normalize_non_black_pixels(data: np.array, mean: np.array, std: np.array):
    assert data.shape[0] == 3, 'Please make sure that the RGB channel is the first dimension of the image array.'

    # Create a mask where black pixels are marked as True
    mask = np.all(data == 0, axis=0)

    # Normalize only the non-black pixels
    for channel in range(data.shape[0]):
        channel_data = data[channel]
        channel_mask = ~mask

        # Normalize only the non-black pixels
        channel_data[channel_mask] = (channel_data[channel_mask] - mean[channel]) / std[channel]

        # Put the data back into the image array
        data[channel] = channel_data

    return data


def normalize(data: np.array, mean: np.array, std: np.array):
    for channel in range(data.shape[0]):
        data[channel] = (data[channel] - mean[channel]) / std[channel]

    return data


def pad_images(images):
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


def scale_values(values, old_min, old_max, new_min, new_max):
    scaled_values = []
    for value in values:
        scaled_value = ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        scaled_values.append(scaled_value)
    return scaled_values


def train_collate_fn(batch):
    imgs1 = [torch.Tensor(b[0]) for b in batch]
    imgs2 = [torch.Tensor(b[1]) for b in batch]
    labels = torch.Tensor([b[2] for b in batch])
    margins = torch.Tensor([b[3] for b in batch])

    imgs1_padded = pad_images(imgs1)
    imgs2_padded = pad_images(imgs2)

    return imgs1_padded, imgs2_padded, labels, margins


def train_collate_fn2(batch):
    imgs1 = [torch.Tensor(b[0]) for b in batch]
    imgs2 = [torch.Tensor(b[1]) for b in batch]
    months1 = torch.Tensor([b[2] for b in batch])
    months2 = torch.Tensor([b[3] for b in batch])
    days1 = torch.Tensor([b[4] for b in batch])
    days2 = torch.Tensor([b[5] for b in batch])
    labels = torch.Tensor([b[6] for b in batch])
    margins = torch.Tensor([b[7] for b in batch])

    imgs1_padded = pad_images(imgs1)
    imgs2_padded = pad_images(imgs2)

    return imgs1_padded, imgs2_padded, months1, months2, days1, days2, labels, margins


def valid_collate_fn(batch):
    images = [torch.Tensor(b[0]) for b in batch]
    labels = torch.Tensor([b[1] for b in batch])

    images_padded = pad_images(images)

    return images_padded, labels


def valid_collate_fn2(batch):
    images = [torch.Tensor(b[0]) for b in batch]
    months = torch.Tensor([b[1] for b in batch])
    days = torch.Tensor([b[2] for b in batch])
    labels = torch.Tensor([b[3] for b in batch])

    images_padded = pad_images(images)

    return images_padded, months, days, labels


def valid_collate_fn_string_labels(batch):
    images = [torch.Tensor(b[0]) for b in batch]
    months = torch.Tensor([b[1] for b in batch])
    days = torch.Tensor([b[2] for b in batch])
    # labels are string so only store them in a list
    labels = [b[3] for b in batch]

    images_padded = pad_images(images)

    return images_padded, months, days, labels

