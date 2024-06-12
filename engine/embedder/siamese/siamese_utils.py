import torch
import torch.nn.functional as F


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


def train_collate_fn(batch):
    imgs1 = [torch.Tensor(b[0]) for b in batch]
    imgs2 = [torch.Tensor(b[1]) for b in batch]
    labels = torch.Tensor([b[2] for b in batch])
    margins = torch.Tensor([b[3] for b in batch])

    imgs1_padded = pad_images(imgs1)
    imgs2_padded = pad_images(imgs2)

    return imgs1_padded, imgs2_padded, labels, margins


def valid_collate_fn(batch):
    images = [torch.Tensor(b[0]) for b in batch]
    labels = torch.Tensor([b[1] for b in batch])

    images_padded = pad_images(images)

    return images_padded, labels
