from matplotlib import patches
from shapely import box

from shapely.geometry import Polygon
from skimage.measure import find_contours

import matplotlib.pyplot as plt
import numpy as np


def sam_collate_fn(single_image_batch):
    return single_image_batch[0]


def display_image_with_mask_and_box(image: np.ndarray, mask: np.ndarray, image_box: box, mask_alpha: float = 0.5,
                                    additional_polygons: list[Polygon] = None):
    """
    Displays a CHW image with a CHW mask, a bounding box, and optional additional polygons overlaid on top of it.

    Parameters:
    - image (np.ndarray): The image to display, in CHW format.
    - mask (np.ndarray): The mask to overlay, in CHW format and same dimensions as the image.
    - image_box (shapely.geometry.box): The bounding box to draw on the image.
    - mask_alpha (float): The transparency of the mask overlay. 0 is fully transparent, 1 is fully opaque.
    - additional_polygons (list[Polygon], optional): A list of shapely Polygon objects to plot.
    """
    if image.shape[1:] != mask.shape[1:]:
        raise ValueError("Image and mask must have the same height and width.")

    # Convert CHW to HWC for plotting
    image_hwc = np.transpose(image, (1, 2, 0))

    # Expand mask to HWC by repeating it across RGB channels, then create RGBA mask
    mask_hwc = np.repeat(mask, 3, axis=0)  # Repeat mask across channel dimension
    mask_hwc = np.transpose(mask_hwc, (1, 2, 0))  # Convert 3HW to HWC
    mask_rgba = np.concatenate([mask_hwc, np.ones((mask_hwc.shape[0], mask_hwc.shape[1], 1)) * mask_alpha], axis=2)

    # Display image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_hwc)
    ax.imshow(mask_rgba, interpolation='none')

    # Draw the bounding box
    rect = patches.Rectangle((image_box.bounds[0], image_box.bounds[1]), image_box.bounds[2] - image_box.bounds[0],
                             image_box.bounds[3] - image_box.bounds[1], linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Plot additional polygons if any
    if additional_polygons:
        for polygon in additional_polygons:
            # Extract exterior coordinates of polygon
            x, y = polygon.exterior.xy
            ax.plot(x, y, color="yellow", linewidth=3)  # Customize color and linewidth as needed

            # Optionally, handle polygon interiors (holes) if necessary
            for interior in polygon.interiors:
                x, y = interior.xy
                ax.plot(x, y, color="yellow", linewidth=3)  # Draw interiors with the same styling

    plt.axis('off')
    plt.show()
