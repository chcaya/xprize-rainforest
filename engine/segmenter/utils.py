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


def mask_to_polygon(mask: np.ndarray, simplify_tolerance: float = 1.0) -> Polygon:
    """
    Converts a 1HW mask to a simplified shapely Polygon by finding the contours of the mask
    and simplifying it.

    Parameters:
    - mask (np.ndarray): The mask to convert, in 1HW format.
    - simplify_tolerance (float): The tolerance for simplifying the polygon. Higher values
      result in more simplified shapes.

    Returns:
    - Polygon: A simplified shapely Polygon object representing the outer boundary of the mask.
    """
    # Ensure mask is 2D
    if mask.ndim != 2:
        raise ValueError("Mask must be in HW format (2D array).")

    # Pad the mask to avoid boundary issues
    padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)

    # Find contours on the mask, assuming mask is binary
    contours = find_contours(padded_mask, 0.5)

    if len(contours) == 0:
        # returning empty, dummy polygon at 0,0
        return Polygon([(0, 0), (0, 0), (0, 0)])

    # Take the longest contour as the main outline of the object
    longest_contour = max(contours, key=len)

    # Convert contour coordinates from (row, column) to (x, y)
    # and revert the padding added to the mask
    longest_contour_adjusted_xy = [(y - 1, x - 1) for x, y in longest_contour]

    # Convert contour to Polygon
    polygon = Polygon(longest_contour_adjusted_xy)

    # Simplify the polygon
    simplified_polygon = polygon.simplify(tolerance=simplify_tolerance, preserve_topology=True)

    return simplified_polygon
