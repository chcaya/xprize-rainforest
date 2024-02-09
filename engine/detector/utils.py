import numpy as np
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from shapely.geometry import Polygon
from rastervision.core.data import Scene
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def custom_collate_fn(batch):
    images = []
    boxes = []

    for x in batch:
        images.append(x['image'])
        boxes.append({
            'boxes': x['boxes'],
            'labels': x['labels']
        })

    return images, boxes


def display_image_with_boxes(image, boxes, box_format='xyxy'):
    """
    Displays an image with bounding boxes.

    Parameters:
    - image: an image.
    - boxes: A list of bounding boxes, each box specified as [x_min, y_min, x_max, y_max] for 'xyxy' format
             or [x_center, y_center, width, height] for 'cxcywh' format.
    - box_format: Format of the bounding boxes provided ('xyxy' or 'cxcywh'). Default is 'xyxy'.
    """

    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots(1, figsize=(15, 15))
    ax.imshow(image)

    # Plot each bounding box
    for box in boxes:
        if box_format == 'cxcywh':  # Convert from center-size format to corner format if necessary
            x_center, y_center, width, height = box
            x_min = x_center - width / 2
            y_min = y_center - height / 2
        else:  # Assuming 'xyxy' format
            x_min, y_min, x_max, y_max = box
            width, height = x_max - x_min, y_max - y_min

        # Create a Rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


def display_train_valid_test_aoi(train_scene: Scene, valid_scene: Scene, test_scene: Scene, show_image: bool, output_file: str or None):
    img = train_scene.raster_source[:, :]

    H, W = img.shape[:2]
    extent = Polygon.from_bounds(0, 0, W, H)
    bg = extent.difference(unary_union(train_scene.aoi_polygons))
    bg = bg if bg.geom_type == 'MultiPolygon' else [bg]

    fig, ax = plt.subplots(1, 1, squeeze=True, figsize=(8, 8))
    ax.imshow(img)

    for p in bg:
        p = mpatches.Polygon(np.array(p.exterior.coords), color='k', linewidth=2, alpha=0.5)
        ax.add_patch(p)

    for aoi in train_scene.aoi_polygons:
        p = mpatches.Polygon(np.array(aoi.exterior.coords), color='blue', linewidth=2, fill=False)
        ax.add_patch(p)

    for aoi in valid_scene.aoi_polygons:
        p = mpatches.Polygon(np.array(aoi.exterior.coords), color='green', linewidth=2, fill=False)
        ax.add_patch(p)

    for aoi in test_scene.aoi_polygons:
        p = mpatches.Polygon(np.array(aoi.exterior.coords), color='red', linewidth=2, fill=False)
        ax.add_patch(p)

    if output_file:
        fig.canvas.draw()
        plt.imsave(output_file, np.array(fig.canvas.renderer.buffer_rgba()))
    if show_image:
        plt.show()
    else:
        plt.close(fig)
