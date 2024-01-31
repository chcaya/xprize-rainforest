import numpy as np
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from shapely.geometry import Polygon
from rastervision.core.data import Scene


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
