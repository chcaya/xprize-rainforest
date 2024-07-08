import os
from pathlib import Path

import numpy as np
import rasterio
from geodataset.geodata import Raster
from tqdm import tqdm


def find_tif_files(root_paths):
    """Recursively find all .tif files in the given root paths."""
    tif_files = []
    for root_path in root_paths:
        for root, _, files in os.walk(root_path):
            for file in files:
                if file.endswith('.tif') and 'dsm' not in file and 'dtm' not in file:
                    tif_files.append(os.path.join(root, file))
    return tif_files


def compute_mean_variance(tif_files, ground_resolution):
    """Compute the mean and variance of all .tif files."""
    sum_pixels = np.zeros(3)
    sum_squared_pixels = np.zeros(3)
    total_pixels = 0

    for tif_file in tqdm(tif_files):
        raster = Raster(Path(tif_file), ground_resolution=ground_resolution)

        data = raster.data[:3]
        alpha_channel = data[3] if data.shape[0] == 4 else None  # Extract alpha channel if present

        # Create a mask for non-border pixels
        if alpha_channel is not None:
            mask = np.all(data == 0, axis=0) | (alpha_channel == 0)
        else:
            mask = np.all(data == 0, axis=0)

        non_empty_pixels_mask = ~mask

        data = data.reshape(3, -1)  # Reshape to (channels, total_pixels_per_image)
        non_empty_pixels_mask = non_empty_pixels_mask.flatten()

        # Filter out border pixels
        filtered_data = data[:, non_empty_pixels_mask]

        filtered_data = filtered_data / 255.0  # Scale to [0, 1]
        raster_sum_pixels = filtered_data.sum(axis=1)
        raster_sum_squared_pixels = (filtered_data ** 2).sum(axis=1)
        raster_total_pixels = filtered_data.shape[1]

        sum_pixels += raster_sum_pixels
        sum_squared_pixels += raster_sum_squared_pixels
        total_pixels += raster_total_pixels

        print(f"Computed mean and variance for {tif_file}: {raster_sum_pixels / raster_total_pixels}, {(raster_sum_squared_pixels / raster_total_pixels) - (raster_sum_pixels / raster_total_pixels) ** 2}")
        print(f"Total computed mean and variance: {sum_pixels / total_pixels}, {(sum_squared_pixels / total_pixels) - (sum_pixels / total_pixels) ** 2}")

    mean = sum_pixels / total_pixels
    variance = (sum_squared_pixels / total_pixels) - (mean ** 2)
    std_dev = np.sqrt(variance)

    return mean, variance, std_dev


if __name__ == "__main__":
    ground_resolution = 0.03

    root_paths = ['/media/hugobaudchon/4 TB/XPrize/Data/raw/quebec_trees_dataset',
                  '/media/hugobaudchon/4 TB/XPrize/Data/raw/panama',
                  '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator',
                  '/media/hugobaudchon/4 TB/XPrize/Data/raw/brazil_zf2']  # Add your folders here
    tif_files = find_tif_files(root_paths)
    mean, variance, std_dev = compute_mean_variance(tif_files, ground_resolution)

    print("Computed Mean:", mean)
    print("Computed Variance:", variance)
    print("Computed Standard Deviation:", std_dev)
