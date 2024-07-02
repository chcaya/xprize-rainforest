import glob
import os.path

from PIL import Image
from pathlib import Path

def is_non_invisible_pixel(pixel, threshold=0):
    """
    Checks if a pixel is non-invisible based on a threshold.
    """
    return pixel[3] > threshold  # Assuming the image has an alpha channel

def get_non_invisible_percentage(tile):
    """
    Calculates the percentage of non-invisible pixels in a tile.
    """
    pixels = tile.getdata()
    non_invisible_pixels = sum(1 for pixel in pixels if is_non_invisible_pixel(pixel))
    return (non_invisible_pixels / len(pixels)) * 100

def compute_optimal_tile_size(image_path, min_size=300, max_size=500):
    """
    Computes the optimal tile size within a specified range that maximizes the covered area.
    """
    img = Image.open(image_path)
    width, height = img.size
    optimal_tile_size = None
    max_covered_area = 0

    for tile_size in range(min_size, max_size + 1):
        num_tiles_horizontal = width // tile_size
        num_tiles_vertical = height // tile_size
        covered_area = num_tiles_horizontal * num_tiles_vertical * tile_size * tile_size

        if covered_area > max_covered_area:
            max_covered_area = covered_area
            optimal_tile_size = tile_size

    return optimal_tile_size


def tile_image(image_path, tile_size, non_invisible_threshold=50):
    """
    Tiles an image using the specified tile size.
    """
    img = Image.open(image_path)
    width, height = img.size
    tiles = []
    num_tiles_horizontal = width // tile_size
    num_tiles_vertical = height // tile_size

    for i in range(num_tiles_vertical):
        for j in range(num_tiles_horizontal):
            left = j * tile_size
            upper = i * tile_size
            right = left + tile_size
            bottom = upper + tile_size
            tile = img.crop((left, upper, right, bottom))
            non_invisible_percentage = get_non_invisible_percentage(tile)
            if non_invisible_percentage >= non_invisible_threshold:
                tiles.append(tile)

    return tiles


def save_tiles(tiles, tile_size, file_prefix, output_dir):
    """
    Saves each tile to disk with an appropriate filename.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, tile in enumerate(tiles):
        tile.save(output_dir / f'{file_prefix}_tile_{idx + 1}_{tile_size}x{tile_size}.png')


if __name__ == "__main__":
    dir_path = Path('/Users/daoud/PycharmAssets/xprize/')
    # todo: convert to args
    drone_photos_path = str(f"dji/3cm/type3")
    image_glob_search_string = str(dir_path / drone_photos_path) + '/*'
    images_paths = glob.glob(image_glob_search_string)
    tile_size = 80

    for image_path in images_paths:
        file_name = image_path.split('/')[-1].split('.')[0]
        tiles = tile_image(image_path, tile_size, non_invisible_threshold=60)
        save_tiles(tiles, tile_size, file_prefix = file_name, output_dir= dir_path / drone_photos_path / f"{file_name}_cropped/")


