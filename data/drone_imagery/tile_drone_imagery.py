from PIL import Image


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


def tile_image(image_path, tile_size):
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
            tiles.append(tile)

    return tiles


def save_tiles(tiles, tile_size):
    """
    Saves each tile to disk with an appropriate filename.
    """
    for idx, tile in enumerate(tiles):
        tile.save(f'tile_{idx + 1}_{tile_size}x{tile_size}.png')


# Example usage
image_path = 'path_to_your_image.jpg'  # Replace with your image path
optimal_tile_size = compute_optimal_tile_size(image_path)
tiles = tile_image(image_path, optimal_tile_size)
save_tiles(tiles, optimal_tile_size)

print(f"Optimal tile size: {optimal_tile_size}, Number of tiles: {len(tiles)}")
