from pathlib import Path

import rasterio
from PIL import Image
from geodataset.geodata import Raster
import tifffile
from matplotlib import pyplot as plt
from rasterio.windows import Window
from tqdm import tqdm

if __name__ == "__main__":
    import cv2
    import numpy as np
    Image.MAX_IMAGE_PIXELS = None

    ground_resolution = 0.03

    image_path = 'C:/Users/Hugo/Documents/XPrize/data/raw/20240521_zf2100ha_highres_m3m_rgb.tif'  # replace with your image path

    # image = tifffile.imread('C:/Users/Hugo/Documents/XPrize/data/raw/20240521_zf2100ha_highres_m3m_rgb.tif')

    with rasterio.open(image_path) as src:
        image = src.read()  # Read the image data
        profile = src.profile  # Read the profile (metadata)
        crs = src.crs  # Coordinate Reference System
        transform = src.transform  # Affine transform

    print(image.shape)

    # The image is read in as (channels, height, width), we need to transpose to (height, width, channels)
    image = np.transpose(image, (1, 2, 0))

    # Convert PIL image to numpy array
    image_np = np.array(image)

    print(image_np.shape)

    image_rgb = image_np[:, :, :3]
    image_alpha = image_np[:, :, 3]

    print('Converting to 8-bit...')
    if image_rgb.dtype != np.uint8:
        image_rgb = cv2.convertScaleAbs(image_rgb, alpha=(255.0 / 65535.0))
    if image_alpha.dtype != np.uint8:
        image_alpha = cv2.convertScaleAbs(image_alpha, alpha=(255.0 / 65535.0))

    # Convert RGB to LAB
    print('Converting to LAB...')
    lab_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_image)

    l_resized = cv2.resize(l, (1000, 1000), interpolation=cv2.INTER_AREA)
    a_resized = cv2.resize(a, (1000, 1000), interpolation=cv2.INTER_AREA)
    b_resized = cv2.resize(b, (1000, 1000), interpolation=cv2.INTER_AREA)

    # Display the resized 'l', 'a', and 'b' channels side by side
    fig, axes = plt.subplots(5, 3, figsize=(12, 12))

    axes[0, 0].imshow(l_resized, cmap='gray')
    axes[0, 0].set_title('Resized L Channel')
    axes[0, 0].axis('off')  # Hide the axes

    axes[0, 1].imshow(a_resized, cmap='gray')
    axes[0, 1].set_title('Resized A Channel')
    axes[0, 1].axis('off')  # Hide the axes

    axes[0, 2].imshow(b_resized, cmap='gray')
    axes[0, 2].set_title('Resized B Channel')
    axes[0, 2].axis('off')  # Hide the axes

    print(np.sum(l == 0), np.sum(image_rgb[:, :, 0] == 0), np.sum(image_rgb[:, :, 1] == 0), np.sum(image_rgb[:, :, 2] == 0), np.sum(image_alpha == 0), np.sum(image_alpha != 0))
    print(np.mean(l), np.mean(l[l != 0]))

    l_mask = image_alpha == 0       # we want to avoid as much as possible transparent pixels impacting clahe
    l_mean = np.mean(l[image_alpha != 0])   # we set transparent pixels luminosity to the mean of non-transparent pixels
    original_l = l.copy()
    l[l_mask] = l_mean

    l_resized = cv2.resize(l, (1000, 1000), interpolation=cv2.INTER_AREA)
    axes[2, 0].imshow(l_resized, cmap='gray')
    axes[2, 0].set_title('Resized L Channel')
    axes[2, 0].axis('off')  # Hide the axes

    print('Applying CLAHE...')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64, 64))
    cl3 = clahe.apply(l)

    cl3[l_mask] = original_l[l_mask]

    cl3_resized = cv2.resize(cl3, (1000, 1000), interpolation=cv2.INTER_AREA)

    axes[1, 2].imshow(cl3_resized, cmap='gray')
    axes[1, 2].set_title('Resized cl3 Channel')
    axes[1, 2].axis('off')  # Hide the axes

    cl3_mean = np.mean(cl3[image_alpha != 0])
    cl3_adjusted = np.clip(cl3 + (l_mean - cl3_mean), 0, 255).astype(np.uint8)

    cl3_adjusted_resized = cv2.resize(cl3_adjusted, (1000, 1000), interpolation=cv2.INTER_AREA)
    axes[3, 0].imshow(cl3_adjusted_resized, cmap='gray')
    axes[3, 0].set_title('Resized cl3_adjusted Channel')
    axes[3, 0].axis('off')  # Hide the axes


    def compute_local_mean(channel, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        local_mean = cv2.filter2D(channel, -1, kernel)
        return local_mean

    print('Computing A band changes...')
    global_mean_a = np.mean(a[image_alpha != 0])
    global_mean_b = np.mean(b[image_alpha != 0])

    local_mean_a1 = compute_local_mean(a, kernel_size=1536)
    a_adjusted1 = a - (local_mean_a1 - global_mean_a)
    a_adjusted1 = np.clip(a_adjusted1, 0, 255).astype(np.uint8)

    local_mean_b1 = compute_local_mean(b, kernel_size=1536)
    b_adjusted1 = b - (local_mean_b1 - global_mean_b)
    b_adjusted1 = np.clip(b_adjusted1, 0, 255).astype(np.uint8)


    #
    # local_mean_a2 = compute_local_mean(a, kernel_size=3072)
    # a_adjusted2 = a - (local_mean_a2 - global_mean_a)
    # a_adjusted2 = np.clip(a_adjusted2, 0, 255).astype(np.uint8)

    # local_mean_a3 = compute_local_mean(a, kernel_size=4096)
    # a_adjusted3 = a - (local_mean_a3 - global_mean_a)
    # a_adjusted3 = np.clip(a_adjusted3, 0, 255).astype(np.uint8)
    #
    # local_mean_a4 = compute_local_mean(a, kernel_size=6144)
    # a_adjusted4 = a - (local_mean_a4 - global_mean_a)
    # a_adjusted4 = np.clip(a_adjusted4, 0, 255).astype(np.uint8)

    a_adjusted1_resized = cv2.resize(a_adjusted1, (4000, 4000), interpolation=cv2.INTER_AREA)
    b_adjusted1_resized = cv2.resize(b_adjusted1, (4000, 4000), interpolation=cv2.INTER_AREA)

    local_mean_a1_resized = cv2.resize(local_mean_a1, (4000, 4000), interpolation=cv2.INTER_AREA)
    local_mean_b1_resized = cv2.resize(local_mean_b1, (4000, 4000), interpolation=cv2.INTER_AREA)

    # a_adjusted2_resized = cv2.resize(a_adjusted2, (1000, 1000), interpolation=cv2.INTER_AREA)
    # a_adjusted3_resized = cv2.resize(a_adjusted3, (1000, 1000), interpolation=cv2.INTER_AREA)
    # a_adjusted4_resized = cv2.resize(a_adjusted4, (1000, 1000), interpolation=cv2.INTER_AREA)

    plt.figure(figsize=(12, 12))
    plt.imshow(a_adjusted_resized, cmap='gray')
    plt.title('Adjusted A Channel')
    plt.axis('off')  # Hide the axes
    plt.show()

    plt.figure(figsize=(12, 12))
    plt.imshow(local_mean_a_resized, cmap='gray')
    plt.title('Adjusted A Channel')
    plt.axis('off')  # Hide the axes
    plt.show()

    plt.figure(figsize=(12, 12))
    plt.imshow(a_adjusted1_resized, cmap='gray')
    plt.title('Adjusted A1 Channel')
    plt.axis('off')  # Hide the axes
    plt.show()

    plt.figure(figsize=(12, 12))
    plt.imshow(local_mean_a1_resized, cmap='gray')
    plt.title('Adjusted A1 Channel')
    plt.axis('off')  # Hide the axes
    plt.show()

    plt.figure(figsize=(12, 12))
    plt.imshow(b_adjusted1_resized, cmap='gray')
    plt.title('Adjusted B1 Channel')
    plt.axis('off')  # Hide the axes
    plt.show()

    plt.figure(figsize=(12, 12))
    plt.imshow(local_mean_b1_resized, cmap='gray')
    plt.title('Adjusted B1 Channel')
    plt.axis('off')  # Hide the axes
    plt.show()



    #
    # axes[3, 2].imshow(a_adjusted2_resized, cmap='gray')
    # axes[3, 2].set_title('Adjusted A2 Channel')
    # axes[3, 2].axis('off')  # Hide the axes
    #
    # axes[4, 0].imshow(a_adjusted3_resized, cmap='gray')
    # axes[4, 0].set_title('Adjusted A3 Channel')
    # axes[4, 0].axis('off')  # Hide the axes
    #
    # axes[4, 1].imshow(a_adjusted4_resized, cmap='gray')
    # axes[4, 1].set_title('Adjusted A4 Channel')
    # axes[4, 1].axis('off')  # Hide the axes

    # plt.show()

    merged_lab_image = cv2.merge((cl3_adjusted, a_adjusted1, b_adjusted1))

    print('Converting back to RGB...')
    final_image = cv2.cvtColor(merged_lab_image, cv2.COLOR_LAB2RGB)

    print('Recombining with alpha channel...')
    final_image = np.dstack((final_image, image_alpha))

    final_image = np.transpose(final_image, (2, 0, 1))

    profile.update(dtype=rasterio.uint8,
                   compress='deflate',
                   tiled=True,
                   blockxsize=512,
                   blockysize=512,)
    profile.update(BIGTIFF="YES")

    print(final_image.shape)
    print(final_image.dtype)

    if np.isnan(final_image).any() or np.isinf(final_image).any():
        raise ValueError("The final image contains NaNs or infinities.")

    if np.any(final_image < 0) or np.any(final_image > 255):
        raise ValueError("The final image contains values outside the range [0, 255].")

    # with rasterio.open('D:/XPrize/Data/clahe/20240521_zf2100ha_highres_m3m_rgb_clahe.tif', 'w', **profile) as dst:
    #     dst.write(final_image)

    chunk_size = 512

    with rasterio.open('D:/XPrize/Data/clahe/20240521_zf2100ha_highres_m3m_rgb_clahe_no_black_adjustedL_adjustedA768_adjustedB768.tif', 'w', **profile) as dst:
        height, width = final_image.shape[1], final_image.shape[2]

        for i in tqdm(range(0, width, chunk_size), desc="Writing clahe-modified raster by chunks..."):
            for j in range(0, height, chunk_size):
                window_width = min(chunk_size, width - i)
                window_height = min(chunk_size, height - j)
                window = Window(i, j, window_width, window_height)

                data_chunk = final_image[:, j:j + window_height, i:i + window_width]
                dst.write(data_chunk, window=window)
