import logging
import os.path
from typing import Dict, List
from pathlib import Path
from PIL import Image
from io import BytesIO
import pandas as pd
import requests
import asyncio
import aiohttp
import aiofiles

from api_utils import fetch_species_details, fetch_research_grade_image_urls, download_image

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_images(image_urls: list, output_dir: Path) -> None:
    """Download and save images from a list of URLs."""
    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            image.save(output_dir / f'species_image_{i + 1}.jpg')
            logging.info(f'Downloaded image {i + 1}')
        except requests.RequestException as e:
            logging.error(f"Failed to download image: {e}")


async def download_image_async(session, url, output_dir, file_name_prefix, index):
    try:
        async with session.get(url) as response:
            response.raise_for_status()  # Ensure we notice bad responses
            # Read the content
            image_data = await response.read()
            image = Image.open(BytesIO(image_data))
            # Asynchronously save the image using an executor
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, save_image, image, output_dir, file_name_prefix, index)
            logging.info(f'Downloaded image {index + 1}')
    except Exception as e:
        logging.error(f"Failed to download image {index + 1}: {e}")


def save_image(image, output_dir, file_name_prefix, index):
    """Save an image file synchronously."""
    with open(output_dir / f'{file_name_prefix}_{index + 1}.jpg', 'wb') as f:
        image.save(f, format='JPEG')


async def download_all_images(image_urls: List[str], output_dir: Path, folder_name: str, file_name_prefix: str) -> None:
    image_dir = output_dir / folder_name
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    async with aiohttp.ClientSession() as session:
        tasks = [download_image_async(session, url, image_dir, file_name_prefix, i) for i, url in enumerate(image_urls)]
        await asyncio.gather(*tasks)


def build_taxon_keys_dict(row) -> Dict[str, str]:
    """Builds a dictionary of taxon keys from a DataFrame row."""
    keys = ['speciesKey', 'genusKey', 'familyKey', 'orderKey', 'classKey', 'phylumKey', 'kingdomKey']
    return {key: int(row[key]) for key in keys if key in row}


# todo: save metadata of image along with image itself
def process_species_images(csv_file_path: Path, image_output_dir: Path, num_rows: int = 1) -> None:
    # read and process csv
    data = pd.read_csv(csv_file_path)
    float_cols = data.select_dtypes(include='float').columns
    # drop duplicates based on family key
    data = data.dropna(subset=['speciesKey'])
    data = data.drop_duplicates(subset='familyKey', keep='first')
    data[float_cols] = data[float_cols].fillna(-1).astype(int)

    for index, row in data.head(num_rows).iterrows():

        family_key = row["familyKey"]
        genus_key = row["genusKey"]
        species_key = row['speciesKey']

        folder_name = f'{family_key}_{genus_key}_{species_key}'

        if os.path.exists(image_output_dir / folder_name):
            logging.info(f"Found existing folder. Skipping downloads for speciesKey {species_key}.")
            continue

        logging.info(f'Processing speciesKey: {species_key}')
        species_details = fetch_species_details(species_key)
        if species_details:
            if verify_and_log_details(index, row, species_details):
                taxon_keys = build_taxon_keys_dict(row)
                image_urls = fetch_research_grade_image_urls(taxon_keys, limit=10)

                # todo: refactor to separate function
                if image_urls:
                    logging.info(f"Found {len(image_urls)} research-grade images for {species_key}: {image_urls}")
                    asyncio.run(download_all_images(image_urls=image_urls,
                                                    output_dir=image_output_dir,
                                                    folder_name=folder_name,
                                                    file_name_prefix=species_key))
                    # for i, url in enumerate(image_urls):
                    #     file_name = f"{species_key}_{i + 1}.jpg"
                    #     download_image(url, image_output_dir, folder_name, file_name)
                    logging.info(f"Downloaded {len(image_urls)} images for speciesKey {species_key}.")
                else:
                    logging.warning(f"No research-grade images found for {taxon_keys}.")
        else:
            logging.warning(f"No details found for speciesKey {row['speciesKey']} at row {index}.")


def build_keys_dict(row) -> dict:
    """Builds a dictionary of expected keys from a DataFrame row."""
    keys_dict = {
        'kingdomKey': row['kingdomKey'],
        'familyKey': row['familyKey'],
        'genusKey': row['genusKey'],
        'classKey': row['classKey'],
        'orderKey': row['orderKey'],
        'speciesKey': row['speciesKey']
    }
    return keys_dict


def verify_keys(data: Dict, **expected_keys) -> bool:
    """Verify if the specified keys in the data match the expected values."""
    for key, value in expected_keys.items():
        if str(data.get(key, '')) != str(value):
            logging.info(f"Mismatch in {key}: API returned {data.get(key)} but expected {value}")
            return False
    return True


def verify_and_log_details(index, row, species_details):
    """Build and verify the keys dictionary, then log the results."""
    keys_to_check = build_keys_dict(row)
    if verify_keys(species_details, **keys_to_check):
        logging.info(f"Verification passed for row {index + 1}.")
        return True
    else:
        logging.warning(f"Verification failed for row {index + 1}.")
        return False


# Example usage
if __name__ == "__main__":
    dir_path = Path('/Users/daoud/PycharmAssets/xprize/')
    brazil_gbif_species_csv = 'brazil_trees_amazonia_sel_occs.csv'
    brazil_photos_taxonomy_csv = 'photos_exif_taxo.csv'

    brazil_gbif_species_csv_path = dir_path / brazil_gbif_species_csv
    brazil_photos_taxonomy_csv_path = dir_path / brazil_photos_taxonomy_csv

    # brazil_gbif_species = pd.read_csv(brazil_gbif_species_csv_path)
    brazil_photos_taxonomy = pd.read_csv(brazil_photos_taxonomy_csv_path)

    image_output_dir = dir_path / 'images/'

    process_species_images(brazil_photos_taxonomy_csv_path, image_output_dir, num_rows=100)
