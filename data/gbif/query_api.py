import logging
from typing import Dict
from pathlib import Path
from PIL import Image
from io import BytesIO
import pandas as pd
import requests

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


def build_taxon_keys_dict(row) -> Dict[str, str]:
    """Builds a dictionary of taxon keys from a DataFrame row."""
    keys = ['speciesKey', 'genusKey', 'familyKey', 'orderKey', 'classKey', 'phylumKey', 'kingdomKey']
    return {key: row[key] for key in keys if key in row}


def process_species_images(csv_file_path: Path, image_output_dir: Path, num_rows: int = 1) -> None:
    data = pd.read_csv(csv_file_path)
    for index, row in data.head(num_rows).iterrows():

        species_key = row['speciesKey']
        logging.info(f'Processing speciesKey: {species_key}')
        species_details = fetch_species_details(species_key)
        if species_details:
            if verify_and_log_details(index, row, species_details):
                taxon_keys = build_taxon_keys_dict(row)
                image_urls = fetch_research_grade_image_urls(taxon_keys)

                # todo: refactor to separate function
                if image_urls:
                    logging.info(f"Found research-grade images for {taxon_keys}: {image_urls}")
                    for i, url in enumerate(image_urls):
                        file_name = f"{species_key}_{i + 1}.jpg"
                        download_image(url, image_output_dir, file_name)
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
    csv_file_name = 'brazil_trees_amazonia_sel_occs.csv'
    csv_file_path = dir_path / csv_file_name
    image_output_dir = dir_path / 'images/'

    process_species_images(csv_file_path, image_output_dir)
