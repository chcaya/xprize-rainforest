from typing import List
import requests
from typing import Dict
import logging
from PIL import Image
from io import BytesIO
from pathlib import Path

# API Endpoints
API_GBIF_OCCURRENCE = 'https://api.gbif.org/v1/occurrence/search'
API_GBIF_SPECIES_DETAIL = 'https://api.gbif.org/v1/species/'


def fetch_data(url: str, params: Dict = None) -> Dict:
    """Utility function to fetch data from a URL."""
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Request failed: {e}")
        return {}


def fetch_species_details(species_id: str) -> Dict:
    """Fetch detailed information for a species using its GBIF ID."""
    return fetch_data(f"{API_GBIF_SPECIES_DETAIL}{species_id}")

# todo: se
def fetch_occurrences_by_taxon(taxon_keys: Dict[str, str], limit: int = 20) -> List[Dict]:
    """Fetch occurrences for given taxon keys."""
    params = {
        'mediaType': 'StillImage',
        'quality': 'RESEARCH',
        'limit': limit
    }
    # Merge the specific taxon keys into the params dictionary
    params.update(taxon_keys)
    data = fetch_data(API_GBIF_OCCURRENCE, params)
    return data.get('results', [])


def fetch_research_grade_image_urls(taxon_keys: Dict[str, str], limit = 20) -> List[str]:
    """Fetch research-grade image URLs from GBIF occurrences."""
    results = fetch_occurrences_by_taxon(taxon_keys, limit=limit)
    image_urls = []
    for result in results:
        if 'media' in result:
            for media in result['media']:
                if 'identifier' in media:
                    image_urls.append(media['identifier'])
    return image_urls[:limit]


def download_image(image_url: str, output_dir: Path, folder_name: str, file_name: str) -> None:
    """Download an image from a given URL and save it to a directory organized by folder name (species or family)."""
    specific_output_dir = output_dir / folder_name
    specific_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        output_path = specific_output_dir / file_name
        image.save(output_path)
        logging.info(f"Image saved to {output_path}")
    except requests.RequestException as e:
        logging.error(f"Failed to download image: {e}")