from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class BioClipDataset(Dataset):
    def __init__(self, image_paths: List[str], taxonomy_data: pd.DataFrame, preprocess):
        self.image_paths = image_paths
        self.taxonomy_data = taxonomy_data
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def get_taxon_key_from_df(self, file_name: str, key: str = 'genusKey') -> int:
        if '_crop' in file_name:
            file_name = file_name.split('_crop')[0] + '.JPG'
        if 'tile' in file_name:
            file_name = file_name.split('_tile')[0] + '.JPG'
        search_row = self.taxonomy_data[self.taxonomy_data['fileName'] == file_name]
        search_row = search_row.fillna(-1)
        return int(search_row.iloc[0][key])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image).unsqueeze(0)

        family_key = self.get_taxon_key_from_df(Path(image_path).name, key='familyKey')
        genus_key = self.get_taxon_key_from_df(Path(image_path).name, key='genusKey')
        label = f'{family_key}_{genus_key}'
        return image, label


