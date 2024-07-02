import glob
from pathlib import Path
from typing import List

import pandas as pd

class BioClipFileLoader:
    def __init__(self, dir_path: Path, taxonomy_file: str):
        self.dir_path = dir_path
        self.taxonomy_file = taxonomy_file
        self.taxonomy_data = self._load_taxonomy_data()

    def _load_taxonomy_data(self) -> pd.DataFrame:
        taxonomy_path = self.dir_path / self.taxonomy_file
        return pd.read_csv(taxonomy_path)

    def get_folders(self, folder_glob_pattern: str) -> List[str]:
        folder_glob_search_str = self.dir_path / folder_glob_pattern
        return glob.glob(str(folder_glob_search_str))

    def get_image_paths(self, folder: str) -> List[str]:
        return glob.glob(str(Path(folder) / '*'))

    def get_taxonomy_data(self) -> pd.DataFrame:
        return self.taxonomy_data
