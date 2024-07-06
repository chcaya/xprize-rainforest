import re
from pathlib import Path
from typing import List, Tuple

import albumentations
import numpy as np
import pandas as pd
import rasterio
from geodataset.dataset.base_dataset import BaseLabeledCocoDataset
from geodataset.utils import rle_segmentation_to_mask, mask_to_polygon

from engine.embedder.contrastive.contrastive_utils import FOREST_QPEB_MEAN, FOREST_QPEB_STD, scale_values, normalize


class BaseContrastiveLabeledCocoDataset(BaseLabeledCocoDataset):
    def __init__(self,
                 fold: str,
                 root_path: Path or List[Path],
                 date_pattern: str or None,
                 day_month_year: Tuple[int, int, int] = None,
                 transform: albumentations.core.composition.Compose = None):
        super().__init__(fold=fold, root_path=root_path, transform=transform)

        self.date_pattern = date_pattern
        self.day_month_year = day_month_year

        assert not (self.date_pattern is None and self.day_month_year is None), \
            "Either date_pattern or day_month_year should be provided."

        self._get_dates()

    def _get_dates(self):
        for idx in self.tiles:
            if self.day_month_year:
                self.tiles[idx]['year'] = self.day_month_year[2]
                self.tiles[idx]['month'] = self.day_month_year[1]
                self.tiles[idx]['day'] = self.day_month_year[0]
            else:
                date_match = re.match(self.date_pattern, self.tiles[idx]['name'])
                if date_match.group('year'):
                    self.tiles[idx]['year'] = int(date_match.group('year'))
                    self.tiles[idx]['month'] = int(date_match.group('month'))
                    self.tiles[idx]['day'] = int(date_match.group('day'))
                elif date_match.group('year2'):
                    self.tiles[idx]['year'] = int(date_match.group('year2'))
                    self.tiles[idx]['month'] = int(date_match.group('month2'))
                    self.tiles[idx]['day'] = int(date_match.group('day2'))
                else:
                    raise ValueError(f"Could not find date in tile name: {self.tiles[idx]['name']}.")

    def __getitem__(self, idx: int):
        pass

    def __len__(self):
        pass


class ContrastiveInternalDataset(BaseContrastiveLabeledCocoDataset):
    def __init__(self,
                 fold: str,
                 root_path: Path or List[Path],
                 date_pattern: str or None,
                 day_month_year: Tuple[int, int, int] = None):
        super().__init__(fold=fold, root_path=root_path, date_pattern=date_pattern, day_month_year=day_month_year, transform=None)

        self.tiles_per_category_id = self._generate_tiles_per_class()

    def _generate_tiles_per_class(self):
        tiles_per_class = {}
        for idx in self.tiles:
            assert len(self.tiles[idx]['labels']) == 1, \
                "SiameseSamplerInternalDataset dataset should have only one label (polygon) per tile."
            category_id = self.tiles[idx]['labels'][0]['category_id']
            if category_id not in tiles_per_class:
                tiles_per_class[category_id] = []
            tiles_per_class[category_id].append(idx)
        return tiles_per_class

    def __getitem__(self, idx: int):
        return self.tiles[idx]

    def __len__(self):
        return len(self.tiles)


class ContrastiveDataset:
    POSITIVE_LABEL = 1
    NEGATIVE_LABEL = 0
    DEAD_DISTANCE = 5

    def __init__(self,
                 dataset_config: dict,
                 min_level: str,
                 image_size: int,
                 random_crop: bool,
                 transform: albumentations.core.composition.Compose,
                 taxa_distances_df: pd.DataFrame or None,
                 max_resampling_times: int,
                 normalize: bool = True,
                 mean: np.array = FOREST_QPEB_MEAN,
                 std: np.array = FOREST_QPEB_STD,
                 min_margin: int = 0.5,
                 max_margin: int = 2):
        self.dataset_config = dataset_config
        self.min_level = min_level
        self.image_size = image_size
        self.random_crop = random_crop
        self.transform = transform
        self.taxa_distances_df = taxa_distances_df
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.min_margin = min_margin
        self.max_margin = max_margin
        self.max_resampling_times = max_resampling_times

        self.datasets = dataset_config

        assert self.min_level in ['species', 'genus', 'family'], f"min_level should be one of ['species', 'genus', 'family'], got {self.min_level}."

        if self.taxa_distances_df is not None:
            self.taxa_distances_df = taxa_distances_df.set_index(['canonicalName1', 'canonicalName2'])
            # scaling negative pairs distances to [min_margin, max_margin]
            self.taxa_distances_df.loc[self.taxa_distances_df['dist'] > 0, 'dist'] = scale_values(
                values=self.taxa_distances_df.loc[self.taxa_distances_df['dist'] > 0, 'dist'],
                old_min=min(self.taxa_distances_df.loc[self.taxa_distances_df['dist'] > 0, 'dist']),
                old_max=max(self.taxa_distances_df.loc[self.taxa_distances_df['dist'] > 0, 'dist']),
                new_min=self.min_margin,
                new_max=self.max_margin
            )

        self.categories_names, self.categories_names_to_idx, self.categories_dists = self._get_categories_distances()
        self.all_samples_labels, self.all_samples_families, self.all_samples_dataset_indices, self.samples_indices_per_label = self._get_all_samples()
        self.families_set = set(self.all_samples_families)
        self._remove_not_represented_categories()

        sorted_dict = {k: len(v) for k, v in sorted(self.samples_indices_per_label.items(), key=lambda item: len(item[1]), reverse=True)}
        print(f'Categories representation BEFORE category balancing: {sorted_dict}')

        self._equilibrate_classes()
        sorted_dict = {k: len(v) for k, v in sorted(self.samples_indices_per_label.items(), key=lambda item: len(item[1]), reverse=True)}
        print(f'Categories representation AFTER category balancing: {sorted_dict}')

        self.index = [i for i in range(len(self))]

    def shuffle(self, seed: int = 0):
        np.random.seed(seed)
        np.random.shuffle(self.index)

    def _get_categories_distances(self):
        categories_names_to_rank = {}
        categories_species = []
        categories_genus = []
        categories_family = []

        for dataset_key in self.datasets:
            category_id_to_metadata_mapping = self.datasets[dataset_key].category_id_to_metadata_mapping
            for category_id in category_id_to_metadata_mapping:
                category = category_id_to_metadata_mapping[category_id]
                category_name = category['name']

                if category_name == 'Dead':
                    categories_names_to_rank[category_name] = None
                else:
                    categories_names_to_rank[category_name] = category['rank']
                    if category['rank'] == 'SPECIES':
                        categories_species.append(category_name)
                        categories_genus.append(category_id_to_metadata_mapping[category['supercategory']]['name'])
                        categories_family.append(category_id_to_metadata_mapping[category_id_to_metadata_mapping[category['supercategory']]['supercategory']]['name'])
                    elif category['rank'] == 'GENUS':
                        categories_species.append(None)
                        categories_genus.append(category_name)
                        categories_family.append(category_id_to_metadata_mapping[category['supercategory']]['name'])
                    elif category['rank'] == 'FAMILY':
                        categories_species.append(None)
                        categories_genus.append(None)
                        categories_family.append(category_name)
                    else:
                        raise ValueError(f"Unknown category rank: {category['rank']}.")

        print('Generating categories distances...')
        categories_dists = {}
        for category_name_1 in categories_names_to_rank.keys():
            for category_name_2 in categories_names_to_rank.keys():
                if category_name_1 == category_name_2:
                    categories_dists[(category_name_1, category_name_2)] = 0
                elif category_name_1 == 'Dead' or category_name_2 == 'Dead':
                    categories_dists[(category_name_1, category_name_2)] = self.DEAD_DISTANCE
                else:
                    if self.taxa_distances_df is not None:
                        categories_dists[(category_name_1, category_name_2)] = self.taxa_distances_df.loc[(category_name_1, category_name_2), 'dist']
                    else:
                        categories_dists[(category_name_1, category_name_2)] = 1

        print('Categories distances successfully generated.')

        categories_names = set(categories_names_to_rank.keys())
        categories_names_to_idx = {k: i + 1 for i, k in enumerate(categories_names)}

        return categories_names, categories_names_to_idx, categories_dists

    def _get_all_samples(self):
        all_samples_dataset_indices = {}
        all_samples_labels = []
        all_samples_families = []
        samples_indices_per_label = {k: [] for k in self.categories_names}
        global_sample_id = 0
        for dataset_key in self.datasets:
            dataset = self.datasets[dataset_key]
            for dataset_sample_id, sample in enumerate(dataset):
                category_id = sample['labels'][0]['category_id']
                if category_id:
                    category_name = dataset.category_id_to_metadata_mapping[category_id]['name']
                    if category_name == 'Dead':
                        all_samples_dataset_indices[global_sample_id] = [dataset_key, dataset_sample_id]
                        all_samples_labels.append(category_name)
                        all_samples_families.append(category_name)
                        samples_indices_per_label[category_name].append(global_sample_id)
                        global_sample_id += 1
                        continue

                    rank = dataset.category_id_to_metadata_mapping[category_id]['rank'].lower()

                    if rank == 'species':
                        genus_name = dataset.category_id_to_metadata_mapping[dataset.category_id_to_metadata_mapping[category_id]['supercategory']]['name']
                        family_name = dataset.category_id_to_metadata_mapping[dataset.category_id_to_metadata_mapping[dataset.category_id_to_metadata_mapping[category_id]['supercategory']]['supercategory']]['name']
                    elif rank == 'genus':
                        genus_name = category_name
                        family_name = dataset.category_id_to_metadata_mapping[dataset.category_id_to_metadata_mapping[category_id]['supercategory']]['name']
                    elif rank == 'family':
                        genus_name = None
                        family_name = category_name
                    else:
                        raise ValueError(f"Unknown category rank: {rank}.")

                    if self.min_level == 'species':
                        if category_name in self.categories_names:
                            all_samples_dataset_indices[global_sample_id] = [dataset_key, dataset_sample_id]
                            all_samples_labels.append(category_name)
                            all_samples_families.append(family_name)
                            samples_indices_per_label[category_name].append(global_sample_id)
                            global_sample_id += 1

                    elif self.min_level == 'genus':
                        if genus_name in self.categories_names:
                            all_samples_dataset_indices[global_sample_id] = [dataset_key, dataset_sample_id]
                            all_samples_labels.append(genus_name)
                            all_samples_families.append(family_name)
                            samples_indices_per_label[genus_name].append(global_sample_id)
                            global_sample_id += 1

                    elif self.min_level == 'family':
                        if family_name in self.categories_names:
                            all_samples_dataset_indices[global_sample_id] = [dataset_key, dataset_sample_id]
                            all_samples_labels.append(family_name)
                            all_samples_families.append(family_name)
                            samples_indices_per_label[family_name].append(global_sample_id)
                            global_sample_id += 1

        return all_samples_labels, all_samples_families, all_samples_dataset_indices, samples_indices_per_label

    def _remove_not_represented_categories(self):
        self.categories_names = set([k for k in self.categories_names if len(self.samples_indices_per_label[k]) != 0])
        self.samples_indices_per_label = {k: v for k, v in self.samples_indices_per_label.items() if len(v) != 0}

    def _equilibrate_classes(self):
        if self.max_resampling_times > 0:
            # get the mean and median values of samples per class
            mean_samples_per_class = int(np.mean([len(v) for v in self.samples_indices_per_label.values()]))
            print(f"Mean samples per class: {mean_samples_per_class}")

            next_index = len(self.all_samples_dataset_indices)
            for category_name in self.categories_names:
                samples_for_category = self.samples_indices_per_label[category_name]
                if len(samples_for_category) < mean_samples_per_class:
                    shuffled_indices = np.random.choice(samples_for_category, size=int(mean_samples_per_class - len(samples_for_category)), replace=True)
                    shuffled_indices = shuffled_indices[:min(mean_samples_per_class, self.max_resampling_times * len(samples_for_category))]
                    for idx_to_be_duplicated in shuffled_indices:
                        self.all_samples_dataset_indices[next_index] = self.all_samples_dataset_indices[idx_to_be_duplicated]
                        self.all_samples_labels.append(self.all_samples_labels[idx_to_be_duplicated])
                        self.all_samples_families.append(self.all_samples_families[idx_to_be_duplicated])
                        self.samples_indices_per_label[category_name].append(next_index)
                        next_index += 1

    def __len__(self):
        return len(self.all_samples_dataset_indices)

    def __getitem__(self, idx):
        real_idx = self.index[idx]
        dataset_key, dataset_idx = self.all_samples_dataset_indices[real_idx]
        label = self.all_samples_labels[real_idx]
        label_id = self.categories_names_to_idx[label]
        family = self.all_samples_families[real_idx]
        family_id = self.categories_names_to_idx[family]
        tile = self.datasets[dataset_key][dataset_idx]
        month, day = tile['month'], tile['day']

        with rasterio.open(tile['path']) as tile_file:
            data = tile_file.read([1, 2, 3])

        if data.shape[1] > self.image_size:
            if self.random_crop:
                x_max = data.shape[1] - self.image_size
                y_max = data.shape[2] - self.image_size

                # Function to perform random crop and check for black pixels
                def random_crop_with_black_check():
                    max_tries = 20
                    for _ in range(max_tries):
                        x = np.random.randint(0, x_max)
                        y = np.random.randint(0, y_max)
                        cropped_data = data[:, x:x + self.image_size, y:y + self.image_size]
                        black_pixels = np.sum(cropped_data == 0) / (self.image_size * self.image_size * 3)
                        if black_pixels <= 0.4:
                            return cropped_data
                    # If no valid crop is found, return the last attempted crop
                    return cropped_data

                data = random_crop_with_black_check()
            else:
                # crop
                data_center = int(data.shape[1] / 2)
                data = data[:,
                            data_center - self.image_size // 2:data_center + self.image_size // 2,
                            data_center - self.image_size // 2:data_center + self.image_size // 2,
                            ]
        elif data.shape[1] < self.image_size:
            # pad
            padding = (self.image_size - data.shape[1]) // 2
            padded_data = np.zeros((data.shape[0], self.image_size, self.image_size), dtype=data.dtype)
            padded_data[:, padding:padding + data.shape[1], padding:padding + data.shape[1]] = data
            data = padded_data

        data = data / 255

        if self.normalize:
            data = normalize(data, self.mean, self.std)

        return data, month, day, label_id, label, family_id, family


class ContrastiveInferDataset(BaseContrastiveLabeledCocoDataset):
    def __init__(self, image_size: int, transform: albumentations.core.composition.Compose,
                 fold: str, root_path: Path or List[Path], date_pattern: str or None,
                 day_month_year: Tuple[int, int, int] = None, normalize: bool = True,
                 mean: np.array = FOREST_QPEB_MEAN, std: np.array = FOREST_QPEB_STD):

        super().__init__(fold=fold, root_path=root_path, date_pattern=date_pattern,
                         day_month_year=day_month_year, transform=transform)
        self.image_size = image_size
        self.transform = transform
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        month, day = tile['month'], tile['day']

        with rasterio.open(tile['path']) as tile_file:
            data = tile_file.read([1, 2, 3])

        if data.shape[1] > self.image_size:
            # crop
            data_center = int(data.shape[1] / 2)
            data = data[:,
                        data_center - self.image_size // 2:data_center + self.image_size // 2,
                        data_center - self.image_size // 2:data_center + self.image_size // 2,
                        ]
        elif data.shape[1] < self.image_size:
            # pad
            padding = (self.image_size - data.shape[1]) // 2
            padded_data = np.zeros((data.shape[0], self.image_size, self.image_size), dtype=data.dtype)
            padded_data[:, padding:padding + data.shape[1], padding:padding + data.shape[1]] = data
            data = padded_data

        data = data / 255

        if self.normalize:
            data = normalize(data, self.mean, self.std)

        return data, month, day


