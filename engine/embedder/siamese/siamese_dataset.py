import itertools
import random
import time
from itertools import combinations
from pathlib import Path
from typing import List

import re

import albumentations
import cv2
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from _datasketches import kll_floats_sketch
from geodataset.dataset.base_dataset import BaseLabeledCocoDataset
from geodataset.utils import rle_segmentation_to_mask, decode_rle_to_polygon
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from tqdm import tqdm

from engine.embedder.siamese.siamese_utils import normalize_non_black_pixels, FOREST_QPEB_MEAN, FOREST_QPEB_STD, \
    normalize, scale_values, LimitedSizeHeap


class BaseSiameseLabeledCocoDataset(BaseLabeledCocoDataset):
    def __init__(self,
                 fold: str,
                 root_path: Path or List[Path],
                 date_pattern: str,
                 transform: albumentations.core.composition.Compose = None):
        super().__init__(fold=fold, root_path=root_path, transform=transform)

        self.date_pattern = date_pattern
        self._get_dates()

    def _get_dates(self):
        for idx in self.tiles:
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


class SiameseSamplerInternalDataset(BaseSiameseLabeledCocoDataset):
    def __init__(self,
                 fold: str,
                 root_path: Path or List[Path],
                 date_pattern: str):
        super().__init__(fold=fold, root_path=root_path, date_pattern=date_pattern, transform=None)

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


class SiameseSamplerDataset:
    POSITIVE_LABEL = 1
    NEGATIVE_LABEL = 0
    DEAD_DISTANCE = 5

    def __init__(self,
                 dataset_config: dict,
                 image_size: int,
                 transform: albumentations.core.composition.Compose,
                 taxa_distances_df: pd.DataFrame or None,
                 n_positive_pairs: int,
                 n_negative_pairs: int,
                 consider_percentile: int,
                 min_positive_pairs_per_category: int = 10,
                 normalize: bool = True,
                 mean: np.array = FOREST_QPEB_MEAN,
                 std: np.array = FOREST_QPEB_STD,
                 min_margin: int = 0.5,
                 max_margin: int = 2):
        self.dataset_config = dataset_config
        self.image_size = image_size
        self.transform = transform
        self.taxa_distances_df = taxa_distances_df
        self.n_positive_pairs = n_positive_pairs
        self.n_negative_pairs = n_negative_pairs
        self.consider_percentile = consider_percentile
        self.min_positive_pairs_per_category = min_positive_pairs_per_category
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.min_margin = min_margin
        self.max_margin = max_margin

        self.datasets = dataset_config['datasets']

        assert 0 <= consider_percentile <= 100, "consider_percentile should be between 0 and 100."

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

        self.categories_names, self.categories_dists = self._get_categories_distances()
        self.all_samples_labels, self.all_samples_indices, self.samples_indices_per_label = self._get_all_samples()
        self._remove_not_represented_categories()

        sorted_dict = {k: len(v) for k, v in sorted(self.samples_indices_per_label.items(), key=lambda item: len(item[1]), reverse=True)}
        # Print the sorted dictionary
        for key, value in sorted_dict.items():
            print(f'{key} => {value}')
        print(f"Total number of samples: {len(self.all_samples_indices)}")
        self.pairs_indices = self.find_semi_random_siamese_pairs()

        print(f"Generated {len(self.pairs_indices)} semi-random pairs.")

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

        return categories_names, categories_dists

    def _get_all_samples(self):
        all_samples_indices = {}
        all_samples_labels = []
        samples_indices_per_label = {k: [] for k in self.categories_names}
        global_sample_id = 0
        for dataset_key in self.datasets:
            dataset = self.datasets[dataset_key]
            for dataset_sample_id, sample in enumerate(dataset):
                category_id = sample['labels'][0]['category_id']
                if category_id:
                    category_name = dataset.category_id_to_metadata_mapping[category_id]['name']
                    if category_name in self.categories_names:
                        all_samples_indices[global_sample_id] = [dataset_key, dataset_sample_id]
                        all_samples_labels.append(category_name)
                        samples_indices_per_label[category_name].append(global_sample_id)
                        global_sample_id += 1

        return all_samples_labels, all_samples_indices, samples_indices_per_label

    def _remove_not_represented_categories(self):
        self.categories_names = set([k for k in self.categories_names if len(self.samples_indices_per_label[k]) != 0])
        self.samples_indices_per_label = {k: v for k, v in self.samples_indices_per_label.items() if len(v) != 0}

    def find_semi_random_siamese_pairs(self):
        positive_pairs_indices = {}
        negative_pairs_indices = {}
        sorted_n_samples_per_label = {k: len(v) for k, v in
                                      sorted(self.samples_indices_per_label.items(), key=lambda item: len(item[1]),
                                             reverse=False)}

        n_positive_pairs_added = 0
        n_negative_pairs_added = 0
        n_category_pairs_processed = 0
        for i, label_name_1 in enumerate(sorted_n_samples_per_label.keys()):
            for j, label_name_2 in enumerate(sorted_n_samples_per_label.keys()):
                if j > i:
                    continue
                elif j == i:
                    # Positive pairs
                    n_positive_sample_pairs_per_category_pair = int(
                        (self.n_positive_pairs - n_positive_pairs_added) / (len(sorted_n_samples_per_label.keys()) - i))
                    label_samples = self.samples_indices_per_label[label_name_1]
                    n_possible_pairs = (len(label_samples) ** 2 - len(label_samples)) / 2
                    if len(label_samples) == 1:
                        item = label_samples[0]
                        pairs = [(item, item)] * self.min_positive_pairs_per_category
                    elif n_possible_pairs < n_positive_sample_pairs_per_category_pair:
                        pairs = list(itertools.combinations(label_samples, 2))
                        if len(pairs) < self.min_positive_pairs_per_category:
                            pairs += [(item, item) for item in self.samples_indices_per_label[label_name_1]]
                        if len(pairs) < self.min_positive_pairs_per_category:
                            pairs = (pairs * ((self.min_positive_pairs_per_category + len(pairs) - 1) // len(pairs)))[:self.min_positive_pairs_per_category]
                    else:
                        random_label_samples_set1 = random.sample(
                            self.samples_indices_per_label[label_name_1],
                            k=min(len(self.samples_indices_per_label[label_name_1]),
                                  n_positive_sample_pairs_per_category_pair)
                        )
                        random_label_samples_set2 = random.sample(
                            self.samples_indices_per_label[label_name_1],
                            k=min(len(self.samples_indices_per_label[label_name_1]),
                                  n_positive_sample_pairs_per_category_pair)
                        )

                        # we start by adding as many pairs of distinct samples as possible ([x1,y1], [x2, y2],...) where xi != xj and yi != yj
                        pairs = [(sample1, sample2) for sample1, sample2 in zip(random_label_samples_set1, random_label_samples_set2)]

                        # we then complete with some xi = xj or yi = yj if necessary
                        if len(pairs) < n_positive_sample_pairs_per_category_pair:
                            possible_pairs_iterator = itertools.product(random_label_samples_set1, random_label_samples_set2)
                            pairs += [next(possible_pairs_iterator) for _ in range(n_positive_sample_pairs_per_category_pair - len(pairs))]

                    positive_pairs_indices[(label_name_1, label_name_1)] = list(pairs)
                    n_positive_pairs_added += len(pairs)
                else:
                    # Negative pairs
                    n_negative_sample_pairs_per_category_pair = int((self.n_negative_pairs - n_negative_pairs_added) / (
                            (len(sorted_n_samples_per_label.keys()) ** 2 - len(sorted_n_samples_per_label.keys())) / 2 - n_category_pairs_processed))
                    n_possible_pairs = len(self.samples_indices_per_label[label_name_1]) * len(self.samples_indices_per_label[label_name_2])
                    if n_possible_pairs < n_negative_sample_pairs_per_category_pair:
                        pairs = list(itertools.product(self.samples_indices_per_label[label_name_1],
                                                       self.samples_indices_per_label[label_name_2]))
                    else:
                        # we get a random subset of samples of size n_negative_sample_pairs_per_category_pair for each category/label pair
                        random_label_1_samples = random.sample(
                            self.samples_indices_per_label[label_name_1],
                            k=min(len(self.samples_indices_per_label[label_name_1]),
                                  n_negative_sample_pairs_per_category_pair)
                        )
                        random_label_2_samples = random.sample(
                            self.samples_indices_per_label[label_name_2],
                            k=min(len(self.samples_indices_per_label[label_name_2]),
                                  n_negative_sample_pairs_per_category_pair)
                        )

                        # we start by adding as many pairs of distinct samples as possible ([x1,y1], [x2, y2],...) where xi != xj and yi != yj
                        pairs = set([(random_label_1_samples[i], random_label_2_samples[i]) for i in range(min(len(random_label_1_samples), len(random_label_2_samples)))])

                        # we then complete with some xi = xj or yi = yj if necessary
                        if len(pairs) < n_negative_sample_pairs_per_category_pair:
                            possible_pairs = set(itertools.product(random_label_1_samples, random_label_2_samples))
                            possible_pairs = possible_pairs - pairs
                            pairs.update(random.sample(possible_pairs, k=n_negative_sample_pairs_per_category_pair - len(pairs)))

                        assert len(pairs) == n_negative_sample_pairs_per_category_pair

                    negative_pairs_indices[(label_name_1, label_name_2)] = list(pairs)
                    n_negative_pairs_added += len(pairs)
                    n_category_pairs_processed += 1

        # for label_1 in self.categories_names:
        #     print(label_1, 'POSITIVE', sum([len(v) for x, v in positive_pairs_indices.items() if x[0] == label_1 or x[1] == label_1]))
        #     print(label_1, 'NEGATIVE', sum([len(v) for x, v in negative_pairs_indices.items() if x[0] == label_1 or x[1] == label_1]))
        #
        # print(sum([len(x) for x in positive_pairs_indices.values()]))
        # print(sum([len(x) for x in negative_pairs_indices.values()]))

        all_pairs = (list(itertools.chain.from_iterable(positive_pairs_indices.values()))
                     + list(itertools.chain.from_iterable(negative_pairs_indices.values())))

        all_pairs_shuffled = random.sample(all_pairs, len(all_pairs))

        self.pairs_indices = all_pairs_shuffled

        return all_pairs_shuffled

    def find_optimal_siamese_pairs(self,
                                   embeddings: torch.Tensor,
                                   compute_device: torch.device,
                                   distance_compute_batch_size: int):
        positive_pairs_indices = {}
        negative_pairs_indices = {}
        sorted_n_samples_per_label = {k: len(v) for k, v in
                                      sorted(self.samples_indices_per_label.items(), key=lambda item: len(item[1]),
                                             reverse=False)}

        n_positive_pairs_added = 0
        n_negative_pairs_added = 0
        n_category_pairs_processed = 0
        for i, label_name_1 in tqdm(enumerate(sorted_n_samples_per_label.keys()),
                                    total=len(sorted_n_samples_per_label.keys()),
                                    desc='Optimal pairs generation'):
            for j, label_name_2 in enumerate(sorted_n_samples_per_label.keys()):
                if j > i:
                    continue
                elif j == i:
                    # Positive pairs
                    n_positive_sample_pairs_per_category_pair = int((self.n_positive_pairs - n_positive_pairs_added)
                                                                    / (len(sorted_n_samples_per_label.keys()) - i))
                    label_samples_ids = self.samples_indices_per_label[label_name_1]
                    n_possible_pairs = (len(label_samples_ids) ** 2 - len(label_samples_ids)) / 2

                    if len(label_samples_ids) == 1:
                        item = label_samples_ids[0]
                        pairs = [(item, item)] * self.min_positive_pairs_per_category
                    elif n_possible_pairs < n_positive_sample_pairs_per_category_pair:
                        pairs = list(itertools.combinations(label_samples_ids, 2))
                        if len(pairs) < self.min_positive_pairs_per_category:
                            pairs += [(item, item) for item in self.samples_indices_per_label[label_name_1]]
                        if len(pairs) < self.min_positive_pairs_per_category:
                            pairs = (pairs * ((self.min_positive_pairs_per_category + len(pairs) - 1) // len(pairs)))[:self.min_positive_pairs_per_category]
                    else:
                        embeddings_label = embeddings[label_samples_ids]
                        best_pairs_indices = self._get_best_samples(
                            embeddings1=embeddings_label,
                            embeddings2=embeddings_label,
                            distance_compute_batch_size=distance_compute_batch_size,
                            compute_device=compute_device,
                            keep_n=n_positive_sample_pairs_per_category_pair,
                            metric='farthest'
                        )
                        pairs = [(label_samples_ids[i1], label_samples_ids[i2]) for i1, i2 in best_pairs_indices]

                    positive_pairs_indices[(label_name_1, label_name_1)] = list(pairs)
                    n_positive_pairs_added += len(pairs)
                else:
                    n_negative_sample_pairs_per_category_pair = int((self.n_negative_pairs - n_negative_pairs_added) / (
                            (len(sorted_n_samples_per_label.keys()) ** 2 - len(
                                sorted_n_samples_per_label.keys())) / 2 - n_category_pairs_processed))
                    label1_samples_ids = self.samples_indices_per_label[label_name_1]
                    label2_samples_ids = self.samples_indices_per_label[label_name_2]
                    n_possible_pairs = len(label1_samples_ids) * len(label2_samples_ids)
                    if n_possible_pairs < n_negative_sample_pairs_per_category_pair:
                        pairs = list(itertools.product(label1_samples_ids, label2_samples_ids))
                    else:
                        embeddings_label1 = embeddings[label1_samples_ids]
                        embeddings_label2 = embeddings[label2_samples_ids]
                        best_pairs_indices = self._get_best_samples(
                            embeddings1=embeddings_label1,
                            embeddings2=embeddings_label2,
                            distance_compute_batch_size=distance_compute_batch_size,
                            compute_device=compute_device,
                            keep_n=n_negative_sample_pairs_per_category_pair,
                            metric='closest'
                        )
                        pairs = [(label1_samples_ids[i1], label2_samples_ids[i2]) for (i1, i2) in best_pairs_indices]

                    negative_pairs_indices[(label_name_1, label_name_2)] = list(pairs)
                    n_negative_pairs_added += len(pairs)
                    n_category_pairs_processed += 1

        all_pairs = (list(itertools.chain.from_iterable(positive_pairs_indices.values()))
                     + list(itertools.chain.from_iterable(negative_pairs_indices.values())))

        all_pairs_shuffled = random.sample(all_pairs, len(all_pairs))

        for label_1 in self.categories_names:
            print(label_1, 'POSITIVE', sum([len(v) for x, v in positive_pairs_indices.items() if x[0] == label_1 or x[1] == label_1]))
            print(label_1, 'NEGATIVE', sum([len(v) for x, v in negative_pairs_indices.items() if x[0] == label_1 or x[1] == label_1]))

        print("Total positive pairs generated:", sum([len(x) for x in positive_pairs_indices.values()]))
        print("Total negative pairs generated:", sum([len(x) for x in negative_pairs_indices.values()]))

        self.pairs_indices = all_pairs_shuffled

        return self.pairs_indices

    def _get_best_samples(self,
                          embeddings1: torch.Tensor,
                          embeddings2: torch.Tensor,
                          distance_compute_batch_size,
                          compute_device: torch.device,
                          keep_n: int,
                          metric: str):

        assert metric in ['closest', 'farthest']

        embeddings1 = embeddings1.to(compute_device)
        embeddings2 = embeddings2.to(compute_device)

        n = embeddings1.shape[0]
        if distance_compute_batch_size < n:
            raise ValueError("distance_compute_batch_size should be greater than the number of samples.")

        batch_size = distance_compute_batch_size // n

        # t_digest = TDigest()
        kll = kll_floats_sketch()
        # First loop to calculate the 80th percentile threshold
        for i in range(0, n, batch_size):
            end_index = min(i + batch_size, n)
            batch_embeddings1 = embeddings1[i:end_index]
            distances = torch.cdist(batch_embeddings1, embeddings2, p=2)
            distances_flat = distances.view(-1)

            # Update the kll with the new batch of distances. If too many items, randomly sample 1M items
            kll_max_update_size = 1000000
            if len(distances_flat) > kll_max_update_size:
                distances_flat = distances_flat[torch.randperm(distances_flat.numel())[:kll_max_update_size]]
            for distance in distances_flat.cpu().numpy():
                kll.update(distance)

        total_distances = embeddings1.shape[0] * embeddings2.shape[0]
        # Get the estimated 80th percentile value
        if metric == 'closest':
            if ((self.consider_percentile / 100) * total_distances) < keep_n:
                # threshold_value = t_digest.percentile(100 * int(keep_n / total_distances))
                threshold_value = kll.get_quantile(int(keep_n / total_distances))
            else:
                # threshold_value = t_digest.percentile(self.consider_percentile)
                threshold_value = kll.get_quantile(self.consider_percentile / 100)
        else:
            if ((self.consider_percentile / 100) * total_distances) < keep_n:
                # threshold_value = t_digest.percentile(100 - 100 * int(keep_n / total_distances))
                threshold_value = kll.get_quantile(1 - int(keep_n / total_distances))
            else:
                # threshold_value = t_digest.percentile(100 - self.consider_percentile)
                threshold_value = kll.get_quantile(1 - self.consider_percentile / 100)

        indices = []
        values_sum = 0
        all_distances_sum = 0
        for i in range(0, n, batch_size):
            end_index = min(i + batch_size, n)
            batch_embeddings1 = embeddings1[i:end_index]
            # Compute distances between the current batch and all embeddings
            distances = torch.cdist(batch_embeddings1, embeddings2, p=2)
            this_batch_size_ratio = distances.numel() / total_distances
            random.seed(time.time())
            if metric == 'closest':
                distances_under_threshold = distances < threshold_value
                indices_under_threshold = torch.nonzero(distances_under_threshold, as_tuple=False)
                # randomly select this_batch_size_ratio * keep_n closest distances
                indices_under_threshold_random_selection = random.sample(indices_under_threshold.tolist(), int(this_batch_size_ratio * keep_n))
                indices.extend(indices_under_threshold_random_selection)
                for idx in indices_under_threshold_random_selection:
                    values_sum += distances[idx[0], idx[1]]
            else:
                distances_over_threshold = distances > threshold_value
                indices_over_threshold = torch.nonzero(distances_over_threshold, as_tuple=False)
                # randomly select this_batch_size_ratio * keep_n farthest distances
                indices_under_threshold_random_selection = random.sample(indices_over_threshold.tolist(), int(this_batch_size_ratio * keep_n))
                indices.extend(indices_under_threshold_random_selection)
                for idx in indices_under_threshold_random_selection:
                    values_sum += distances[idx[0], idx[1]]
            all_distances_sum += torch.sum(distances)

        indices = (indices * (keep_n // len(indices) + 1))[:keep_n] if len(indices) < keep_n else indices[:keep_n]
        if metric == 'closest':
            print('NEGATIVE:', values_sum / len(indices), 'DISTANCE MEAN:', all_distances_sum / total_distances, len(indices), keep_n)
        else:
            print('POSITIVE:', values_sum / len(indices), 'DISTANCE MEAN:', all_distances_sum / total_distances, len(indices), keep_n)
        return indices

    def __len__(self):
        return len(self.pairs_indices)

    def __getitem__(self, idx):
        global_idx_1, global_idx_2 = self.pairs_indices[idx]
        dataset_1_key, dataset_idx_1 = self.all_samples_indices[global_idx_1]
        dataset_2_key, dataset_idx_2 = self.all_samples_indices[global_idx_2]
        label_1 = self.all_samples_labels[global_idx_1]
        label_2 = self.all_samples_labels[global_idx_2]
        label = int(label_1 == label_2)
        margin = self.categories_dists[(label_1, label_2)]
        tile_1 = self.datasets[dataset_1_key][dataset_idx_1]
        tile_2 = self.datasets[dataset_2_key][dataset_idx_2]
        month1, month2 = tile_1['month'], tile_2['month']
        day1, day2 = tile_1['day'], tile_2['day']

        with rasterio.open(tile_1['path']) as tile_file:
            data_1 = tile_file.read([1, 2, 3])
        with rasterio.open(tile_2['path']) as tile_file:
            data_2 = tile_file.read([1, 2, 3])

        # def display_side_by_side(img1, img2):
        #     # Ensure both images have the same number of channels (3 for RGB)
        #     if img1.shape[0] != 3 or img2.shape[0] != 3:
        #         raise ValueError("Both images must have 3 channels (RGB)")
        #
        #     # Get dimensions of both images
        #     H1, W1 = img1.shape[1], img1.shape[2]
        #     H2, W2 = img2.shape[1], img2.shape[2]
        #
        #     # Determine the maximum height and total width needed for combined image
        #     max_height = max(H1, H2)
        #     total_width = W1 + W2
        #
        #     # Create a blank canvas for combined image
        #     combined_img = np.zeros((3, max_height, total_width), dtype=np.uint8)
        #
        #     # Paste img1 onto the canvas
        #     combined_img[:, :H1, :W1] = img1
        #
        #     # Paste img2 onto the canvas
        #     combined_img[:, :H2, W1:W1 + W2] = img2
        #
        #     # Convert from (3, max_height, total_width) to (total_width, max_height, 3) for display
        #     combined_img = np.transpose(combined_img, (2, 1, 0))
        #
        #     # Display the combined image using matplotlib
        #     plt.figure(figsize=(30, 15))  # Adjust figure size as needed
        #     plt.imshow(combined_img)
        #     plt.axis('off')
        #     plt.show()

        if data_1.shape[1] > self.image_size:
            # crop
            data_1_center = int(data_1.shape[1] / 2)
            data_1 = data_1[:,
                            data_1_center - self.image_size // 2:data_1_center + self.image_size // 2,
                            data_1_center - self.image_size // 2:data_1_center + self.image_size // 2,
                            ]
        elif data_1.shape[1] < self.image_size:
            # pad
            padding = (self.image_size - data_1.shape[1]) // 2
            padded_data = np.zeros((data_1.shape[0], self.image_size, self.image_size), dtype=data_1.dtype)
            padded_data[:, padding:padding + data_1.shape[1], padding:padding + data_1.shape[1]] = data_1
            data_1 = padded_data

        if data_2.shape[1] > self.image_size:
            # crop
            data_2_center = int(data_2.shape[2] / 2)
            data_2 = data_2[:,
                            data_2_center - self.image_size // 2:data_2_center + self.image_size // 2,
                            data_2_center - self.image_size // 2:data_2_center + self.image_size // 2,
                            ]
        elif data_2.shape[1] < self.image_size:
            # pad
            padding = (self.image_size - data_2.shape[1]) // 2
            padded_data = np.zeros((data_2.shape[0], self.image_size, self.image_size), dtype=data_2.dtype)
            padded_data[:, padding:padding + data_2.shape[1], padding:padding + data_2.shape[1]] = data_2
            data_2 = padded_data

        if self.transform:
            data_1 = self.transform(image=data_1.transpose((1, 2, 0)))['image'].transpose((2, 0, 1))
            data_2 = self.transform(image=data_2.transpose((1, 2, 0)))['image'].transpose((2, 0, 1))

        # print(label_1, label_2, label, margin, month1, month2, day1, day2)
        # display_side_by_side(data_1, data_2)

        data_1 = data_1 / 255
        data_2 = data_2 / 255

        if self.normalize:
            data_1 = normalize(data_1, self.mean, self.std)
            data_2 = normalize(data_2, self.mean, self.std)

        return data_1, data_2, month1, month2, day1, day2, label, margin


class SingleItemsSiameseSamplerDatasetWrapper:
    def __init__(self,
                 siamese_sampler_dataset: SiameseSamplerDataset):
        self.siamese_sampler_dataset = siamese_sampler_dataset

    def __len__(self):
        return len(self.siamese_sampler_dataset.all_samples_labels)

    def __getitem__(self, global_idx: int):
        dataset_key, dataset_idx = self.siamese_sampler_dataset.all_samples_indices[global_idx]
        label = self.siamese_sampler_dataset.all_samples_labels[global_idx]
        tile = self.siamese_sampler_dataset.datasets[dataset_key][dataset_idx]
        month, day = tile['month'], tile['day']

        with rasterio.open(tile['path']) as tile_file:
            data = tile_file.read([1, 2, 3])

        if data.shape[1] > self.siamese_sampler_dataset.image_size:
            # crop
            data_center = int(data.shape[1] / 2)
            data = data[:,
                        data_center - self.siamese_sampler_dataset.image_size // 2:data_center + self.siamese_sampler_dataset.image_size // 2,
                        data_center - self.siamese_sampler_dataset.image_size // 2:data_center + self.siamese_sampler_dataset.image_size // 2,
                        ]
        elif data.shape[1] < self.siamese_sampler_dataset.image_size:
            # pad
            padding = (self.siamese_sampler_dataset.image_size - data.shape[1]) // 2
            padded_data = np.zeros((data.shape[0], self.siamese_sampler_dataset.image_size, self.siamese_sampler_dataset.image_size), dtype=data.dtype)
            padded_data[:, padding:padding + data.shape[1], padding:padding + data.shape[1]] = data
            data = padded_data

        data = data / 255

        if self.siamese_sampler_dataset.normalize:
            data = normalize(data, self.siamese_sampler_dataset.mean, self.siamese_sampler_dataset.std)

        return data, month, day, label


class SiameseValidationDataset(BaseSiameseLabeledCocoDataset):
    def __init__(self,
                 fold: str,
                 root_path: Path or List[Path],
                 date_pattern: str,
                 image_size: int,
                 normalize: bool = True,
                 mean: np.array = FOREST_QPEB_MEAN,
                 std: np.array = FOREST_QPEB_STD):
        super().__init__(fold=fold, root_path=root_path, date_pattern=date_pattern, transform=None)

        self.image_size = image_size
        self.normalize = normalize
        self.mean = mean
        self.std = std

        for idx in self.tiles:
            assert len(self.tiles[idx]['labels']) == 1, \
                "SiameseValidationDataset dataset should have exactly one annotation label (polygon) per tile."

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx: int):
        tile = self.tiles[idx]

        with rasterio.open(tile['path']) as tile_file:
            data = tile_file.read([1, 2, 3])

        label = tile['labels'][0]['category_id']
        month, day = tile['month'], tile['day']

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

        if label is None:
            label = -1

        data = data / 255

        if self.normalize:
            data = normalize(data, self.mean, self.std)

        return data, month, day, label


