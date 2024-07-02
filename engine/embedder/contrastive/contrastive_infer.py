from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from geodataset.utils import rle_segmentation_to_mask, mask_to_polygon, tiles_polygons_gdf_to_crs_gdf, \
    GeoPackageNameConvention
from torch import nn
from tqdm import tqdm

from engine.embedder.contrastive.contrastive_dataset import ContrastiveDataset, ContrastiveInternalDataset, \
    ContrastiveInferDataset
from engine.embedder.contrastive.contrastive_model import XPrizeTreeEmbedder2NoDate, XPrizeTreeEmbedder, \
    XPrizeTreeEmbedder2
from engine.embedder.contrastive.contrastive_utils import ConditionalAutocast, FOREST_QPEB_MEAN, FOREST_QPEB_STD, \
    IMAGENET_MEAN, IMAGENET_STD, contrastive_infer_collate_fn


def contrastive_classifier_embedder_infer(data_roots: str or List[str],
                                          fold: str,
                                          day_month_year: Tuple[int, int, int],
                                          image_size: int,
                                          mean_std_descriptor: str,
                                          contrastive_checkpoint: str,
                                          batch_size: int,
                                          product_name: str,
                                          ground_resolution: float,
                                          scale_factor: float,
                                          output_folder: Path):

    if mean_std_descriptor == 'forest_qpeb':
        mean = FOREST_QPEB_MEAN
        std = FOREST_QPEB_STD
    elif mean_std_descriptor == 'imagenet':
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    else:
        raise ValueError(f'Unknown mean_std_descriptor: {mean_std_descriptor}')

    dataset = ContrastiveInferDataset(
        root_path=data_roots,
        date_pattern=None,
        day_month_year=day_month_year,
        fold=fold,
        image_size=image_size,
        transform=None,
        normalize=True,
        mean=mean,
        std=std,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # not shuffling is REQUIRED for the inference (order is important)
        num_workers=4,
        collate_fn=contrastive_infer_collate_fn
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Loading model from {contrastive_checkpoint}')
    model = XPrizeTreeEmbedder2NoDate.from_checkpoint(contrastive_checkpoint).to(device)
    model.eval()

    embeddings, predicted_families, predicted_families_scores = infer_model_without_labels(
        model=model, dataloader=loader, device=device, use_mixed_precision=False, desc='Infering...'
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    tiles_paths = []
    polygons = []

    for tile_idx in range(len(dataset)):
        tile = dataset.tiles[tile_idx]
        label = tile['labels'][0]
        segmentation = label['segmentation']
        if ('is_rle_format' in label and label['is_rle_format']) or isinstance(segmentation, dict):
            # RLE format
            mask = rle_segmentation_to_mask(segmentation)
            polygon = mask_to_polygon(mask)
            polygons.append(polygon)
            tiles_paths.append(str(tile['path']))
        else:
            raise NotImplementedError("Please make sure that the masks are encoded using RLE.")

    tiles_polygons_gdf = gpd.GeoDataFrame({
        'a_id': [str(x) for x in range(len(polygons))],     # this is just a dummy column so that QGIS doesn't use the 'embeddings' column as base label for the geometries
        'polygon_id': range(len(polygons)),
        'geometry': polygons,
        'embeddings': embeddings.tolist(),
        'predicted_family': predicted_families,
        'predicted_family_scores': predicted_families_scores.tolist(),
        'tile_path': tiles_paths
    })

    tiles_polygons_gdf_crs = tiles_polygons_gdf_to_crs_gdf(tiles_polygons_gdf)

    tiles_polygons_gdf_crs['area'] = tiles_polygons_gdf_crs['geometry'].area

    geopackage_name = GeoPackageNameConvention.create_name(
        product_name=product_name,
        fold='inferembedderclassifier',
        ground_resolution=ground_resolution,
        scale_factor=scale_factor,
    )

    tiles_polygons_gdf_crs['embeddings'] = tiles_polygons_gdf_crs['embeddings'].apply(lambda x: str(x))

    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / geopackage_name
    tiles_polygons_gdf_crs.to_file(output_path, driver='GPKG')
    print(f"Successfully saved the embeddings and classification predictions at {output_path}.")

    return tiles_polygons_gdf_crs, output_path


def infer_model_without_labels(model, dataloader, device, use_mixed_precision, desc='Infering...', as_numpy=True):
    all_embeddings = torch.tensor([])
    all_predicted_families = []
    all_predicted_families_scores = torch.tensor([])

    for images, months, days in tqdm(dataloader, total=len(dataloader), desc=desc):
        embeddings, predicted_families, predicted_families_scores = infer_batch(
            images=images, months=months, days=days, model=model,
            device=device, use_mixed_precision=use_mixed_precision
        )

        all_embeddings = torch.cat((all_embeddings, embeddings.detach().cpu()), dim=0)
        all_predicted_families.extend(predicted_families)
        all_predicted_families_scores = torch.cat(
            (all_predicted_families_scores, predicted_families_scores.detach().cpu()), dim=0)

    if as_numpy:
        all_embeddings = all_embeddings.numpy()
        all_predicted_families = np.array(all_predicted_families)
        all_predicted_families_scores = all_predicted_families_scores.numpy()

    return all_embeddings, all_predicted_families, all_predicted_families_scores


def infer_model_with_labels(model, dataloader, device, use_mixed_precision, desc='Infering...', as_numpy=True):
    all_labels = []
    all_labels_ids = torch.tensor([])
    all_families = []
    all_families_ids = torch.tensor([])
    all_embeddings = torch.tensor([])
    all_predicted_families = []
    all_predicted_families_scores = torch.tensor([])
    with torch.no_grad():
        for images, months, days, labels_ids, labels, families_ids, families in tqdm(dataloader, total=len(dataloader),
                                                                                     desc=desc):
            embeddings, predicted_families, predicted_families_scores = infer_batch(
                images=images, months=months, days=days, model=model,
                device=device, use_mixed_precision=use_mixed_precision
            )

            all_labels.extend(labels)
            all_labels_ids = torch.cat((all_labels_ids, labels_ids.detach().cpu()), dim=0)
            all_families.extend(families)
            all_families_ids = torch.cat((all_families_ids, families_ids.detach().cpu()), dim=0)
            all_embeddings = torch.cat((all_embeddings, embeddings.detach().cpu()), dim=0)
            all_predicted_families.extend(predicted_families)
            all_predicted_families_scores = torch.cat(
                (all_predicted_families_scores, predicted_families_scores.detach().cpu()), dim=0)

        if as_numpy:
            all_labels = np.array(all_labels)
            all_labels_ids = all_labels_ids.numpy()
            all_families = np.array(all_families)
            all_families_ids = all_families_ids.numpy()
            all_embeddings = all_embeddings.numpy()
            all_predicted_families = np.array(all_predicted_families)
            all_predicted_families_scores = all_predicted_families_scores.numpy()

        return all_labels, all_labels_ids, all_families, all_families_ids, all_embeddings, all_predicted_families, all_predicted_families_scores


def infer_batch(images, months, days, model, device, use_mixed_precision):
    with torch.no_grad():
        data = torch.Tensor(images).to(device)
        months = torch.Tensor(months).to(device)
        days = torch.Tensor(days).to(device)
        if len(data.shape) == 3:
            data = data.unsqueeze(0)

        if isinstance(model, nn.DataParallel):
            actual_model = model.module
        else:
            actual_model = model

        with ConditionalAutocast(use_mixed_precision):
            if isinstance(actual_model, XPrizeTreeEmbedder2NoDate):
                output = model(data)
            else:
                output = model(data, months, days)

        if isinstance(actual_model, XPrizeTreeEmbedder):
            embeddings = output
            predicted_families_scores = None
            predicted_families = [None] * len(images)
        elif isinstance(actual_model, (XPrizeTreeEmbedder2, XPrizeTreeEmbedder2NoDate)):
            embeddings, classifier_logits = output[0], output[1]
            predicted_families_ids = torch.argmax(classifier_logits, dim=1)
            predicted_families_scores = torch.max(torch.softmax(classifier_logits, dim=1), dim=1)[0]
            predicted_families = [actual_model.ids_to_families_mapping[int(family_id)] for family_id in
                                  predicted_families_ids]
        else:
            raise ValueError(f'Unknown model type: {actual_model.__class__}')

        return embeddings, predicted_families, predicted_families_scores
