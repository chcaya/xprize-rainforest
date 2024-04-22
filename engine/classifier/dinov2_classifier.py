import itertools
import json
import math
import time
from functools import partial
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as tvf
from einops import rearrange
import geopandas as gpd
from geodataset.geodata import Raster
from geodataset.labels import RasterPolygonLabels
from matplotlib import pyplot as plt
from scipy import sparse
from shapely import box
from shapely.affinity import translate, affine_transform
from skimage.measure import block_reduce
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import albumentations as A


class LabeledDINOv2Dataset:
    def __init__(self,
                 rasters_labels_configs: List[Dict[str, Path or str]],
                 gpkg_aoi: Path,
                 ground_resolution: float or None,
                 scale_factor: float or None,
                 min_intersection_ratio: float,
                 output_size: int,
                 categories_coco: List[dict],
                 augment_data: bool):

        self.rasters_labels_configs = rasters_labels_configs
        self.gpkg_aoi = gpkg_aoi
        self.ground_resolution = ground_resolution
        self.scale_factor = scale_factor
        self.min_intersection_ratio = min_intersection_ratio
        self.output_size = output_size
        self.categories_coco = categories_coco
        self.augment_data = augment_data

        self.categories_mapping = self._setup_categories()
        self.rasters, self.polygons_labels, self.category_columns = self._load_rasters()
        self.aoi_polygons = self._load_aoi_polygon_for_each_raster()
        self.masks = self._load_masks_for_aoi()

        self.augmentation = A.Compose([
            A.Rotate(limit=45, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ])

    def _setup_categories(self):
        categories_mapping = {}
        for category in self.categories_coco:
            categories_mapping[category['name']] = category['id']
            if category['other_names']:
                for other_name in category['other_names']:
                    categories_mapping[other_name] = category['id']

        return categories_mapping

    def _load_rasters(self):
        rasters = {}
        polygons_labels = {}
        category_columns = {}
        for raster_config in self.rasters_labels_configs:
            raster = Raster(path=raster_config['raster_path'],
                            ground_resolution=self.ground_resolution,
                            scale_factor=self.scale_factor)
            labels = RasterPolygonLabels(path=raster_config['labels_path'],
                                         associated_raster=raster,
                                         main_label_category_column_name=raster_config['main_label_category_column_name'],
                                         other_labels_attributes_column_names=None)

            rasters[raster_config['raster_path']] = raster
            polygons_labels[raster_config['raster_path']] = labels.geometries_gdf
            category_columns[raster_config['raster_path']] = labels.main_label_category_column_name

        return rasters, polygons_labels, category_columns

    def _load_aoi_polygon_for_each_raster(self):
        aoi_polygon = gpd.read_file(self.gpkg_aoi)
        aoi_polygons = {}
        for raster_path in self.rasters.keys():
            aoi_polygon_for_raster = aoi_polygon.copy()
            aoi_polygon_for_raster = self.rasters[raster_path].adjust_geometries_to_raster_crs_if_necessary(gdf=aoi_polygon_for_raster)
            aoi_polygon_for_raster = self.rasters[raster_path].adjust_geometries_to_raster_pixel_coordinates(gdf=aoi_polygon_for_raster)
            aoi_polygons[raster_path] = aoi_polygon_for_raster
        return aoi_polygons

    @staticmethod
    def create_centroid_centered_mask(polygon, output_size):
        binary_mask = np.zeros((output_size, output_size), dtype=np.uint8)
        x, y = polygon.centroid.coords[0]
        x, y = int(x), int(y)
        mask_box = box(x - 0.5 * output_size,
                       y - 0.5 * output_size,
                       x + 0.5 * output_size,
                       y + 0.5 * output_size)

        polygon_intersection = mask_box.intersection(polygon)

        translated_polygon_intersection = translate(
            polygon_intersection,
            xoff=-mask_box.bounds[0],
            yoff=-mask_box.bounds[1]
        )

        contours = np.array(translated_polygon_intersection.exterior.coords).reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(binary_mask, [contours], 1)

        return binary_mask, mask_box

    def _load_masks_for_aoi(self):
        id = 0
        masks = {}
        for raster_path in self.rasters.keys():
            raster = self.rasters[raster_path]
            polygons = self.polygons_labels[raster_path].copy()

            # Get the polygons masks inside the aoi using min_intersection_ratio
            aoi_polygon = self.aoi_polygons[raster_path]
            polygons['polygons_id'] = polygons.index
            polygons['polygons_area'] = polygons.geometry.area

            intersections = gpd.overlay(polygons, aoi_polygon, how='intersection')
            intersections['intersection_area'] = intersections.geometry.area
            intersections['intersecting_ratio'] = intersections['intersection_area'] / intersections['polygons_area']

            # Filter geometries based on the threshold percentage
            intersections = intersections[intersections['intersecting_ratio'] > self.min_intersection_ratio]
            polygons = polygons.loc[polygons['polygons_id'].isin(intersections['polygons_id'])]

            polygons['centroid'] = polygons.centroid
            polygons['temp'] = polygons['geometry'].astype(object).apply(
                partial(self.create_centroid_centered_mask, output_size=self.output_size)
            )
            # Split the temporary column into the desired two columns
            polygons['mask'] = polygons['temp'].astype(object).apply(lambda x: x[0])
            polygons['mask_box'] = polygons['temp'].astype(object).apply(lambda x: x[1])

            # Drop the temporary column
            polygons.drop(columns=['temp'], inplace=True)

            for _, polygon_row in polygons.iterrows():
                mask_box = polygon_row['mask_box']
                mask_bounds = mask_box.bounds
                mask_bounds = [int(x) for x in mask_bounds]

                data = raster.data[
                       :3,                                  # removing Alpha channel if present
                       mask_bounds[1]:mask_bounds[3],
                       mask_bounds[0]:mask_bounds[2]
                ]

                masked_data = data * polygon_row['mask']
                masked_data = masked_data / 255.0
                # masked_data += 1.0 - polygon_row['mask']

                label = polygon_row[self.category_columns[raster_path]]
                if label in self.categories_mapping:
                    masks[id] = {
                        'masked_data': masked_data,
                        'label': self.categories_mapping[label],
                        'polygons_id': polygon_row['polygons_id'],
                        'raster_path': raster_path
                    }
                    id += 1

        return masks

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        mask_data = self.masks[idx]
        label = mask_data['label']
        mask = mask_data['masked_data']
        if self.augment_data:
            mask = self.augmentation(image=mask.transpose(1, 2, 0))['image']
            mask = mask.transpose(2, 0, 1)
        return mask, label


def display_image(image_data):
    image_display = np.transpose(image_data, (1, 2, 0))

    # Display the image using matplotlib
    plt.imshow(image_display)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()


class DINOv2Preprocessor:
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, vit_patch_size: int):
        self.vit_patch_size = vit_patch_size

    def _get_pad(self, size):
        new_size = math.ceil(size / self.vit_patch_size) * self.vit_patch_size
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    def preprocess(self, x: torch.Tensor):
        x = x.to(torch.float32)
        x = tvf.normalize(x, mean=list(self.MEAN), std=list(self.STD))
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        x = F.pad(x, pads)
        num_h_patches, num_w_patches = x.shape[2] // self.vit_patch_size, x.shape[3] // self.vit_patch_size
        return x, pads, num_h_patches, num_w_patches


class DINOv2Inference:
    SUPPORTED_SIZES = ['small', 'base', 'large', 'giant']

    def __init__(self, size: str):
        self.size = size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vit_patch_size = 14        # TODO put in config?
        self.model = self._load_model()
        self.preprocessor = DINOv2Preprocessor(self.vit_patch_size)

    def _load_model(self):
        assert self.size in self.SUPPORTED_SIZES, \
            f"Invalid DINOv2 model size: \'{self.size}\'. Valid value are {self.SUPPORTED_SIZES}."

        model_name = f"dinov2_vit{self.size[0]}{self.vit_patch_size}_reg"

        return torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True).to(self.device)

    def __call__(self, x: torch.Tensor):
        # with torch.inference_mode():
        pp_x, pads, num_h_patches, num_w_patches = self.preprocessor.preprocess(x)
        pp_x = pp_x.to(self.device)

        output = self.model(pp_x, is_training=True)
        output = output['x_norm_patchtokens']

        return output


class FCNNHead(nn.Module):
    def __init__(self, num_patches, embedding_dim, num_classes, dropout):
        super(FCNNHead, self).__init__()
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(num_patches * embedding_dim, 8192),  # First layer
            nn.ReLU(),
            nn.Linear(8192, 2048),  # Output layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),  # Output layer
            nn.ReLU(),
            nn.Linear(1024, 512),  # Output layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)  # Output layer
        )

    def forward(self, x):
        x = self.flatten(x)  # Flatten the input
        x = self.fc_layers(x)  # Pass through fully connected layers
        return x


def train_one_epoch(loader, dino_model, classifier_model, optimizer, criterion, device, writer, epoch_index):
    classifier_model.train()
    running_loss = 0.0
    running_steps = 0
    step = epoch_index * len(loader)
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            features = dino_model(inputs)

        features = features.detach().clone()

        outputs = classifier_model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if running_steps != 0 and step % 100 == 0:
            writer.add_scalar('Loss/train', running_loss / running_steps, step)
            running_loss = 0.0
            running_steps = 0

        step += 1
        running_steps += 1


def validate_one_epoch(loader, dino_model, classifier_model, criterion, device, writer, epoch_index, output_folder):
    classifier_model.eval()
    running_loss = 0.0
    best_f1 = 0
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            features = dino_model(inputs)
            outputs = classifier_model(features)
            # print("v", outputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate and log metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    accuracy = accuracy_score(all_labels, all_predictions)
    writer.add_scalar('Accuracy/validation', accuracy, epoch_index)
    writer.add_scalar('Precision/validation/macro', macro_precision, epoch_index)
    writer.add_scalar('Recall/validation/macro', macro_recall, epoch_index)
    writer.add_scalar('F1-Score/validation/macro', macro_f1, epoch_index)
    writer.add_scalar('Loss/validation', running_loss / len(loader), epoch_index)
    writer.add_scalar(f'Precision/validation/weighted', weighted_precision, epoch_index)
    writer.add_scalar('Recall/validation/weighted', weighted_recall, epoch_index)
    writer.add_scalar('F1-Score/validation/weighted', weighted_f1, epoch_index)

    output_folder = Path(output_folder)
    torch.save(classifier_model.state_dict(), output_folder / f'model_epoch_{epoch_index}.pth')

    # if macro_f1 > best_f1:
    #     best_f1 = macro_f1
    #     torch.save(classifier_model.state_dict(), output_folder / f'best_checkpoint.pth')
    #     print(f"New best model saved with F1-Score: {best_f1}")


def validate_save_gdf(dataset, loader, dino_model, classifier_model, device, output_folder):
    classifier_model.eval()
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            features = dino_model(inputs)
            outputs = classifier_model(features)

            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate and log metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_labels, all_predictions,
                                                                                 average='macro')
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(all_labels, all_predictions,
                                                                                          average='weighted')
    accuracy = accuracy_score(all_labels, all_predictions)

    print(f"Accuracy: {accuracy}")
    print(f"Macro Precision: {macro_precision}")
    print(f"Macro Recall: {macro_recall}")
    print(f"Macro F1: {macro_f1}")
    print(f"Weighted Precision: {weighted_precision}")
    print(f"Weighted Recall: {weighted_recall}")
    print(f"Weighted F1: {weighted_f1}")

    df_data = []
    for i in range(len(dataset.masks)):
        mask_data = dataset.masks[i]

        df_data.append(
                {
                    'label': mask_data['label'],
                    'prediction': all_predictions[i],
                    'raster_path': mask_data['raster_path'],
                    'polygons_id': mask_data['polygons_id']
                }
        )

    df = pd.DataFrame(df_data)
    df['geometry'] = None

    for i, row in df.iterrows():
        df.loc[i, 'geometry'] = dataset.polygons_labels[row['raster_path']].loc[row['polygons_id']]['geometry']

    gdfs = []
    for raster_path in df['raster_path'].unique():
        raster = dataset.rasters[raster_path]

        raster_labels_data = df.loc[df['raster_path'] == raster_path].copy(deep=True)

        transform = raster.metadata['transform']

        shapely_transform = (
            transform[0],  # a (scale factor for x)
            transform[1],  # b (rotation factor, x to y, usually 0 in north-up images)
            transform[3],  # d (rotation factor, y to x, usually 0)
            transform[4],  # e (scale factor for y)
            transform[2],  # xoff (translation offset for x)
            transform[5]  # yoff (translation offset for y)
        )

        raster_labels_data['geometry'] = raster_labels_data['geometry'].apply(
            lambda x: affine_transform(x, shapely_transform)
        )
        gdf = gpd.GeoDataFrame(raster_labels_data, crs=raster.metadata['crs'], geometry='geometry')
        gdfs.append(gdf)

    common_crs = gdfs[0].crs
    for gdf in gdfs:
        gdf.to_crs(common_crs, inplace=True)

    final_gdf = gpd.GeoDataFrame(pd.concat(gdfs), crs=common_crs)
    final_gdf['raster_name'] = final_gdf['raster_path'].apply(lambda x: Path(x).name)
    final_gdf.drop(columns=['raster_path', 'polygons_id'], inplace=True)

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_folder) / 'predictions.gpkg'
    final_gdf.to_file(str(output_path), driver='GPKG')


def train_main():
    image_size = 224
    assert image_size % 14 == 0, "Output size must be a multiple of 14"

    rasters_labels_configs = [
        {
            'raster_path': Path(
                'C:/Users/Hugo/Documents/Data/raw/quebec_trees_dataset_2021-09-02/2021-09-02/zone1/2021-09-02-sbl-z1-rgb-cog.tif'),
            'labels_path': Path('C:/Users/Hugo/Documents/Data/raw/quebec_trees_dataset_2021-09-02/Z1_polygons.gpkg'),
            'main_label_category_column_name': 'Label'
        },
        {
            'raster_path': Path(
                'C:/Users/Hugo/Documents/Data/raw/quebec_trees_dataset_2021-09-02/2021-09-02/zone2/2021-09-02-sbl-z2-rgb-cog.tif'),
            'labels_path': Path('C:/Users/Hugo/Documents/Data/raw/quebec_trees_dataset_2021-09-02/Z2_polygons.gpkg'),
            'main_label_category_column_name': 'Label'
        },
        {
            'raster_path': Path(
                'C:/Users/Hugo/Documents/Data/raw/quebec_trees_dataset_2021-09-02/2021-09-02/zone3/2021-09-02-sbl-z3-rgb-cog.tif'),
            'labels_path': Path('C:/Users/Hugo/Documents/Data/raw/quebec_trees_dataset_2021-09-02/Z3_polygons.gpkg'),
            'main_label_category_column_name': 'Label'
        },
    ]

    train_dataset = LabeledDINOv2Dataset(
        rasters_labels_configs=rasters_labels_configs,
        gpkg_aoi=Path('C:/Users/Hugo/PycharmProjects/geodataset/geodataset/utils/aois/quebec_trees/train_aoi.geojson'),
        ground_resolution=0.05,
        scale_factor=None,
        min_intersection_ratio=0.7,
        output_size=image_size,  # Multiple of 14 to avoid padding
        categories_coco=json.load(open(
            'C:/Users/Hugo/PycharmProjects/geodataset/geodataset/utils/categories/quebec_trees/quebec_trees_categories.json',
            "rb"))['categories'],
        augment_data=True
    )

    valid_dataset = LabeledDINOv2Dataset(
        rasters_labels_configs=rasters_labels_configs,
        gpkg_aoi=Path('C:/Users/Hugo/PycharmProjects/geodataset/geodataset/utils/aois/quebec_trees/valid_aoi.geojson'),
        ground_resolution=0.05,
        scale_factor=None,
        min_intersection_ratio=0.7,
        output_size=image_size,  # Multiple of 14 to avoid padding
        categories_coco=json.load(open(
            'C:/Users/Hugo/PycharmProjects/geodataset/geodataset/utils/categories/quebec_trees/quebec_trees_categories.json',
            "rb"))['categories'],
        augment_data=False
    )

    print("Train dataset size:", len(train_dataset))
    print("Valid dataset size:", len(valid_dataset))

    # Configuration and initialization
    num_classes = max(train_dataset.categories_mapping.values()) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_model = DINOv2Inference(size='large')
    classification_head = FCNNHead(
        embedding_dim=1024,
        num_patches=(image_size // 14) ** 2,
        num_classes=num_classes,
        dropout=0.2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classification_head.parameters(), lr=0.00005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    output_folder = f'C:/Users/Hugo/PycharmProjects/xprize-rainforest/output/classifier_dinov2_small/{time.time()}'
    writer = SummaryWriter(output_folder)

    collate_fn = lambda x: (torch.stack([torch.tensor(data[0]) for data in x]), torch.tensor([data[1] for data in x]))

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn,
                              shuffle=True)  # Define your training DataLoader
    val_loader = DataLoader(valid_dataset, batch_size=2, collate_fn=collate_fn)  # Define your validation DataLoader

    num_epochs = 100
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_one_epoch(train_loader, dino_model, classification_head, optimizer, criterion, device, writer, epoch)
        validate_one_epoch(val_loader, dino_model, classification_head, criterion, device, writer, epoch, output_folder)
        scheduler.step()

    writer.close()


def valid_main():
    image_size = 224
    assert image_size % 14 == 0, "Output size must be a multiple of 14"

    rasters_labels_configs = [
        {
            'raster_path': Path(
                'C:/Users/Hugo/Documents/Data/raw/quebec_trees_dataset_2021-09-02/2021-09-02/zone1/2021-09-02-sbl-z1-rgb-cog.tif'),
            'labels_path': Path('C:/Users/Hugo/Documents/Data/raw/quebec_trees_dataset_2021-09-02/Z1_polygons.gpkg'),
            'main_label_category_column_name': 'Label'
        },
        {
            'raster_path': Path(
                'C:/Users/Hugo/Documents/Data/raw/quebec_trees_dataset_2021-09-02/2021-09-02/zone2/2021-09-02-sbl-z2-rgb-cog.tif'),
            'labels_path': Path('C:/Users/Hugo/Documents/Data/raw/quebec_trees_dataset_2021-09-02/Z2_polygons.gpkg'),
            'main_label_category_column_name': 'Label'
        },
        {
            'raster_path': Path(
                'C:/Users/Hugo/Documents/Data/raw/quebec_trees_dataset_2021-09-02/2021-09-02/zone3/2021-09-02-sbl-z3-rgb-cog.tif'),
            'labels_path': Path('C:/Users/Hugo/Documents/Data/raw/quebec_trees_dataset_2021-09-02/Z3_polygons.gpkg'),
            'main_label_category_column_name': 'Label'
        },
    ]

    valid_dataset = LabeledDINOv2Dataset(
        rasters_labels_configs=rasters_labels_configs,
        gpkg_aoi=Path('C:/Users/Hugo/PycharmProjects/geodataset/geodataset/utils/aois/quebec_trees/valid_aoi.geojson'),
        ground_resolution=0.05,
        scale_factor=None,
        min_intersection_ratio=0.7,
        output_size=image_size,  # Multiple of 14 to avoid padding
        categories_coco=json.load(open(
            'C:/Users/Hugo/PycharmProjects/geodataset/geodataset/utils/categories/quebec_trees/quebec_trees_categories.json',
            "rb"))['categories'],
        augment_data=False
    )

    print("Valid dataset size:", len(valid_dataset))

    num_classes = max(valid_dataset.categories_mapping.values()) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_model = DINOv2Inference(size='small')
    classification_head = FCNNHead(
        embedding_dim=384,
        num_patches=(image_size // 14) ** 2,
        num_classes=num_classes,
        dropout=0.2
    ).to(device)
    classification_head.load_state_dict(torch.load('C:/Users/Hugo/PycharmProjects/xprize-rainforest/output/classifier_dinov2_small/1713653391.783514/model_epoch_20.pth'))

    collate_fn = lambda x: (torch.stack([torch.tensor(data[0]) for data in x]), torch.tensor([data[1] for data in x]))
    val_loader = DataLoader(valid_dataset, batch_size=2, collate_fn=collate_fn)  # Define your validation DataLoader
    output_folder = f'C:/Users/Hugo/PycharmProjects/xprize-rainforest/output/classifier_dinov2_small/gdf_output'
    validate_save_gdf(valid_dataset, val_loader, dino_model, classification_head, device, output_folder)



if __name__ == "__main__":
    # train_main()
    valid_main()
