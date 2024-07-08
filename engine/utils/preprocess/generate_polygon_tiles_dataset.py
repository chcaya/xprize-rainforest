import importlib.resources as pkg_resources
import json
from pathlib import Path

import geodataset.utils.aois as aois
import geodataset.utils.categories as categories
from geodataset.aoi import AOIFromPackageConfig
from geodataset.tilerize import PolygonTilerizer


def tilerize_quebec_trees(output_folder: Path, ground_resolution: float, max_tile_size: int):
    aoi_config = AOIFromPackageConfig(aois={
        'train': pkg_resources.files(aois) / 'quebec_trees/train_aoi.geojson',
        'valid': pkg_resources.files(aois) / 'quebec_trees/valid_aoi.geojson',
        'test': pkg_resources.files(aois) / 'quebec_trees/inference_zone.gpkg'
    })

    labels = [
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/quebec_trees/from_annotations/Z1_polygons_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/quebec_trees/from_annotations/Z2_polygons_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/quebec_trees/from_annotations/Z3_polygons_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/quebec_trees/from_boxes/2021_09_02_sbl_z1_rgb_cog_SAMthreshold0p5_from_boxes_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/quebec_trees/from_boxes/2021_09_02_sbl_z2_rgb_cog_SAMthreshold0p5_from_boxes_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/quebec_trees/from_boxes/2021_09_02_sbl_z3_rgb_cog_SAMthreshold0p5_from_boxes_species.gpkg',
    ]

    rasters = [
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/quebec_trees_dataset/quebec_trees_dataset_2021-09-02/2021-09-02/zone1/2021-09-02-sbl-z1-rgb-cog.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/quebec_trees_dataset/quebec_trees_dataset_2021-09-02/2021-09-02/zone2/2021-09-02-sbl-z2-rgb-cog.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/quebec_trees_dataset/quebec_trees_dataset_2021-09-02/2021-09-02/zone3/2021-09-02-sbl-z3-rgb-cog.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/quebec_trees_dataset/quebec_trees_dataset_2021-09-02/2021-09-02/zone1/2021-09-02-sbl-z1-rgb-cog.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/quebec_trees_dataset/quebec_trees_dataset_2021-09-02/2021-09-02/zone2/2021-09-02-sbl-z2-rgb-cog.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/quebec_trees_dataset/quebec_trees_dataset_2021-09-02/2021-09-02/zone3/2021-09-02-sbl-z3-rgb-cog.tif',
    ]

    output_suffixes = [
        'from_annotations',
        'from_annotations',
        'from_annotations',
        'from_boxes',
        'from_boxes',
        'from_boxes',
    ]

    tifs = list(Path('/media/hugobaudchon/4 TB/XPrize/Data/raw/quebec_trees_dataset').rglob('*.tif'))

    for gpkg in list(Path('/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/quebec_trees/from_detector').rglob('*.gpkg')):
        labels.append(gpkg)
        rasters.append(next(tif for tif in tifs if "-".join(gpkg.stem.split("_")[:5]) in tif.name))
        output_suffixes.append('from_detector')

    for label, raster, output_suffix in zip(labels, rasters, output_suffixes):
        tilerizer = PolygonTilerizer(
            raster_path=Path(raster),
            labels_path=Path(label),
            output_path=output_folder / f'quebec_trees/{output_suffix}',
            output_name_suffix=output_suffix,
            ground_resolution=ground_resolution,
            scale_factor=None,
            use_variable_tile_size=True,
            variable_tile_size_pixel_buffer=10,
            aois_config=aoi_config,
            tile_size=max_tile_size,
            coco_categories_list=json.load(open(str(pkg_resources.files(categories) / 'quebec_trees/quebec_trees_categories.json')))['categories'],
            main_label_category_column_name='Label'
        )

        tilerizer.generate_coco_dataset()


def tilerize_brazil_trees(output_folder: Path, ground_resolution: float, max_tile_size: int):
    aoi_config = AOIFromPackageConfig(aois={
        'train': pkg_resources.files(aois) / 'brazil_zf2_trees/brazil_zf2_species_train_aoi.gpkg',
        'valid': pkg_resources.files(aois) / 'brazil_zf2_trees/brazil_zf2_species_valid_aoi.gpkg',
        'test': pkg_resources.files(aois) / 'brazil_zf2_trees/brazil_zf2_species_test_aoi.gpkg'
    })

    labels = [
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/brazil_zf2/from_boxes/20240130_zf2quad_m3m_rgb_SAMthreshold0p2_from_boxes_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/brazil_zf2/from_detector/20231213_zf2campirana_mini3pro_rgb_aligned_SAMthreshold1p0_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/brazil_zf2/from_detector/20240130_zf2quad_m3m_rgb_SAMthreshold0p2_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/brazil_zf2/from_detector/20240131_zf2campirana_m3m_rgb_SAMthreshold1p0_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/brazil_zf2/from_detector/20240519_zf2campinarana_m3m_rgb_aligned_SAMthreshold1p0_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/brazil_zf2/from_detector/20240520_zf2quad_m3m_rgb_SAMthreshold0p2_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/brazil_zf2/from_detector/20240521_zf2100ha_highres_m3m_rgb_SAMthreshold1p0_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/brazil_zf2/from_detector/20240523_zf2quad_subsample_m3m_rgb_aligned_SAMthreshold0p2_species.gpkg'
    ]

    rasters = [
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/brazil_zf2/20240130_zf2quad_m3m/20240130_zf2quad_m3m_rgb.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/brazil_zf2/20231213_zf2_campirana_mini3pro_aligned/20231213_zf2campirana_mini3pro_rgb_aligned.cog.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/brazil_zf2/20240130_zf2quad_m3m/20240130_zf2quad_m3m_rgb.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/brazil_zf2/20240131_zf2campirana_m3m/20240131_zf2campirana_m3m_rgb.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/brazil_zf2/20240519_zf2_campirana_m3m_aligned/20240519_zf2campinarana_m3m_rgb_aligned.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/brazil_zf2/20240520_zf2quad_m3m/20240520_zf2quad_m3m_rgb.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/brazil_zf2/20240521_zf2100ha_highres_m3m/20240521_zf2100ha_highres_m3m_rgb.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/brazil_zf2/20240523_zf2quad_subsample_m3m_aligned/20240523_zf2quad_subsample_m3m_rgb_aligned.tif'
    ]

    output_suffixes = [
        'from_boxes',
        'from_detector',
        'from_detector',
        'from_detector',
        'from_detector',
        'from_detector',
        'from_detector',
        'from_detector',
    ]

    for label, raster, output_suffix in zip(labels, rasters, output_suffixes):
        tilerizer = PolygonTilerizer(
            raster_path=Path(raster),
            labels_path=Path(label),
            output_path=output_folder / f'brazil_zf2/{output_suffix}',
            output_name_suffix=output_suffix,
            ground_resolution=ground_resolution,
            scale_factor=None,
            use_variable_tile_size=True,
            variable_tile_size_pixel_buffer=10,
            aois_config=aoi_config,
            tile_size=max_tile_size,
            coco_categories_list=json.load(open(str(pkg_resources.files(categories) / 'brazil_zf2_trees/brazil_zf2_trees_categories.json')))['categories'],
            main_label_category_column_name='canonicalName'
        )

        tilerizer.generate_coco_dataset()


def tilerize_brazil_trees_additional_data(output_folder: Path, ground_resolution: float, max_tile_size: int):
    aoi_config = AOIFromPackageConfig(aois={
        'train': pkg_resources.files(aois) / 'brazil_zf2_trees/brazil_zf2_species_train_aoi.gpkg',
        'valid': pkg_resources.files(aois) / 'brazil_zf2_trees/brazil_zf2_species_valid_aoi.gpkg',
        'test': pkg_resources.files(aois) / 'brazil_zf2_trees/brazil_zf2_species_test_aoi.gpkg'
    })

    labels = [
        '/media/hugo/Hard Disk 1/XPrize/Data/raw/brazil_zf2_new/20240130_zf2tower_ms_m3m_labels_aligned_SAMthreshold1p0.gpkg',
        '/media/hugo/Hard Disk 1/XPrize/Data/raw/brazil_zf2_new/20240130_zf2transectew_m3m_labels_aligned_SAMthreshold1p0.gpkg',
        '/media/hugo/Hard Disk 1/XPrize/Data/raw/brazil_zf2_new/20240131_zf2block4_ms_m3m_labels_aligned_SAMthreshold1p0.gpkg',
        '/media/hugo/Hard Disk 1/XPrize/Data/raw/brazil_zf2_new/20240520_zf2quad_m3m_labels_aligned_SAMthreshold1p0.gpkg',
    ]

    rasters = [
        '/home/hugo/Documents/xprize/data/20240130_zf2tower_ms_m3m_rgb.cog.tif',
        '/media/hugo/Hard Disk 1/XPrize/Data/raw/brazil_zf2_new/20240130_zf2transectew_m3m_rgb.tif',
        '/home/hugo/Documents/xprize/data/20240131_zf2block4_ms_m3m_rgb.cog.tif',
        '/home/hugo/Documents/xprize/data/20240520_zf2quad_m3m_rgb.cog.tif',
    ]

    output_suffixes = [
        'from_detector',
        'from_detector',
        'from_detector',
        'from_detector',
    ]

    for label, raster, output_suffix in zip(labels, rasters, output_suffixes):
        tilerizer = PolygonTilerizer(
            raster_path=Path(raster),
            labels_path=Path(label),
            output_path=output_folder / f'brazil_zf2_additional_data/{output_suffix}',
            output_name_suffix=output_suffix,
            ground_resolution=ground_resolution,
            scale_factor=None,
            use_variable_tile_size=True,
            variable_tile_size_pixel_buffer=10,
            aois_config=aoi_config,
            tile_size=max_tile_size,
            coco_categories_list=json.load(open(str(pkg_resources.files(categories) / 'brazil_zf2_trees/brazil_zf2_trees_categories.json')))['categories'],
            main_label_category_column_name='canonicalName'
        )

        tilerizer.generate_coco_dataset()


def tilerize_equator_trees(output_folder: Path, ground_resolution: float, max_tile_size: int):
    aoi_config = AOIFromPackageConfig(aois={
        'train': pkg_resources.files(aois) / 'equator_tiputini_trees/equator_species_train_aoi.gpkg',
        'valid': pkg_resources.files(aois) / 'equator_tiputini_trees/equator_species_valid_aoi.gpkg',
        'test': pkg_resources.files(aois) / 'equator_tiputini_trees/equator_species_test_aoi.gpkg'
    })

    labels = [
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/equator/from_boxes/20170810_transectotoni_mavicpro_rgb_SAMthreshold1p0_from_boxes_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/equator/from_boxes/20230525_tbslake_m3e_rgb_SAMthreshold1p0_from_boxes_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/equator/from_boxes/20231018_inundated_m3e_rgb_SAMthreshold1p0_from_boxes_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/equator/from_boxes/20231018_pantano_m3e_rgb_SAMthreshold1p0_from_boxes_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/equator/from_boxes/20231018_terrafirme_m3e_rgb_SAMthreshold1p0_from_boxes_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/equator/from_detector/20170810_transectotoni_mavicpro_rgb_SAMthreshold1p0_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/equator/from_detector/20230525_tbslake_m3e_rgb_SAMthreshold1p0_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/equator/from_detector/20231018_inundated_m3e_rgb_SAMthreshold1p0_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/equator/from_detector/20231018_pantano_m3e_rgb_SAMthreshold1p0_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/equator/from_detector/20231018_terrafirme_m3e_rgb_SAMthreshold1p0_species.gpkg',
    ]

    rasters = [
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/20170810_transectotoni_mavicpro/20170810_transectotoni_mavicpro_rgb.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/20230525_tbslake_m3e/20230525_tbslake_m3e_rgb.cog.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/20231018_inundated_m3e/20231018_inundated_m3e_rgb.cog.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/20231018_pantano_m3e/20231018_pantano_m3e_rgb.cog.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/20231018_terrafirme_m3e/20231018_terrafirme_m3e_rgb.cog.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/20170810_transectotoni_mavicpro/20170810_transectotoni_mavicpro_rgb.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/20230525_tbslake_m3e/20230525_tbslake_m3e_rgb.cog.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/20231018_inundated_m3e/20231018_inundated_m3e_rgb.cog.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/20231018_pantano_m3e/20231018_pantano_m3e_rgb.cog.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/equator/20231018_terrafirme_m3e/20231018_terrafirme_m3e_rgb.cog.tif'
    ]

    output_suffixes = [
        'from_boxes',
        'from_boxes',
        'from_boxes',
        'from_boxes',
        'from_boxes',
        'from_detector',
        'from_detector',
        'from_detector',
        'from_detector',
        'from_detector'
    ]

    for label, raster, output_suffix in zip(labels, rasters, output_suffixes):
        tilerizer = PolygonTilerizer(
            raster_path=Path(raster),
            labels_path=Path(label),
            output_path=output_folder / f'equator/{output_suffix}',
            output_name_suffix=output_suffix,
            ground_resolution=ground_resolution,
            scale_factor=None,
            use_variable_tile_size=True,
            variable_tile_size_pixel_buffer=10,
            aois_config=aoi_config,
            tile_size=max_tile_size,
            coco_categories_list=json.load(open(str(pkg_resources.files(categories) / 'equator_tiputini_trees/equator_tiputini_trees_categories.json')))['categories'],
            main_label_category_column_name='canonicalName'
        )

        tilerizer.generate_coco_dataset()


def tilerize_panama_trees(output_folder: Path, ground_resolution: float, max_tile_size: int):
    aoi_config = AOIFromPackageConfig(aois={
        'train': pkg_resources.files(aois) / 'panama_bci_trees/panama_bci_species_train_aoi.gpkg',
        'valid': pkg_resources.files(aois) / 'panama_bci_trees/panama_bci_species_valid_aoi.gpkg',
        'test': pkg_resources.files(aois) / 'panama_bci_trees/panama_bci_species_test_aoi.gpkg'
    })

    labels = [
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/panama/from_annotations/20200801_bci50ha_p4pro_labels_masks_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/panama/from_annotations/20220929_bci50ha_p4pro_labels_masks_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/panama/from_boxes/20200801_bci50ha_p4pro_rgb_SAMthreshold0p5_from_boxes_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/panama/from_boxes/20220929_bci50ha_p4pro_rgb_SAMthreshold0p5_from_boxes_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/panama/from_detector/20200801_bci50ha_p4pro_rgb_SAMthreshold0p5_species.gpkg',
        '/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/panama/from_detector/20220929_bci50ha_p4pro_rgb_SAMthreshold0p5_species.gpkg'
    ]

    rasters = [
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/panama/20200801_bci50ha_p4pro/20200801_bci50ha_p4pro_rgb.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/panama/20220929_bci50ha_p4pro/20220929_bci50ha_p4pro_rgb.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/panama/20200801_bci50ha_p4pro/20200801_bci50ha_p4pro_rgb.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/panama/20220929_bci50ha_p4pro/20220929_bci50ha_p4pro_rgb.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/panama/20200801_bci50ha_p4pro/20200801_bci50ha_p4pro_rgb.tif',
        '/media/hugobaudchon/4 TB/XPrize/Data/raw/panama/20220929_bci50ha_p4pro/20220929_bci50ha_p4pro_rgb.tif'
    ]

    output_suffixes = [
        'from_annotations',
        'from_annotations',
        'from_boxes',
        'from_boxes',
        'from_detector',
        'from_detector'
    ]

    tifs = list(Path('/media/hugobaudchon/4 TB/XPrize/Data/raw/panama/BCI_50ha_timeseries_local_alignment').rglob('*.tif'))

    for gpkg in list(Path('/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched/panama/from_detector').rglob('*.gpkg')):
        if '_local' not in gpkg.stem:
            continue
        labels.append(gpkg)
        rasters.append([tif for tif in tifs if "_".join(gpkg.stem.split("_")[:6]) in tif.name.lower()][0])
        output_suffixes.append('from_detector')

    for label, raster, output_suffix in zip(labels, rasters, output_suffixes):
        tilerizer = PolygonTilerizer(
            raster_path=Path(raster),
            labels_path=Path(label),
            output_path=output_folder / f'panama/{output_suffix}',
            output_name_suffix=output_suffix,
            ground_resolution=ground_resolution,
            scale_factor=None,
            use_variable_tile_size=True,
            variable_tile_size_pixel_buffer=10,
            aois_config=aoi_config,
            tile_size=max_tile_size,
            coco_categories_list=json.load(open(str(pkg_resources.files(categories) / 'panama_bci_trees/panama_bci_trees_categories.json')))['categories'],
            main_label_category_column_name='canonicalName'
        )

        tilerizer.generate_coco_dataset()


if __name__ == "__main__":
    source_folder = Path('/media/hugobaudchon/4 TB/XPrize/Data/SAM_segmentations_matched')

    ground_resolution = 0.03
    max_tile_size = 1536

    output_folder = Path(f'/home/hugo/Documents/xprize/data/FINAL_polygon_dataset_{max_tile_size}px_gr{str(ground_resolution).replace(".", "p")}')

    # tilerize_quebec_trees(output_folder=output_folder, ground_resolution=ground_resolution, max_tile_size=max_tile_size)
    # tilerize_brazil_trees(output_folder=output_folder, ground_resolution=ground_resolution, max_tile_size=max_tile_size)
    # tilerize_equator_trees(output_folder=output_folder, ground_resolution=ground_resolution, max_tile_size=max_tile_size)
    # tilerize_panama_trees(output_folder=output_folder, ground_resolution=ground_resolution, max_tile_size=max_tile_size)
    tilerize_brazil_trees_additional_data(output_folder=output_folder, ground_resolution=ground_resolution, max_tile_size=max_tile_size)


