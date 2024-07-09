from pathlib import Path
import time
import geopandas as gpd

from geodataset.utils import GeoPackageNameConvention

from engine.embedder.contrastive.contrastive_infer import contrastive_classifier_embedder_infer
from engine.embedder.dinov2.dinov2 import infer_dinov2
from engine.embedder.siamese.siamese_infer import siamese_classifier
from engine.pipelines.pipeline_base import BaseRasterPipeline

from config.config_parsers.pipeline_parsers import PipelineClassifierIOConfig

from mains.tilerizer_mains import tilerizer_main
from mains.coco_to_geopackage_mains import coco_to_geopackage_main


class PipelineClassifier(BaseRasterPipeline):
    def __init__(self, pipeline_classifier_config: PipelineClassifierIOConfig):
        super().__init__(
            raster_path=pipeline_classifier_config.raster_path,
            aoi_geopackage_path=pipeline_classifier_config.aoi_geopackage_path,
            output_folder=pipeline_classifier_config.output_folder
        )

        self.config = pipeline_classifier_config

        self.classifier_tilerizer_output_folder = Path(self.output_folder) / 'classifier_tilerizer_output'
        self.classifier_output_folder = Path(self.output_folder) / 'classifier_output'

    @classmethod
    def from_config(cls, pipeline_classifier_config: PipelineClassifierIOConfig):
        return cls(pipeline_classifier_config)

    def run(self):
        start_time = time.time()

        embedder_tilerizer_config = self._get_tilerizer_config(
            tilerizer_config=self.config.classifier_tilerizer_config,
            output_folder=self.classifier_tilerizer_output_folder,
            labels_path=self.config.segmentations_geopackage_path,
            main_label_category_column_name=None,
            other_labels_attributes_column_names=['detector_score', 'segmenter_score']
        )

        embedder_tiles_path, coco_paths = tilerizer_main(
            config=embedder_tilerizer_config
        )

        data_roots = [coco_paths['infer'].parent, embedder_tiles_path]

        if self.config.classifier_contrastive_embedder_config:
            tiles_polygons_gdf_crs = contrastive_classifier_embedder_infer(
                backbone_name=self.config.classifier_contrastive_embedder_config.backbone_name,
                final_embedding_size=self.config.classifier_contrastive_embedder_config.final_embedding_size,
                data_roots=data_roots,
                fold=self.AOI_NAME,
                day_month_year=self.config.day_month_year,
                image_size=self.config.classifier_contrastive_embedder_config.image_size,
                mean_std_descriptor=self.config.classifier_contrastive_embedder_config.mean_std_descriptor,
                contrastive_checkpoint=self.config.classifier_contrastive_embedder_config.checkpoint_path,
                batch_size=self.config.classifier_contrastive_embedder_config.batch_size
            )

        if self.config.classifier_dinov2_embedder_config:
            dinov2_embeddings_gdf = infer_dinov2(
                data_roots=data_roots,
                image_size_center_crop_pad=self.config.classifier_dinov2_embedder_config.image_size_center_crop_pad,
                size=self.config.classifier_dinov2_embedder_config.size,
                use_cls_token=self.config.classifier_dinov2_embedder_config.use_cls_token,
            )
            dinov2_embeddings_gdf.drop('down_sampled_masks', axis=1, inplace=True)

        if self.config.classifier_contrastive_embedder_config and self.config.classifier_dinov2_embedder_config:
            tiles_polygons_gdf_crs.rename(columns={'embeddings': 'embeddings_contrastive'}, inplace=True)
            dinov2_embeddings_gdf.rename(columns={'embeddings': 'embeddings_dinov2'}, inplace=True)
            tiles_polygons_gdf_crs['tile_path'] = tiles_polygons_gdf_crs['tile_path'].astype(str)
            dinov2_embeddings_gdf['tile_path'] = dinov2_embeddings_gdf['tile_path'].astype(str)
            gdf = tiles_polygons_gdf_crs.merge(dinov2_embeddings_gdf, on='tile_path')
            gdf.rename({'geometry_x': 'geometry', 'area_x': 'area'}, axis=1, inplace=True)
            gdf.drop(['geometry_y', 'area_y'], axis=1, inplace=True)
            gdf = gpd.GeoDataFrame(gdf)
            gdf.set_geometry('geometry')
        elif self.config.classifier_contrastive_embedder_config:
            gdf = tiles_polygons_gdf_crs
        elif self.config.classifier_dinov2_embedder_config:
            gdf = dinov2_embeddings_gdf
        else:
            raise ValueError("At least one of the embedder configurations must be provided.")

        geopackage_name = GeoPackageNameConvention.create_name(
            product_name=self.raster_name,
            fold='inferembedderclassifier',
            ground_resolution=self.config.classifier_tilerizer_config.raster_resolution_config.ground_resolution,
            scale_factor=self.config.classifier_tilerizer_config.raster_resolution_config.scale_factor,
        )

        self.classifier_output_folder.mkdir(parents=True, exist_ok=True)
        output_path = self.classifier_output_folder / geopackage_name

        gdf.to_file(output_path, driver='GPKG')
        print(f"Successfully saved the embeddings and classification predictions at {output_path}.")

        end_time = time.time()
        print(f"It took {end_time - start_time} seconds to run the raster through the Embedder/Classifier pipeline.")

        return gdf, output_path
