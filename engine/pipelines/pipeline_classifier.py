from pathlib import Path
import time

from engine.embedder.siamese.siamese_infer import siamese_classifier
from engine.pipelines.pipeline_base import BaseRasterPipeline

from config.config_parsers.pipeline_parsers import PipelineClassifierIOConfig

from mains.tilerizer_mains import tilerizer_main
from mains.coco_to_geopackage_mains import coco_to_geopackage_main


class PipelineClassifier(BaseRasterPipeline):
    def __init__(self, pipeline_classifier_config: PipelineClassifierIOConfig):
        super().__init__(
            raster_path=pipeline_classifier_config.raster_path,
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

        # Getting embeddings of each objects
        classifier_coco_path = siamese_classifier(
            data_roots=[coco_paths['infer'].parent, embedder_tiles_path],
            fold='infer',
            siamese_checkpoint=self.config.classifier_embedder_config.checkpoint_path,
            scaler_checkpoint=self.config.classifier_infer_config.scaler_checkpoint_path,
            svc_checkpoint=self.config.classifier_infer_config.classifier_checkpoint_path,
            batch_size=self.config.classifier_embedder_config.batch_size,
            backbone_model_resnet_name=self.config.classifier_embedder_config.architecture_config.backbone_model_resnet_name,
            final_embedding_size=self.config.classifier_embedder_config.architecture_config.final_embedding_size,
            product_name=self.raster_name,
            ground_resolution=self.config.classifier_tilerizer_config.raster_resolution_config.ground_resolution,
            scale_factor=self.config.classifier_tilerizer_config.raster_resolution_config.scale_factor,
            output_path=self.classifier_output_folder
        )
        coco_to_geopackage_config = self._get_coco_to_geopackage_config(
            input_tiles_root=embedder_tiles_path,
            coco_path=classifier_coco_path,
            output_folder=self.classifier_output_folder
        )
        tree_segments_classified_gdf, classifier_geopackage_path = coco_to_geopackage_main(
            config=coco_to_geopackage_config
        )

        end_time = time.time()
        print(f"It took {end_time - start_time} seconds to run the raster through the Classifier pipeline.")

        return classifier_geopackage_path
