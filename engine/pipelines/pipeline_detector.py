from pathlib import Path
import time


from geodataset.utils.file_name_conventions import CocoNameConvention

from engine.pipelines.pipeline_base import BaseRasterPipeline

from config.config_parsers.detector_parsers import DetectorInferIOConfig
from config.config_parsers.pipeline_parsers import PipelineDetectorIOConfig

from mains.tilerizer_mains import tilerizer_main
from mains.aggregator_mains import aggregator_main_with_polygons_input
from mains.coco_to_geopackage_mains import coco_to_geopackage_main
from mains.detector_mains import detector_infer_main


class PipelineDetector(BaseRasterPipeline):
    def __init__(self, pipeline_detector_config: PipelineDetectorIOConfig):
        super().__init__(
            raster_path=pipeline_detector_config.raster_path,
            aoi_geopackage_path=pipeline_detector_config.aoi_geopackage_path,
            output_folder=pipeline_detector_config.output_folder
        )

        self.config = pipeline_detector_config
        self.scores_weights_config = self.config.detector_aggregator_config.scores_weights

        self.detector_tilerizer_output_folder = Path(self.output_folder) / 'detector_tilerizer_output'
        self.detector_output_folder = Path(self.output_folder) / 'detector_output'
        self.detector_aggregator_output_folder = Path(self.output_folder) / 'detector_aggregator_output'

    @classmethod
    def from_config(cls, pipeline_detector_config: PipelineDetectorIOConfig):
        return cls(pipeline_detector_config)

    def run(self):
        start_time = time.time()

        # Creating tiles for the detector
        detector_tilerizer_config = self._get_tilerizer_config(
            tilerizer_config=self.config.detector_tilerizer_config,
            output_folder=self.detector_tilerizer_output_folder,
            labels_path=None,
            main_label_category_column_name=None,
            other_labels_attributes_column_names=None
        )
        detector_tiles_path, _ = tilerizer_main(
            config=detector_tilerizer_config
        )

        # Detecting trees
        detector_config = self._get_detector_infer_config()
        detector_output = detector_infer_main(
            config=detector_config
        )

        if self.config.save_detector_intermediate_output:
            detector_tiles_paths, detector_polygons, detector_polygons_scores, _ = detector_output
        else:
            detector_tiles_paths, detector_polygons, detector_polygons_scores = detector_output

        # Aggregating detected trees
        detector_aggregator_output_file = CocoNameConvention.create_name(
            product_name=self.raster_name,
            fold='inferdetectoraggregator',
            scale_factor=self.config.detector_tilerizer_config.raster_resolution_config.scale_factor,
            ground_resolution=self.config.detector_tilerizer_config.raster_resolution_config.ground_resolution
        )
        detector_aggregator_output_path = self.detector_aggregator_output_folder / detector_aggregator_output_file

        polygons_scores = {'detector_score': detector_polygons_scores}
        polygons_scores_weights = {'detector_score': self.scores_weights_config['detector_score'] if self.scores_weights_config and 'detector_score' in self.scores_weights_config else 1.0}

        aggregator_main_with_polygons_input(
            config=self.config.detector_aggregator_config,
            tiles_paths=detector_tiles_paths,
            polygons=detector_polygons,
            polygons_scores=polygons_scores,
            polygons_scores_weights=polygons_scores_weights,
            output_path=detector_aggregator_output_path
        )

        # Converting aggregated trees from coco to geopackage
        coco_to_geopackage_config = self._get_coco_to_geopackage_config(
            input_tiles_root=detector_tiles_path,
            coco_path=detector_aggregator_output_path,
            output_folder=self.detector_aggregator_output_folder
        )
        _, detector_aggregator_geopackage_path = coco_to_geopackage_main(
            config=coco_to_geopackage_config
        )

        end_time = time.time()
        print(f"It took {end_time - start_time} seconds to run the raster through the Detector pipeline.")

        return detector_aggregator_geopackage_path

    def _get_detector_infer_config(self):
        if self.config.save_detector_intermediate_output:
            output_folder = str(self.detector_output_folder)
        else:
            output_folder = None

        detector_infer_config = DetectorInferIOConfig(
            **self.config.detector_infer_config.as_dict(),
            input_tiles_root=str(self.detector_tilerizer_output_folder),
            infer_aoi_name=self.AOI_NAME,
            output_folder=output_folder,
            coco_n_workers=self.config.coco_n_workers
        )

        return detector_infer_config
