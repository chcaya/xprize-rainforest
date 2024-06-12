from pathlib import Path
import time

from config.config_parsers.pipeline_parsers import PipelineXPrizeIOConfig, PipelineDetectorIOConfig, \
    PipelineSegmenterIOConfig, PipelineClassifierIOConfig
from engine.pipelines.pipeline_classifier import PipelineClassifier
from engine.pipelines.pipeline_detector import PipelineDetector
from engine.pipelines.pipeline_segmenter import PipelineSegmenter


class PipelineXPrize:
    def __init__(self, xprize_config: PipelineXPrizeIOConfig):
        self.config = xprize_config

    @classmethod
    def from_config(cls, xprize_config: PipelineXPrizeIOConfig):
        return cls(xprize_config)

    def run(self):
        start_time = time.time()
        pipeline_detector_config = self._get_pipeline_detector_config()
        boxes_geopackage_path = PipelineDetector(pipeline_detector_config=pipeline_detector_config).run()
        pipeline_segmenter_config = self._get_pipeline_segmenter_config(boxes_geopackage_path=boxes_geopackage_path)
        segmentations_geopackage_path = PipelineSegmenter(pipeline_segmenter_config=pipeline_segmenter_config).run()
        pipeline_classifier_config = self._get_pipeline_classifier_config(segmentations_geopackage_path=segmentations_geopackage_path)
        classifier_geopackage_path = PipelineClassifier(pipeline_classifier_config=pipeline_classifier_config).run()

        end_time = time.time()
        print(f"\nThe final geopackage is saved at {classifier_geopackage_path}.")
        print(f"It took {end_time - start_time} seconds to run the raster through the whole XPrize pipeline.")

    def _get_pipeline_detector_config(self):
        pipeline_detector_config = PipelineDetectorIOConfig(
            **self.config.pipeline_detector_config.as_dict(),
            raster_path=self.config.raster_path,
            output_folder=str(self.config.output_folder),
            coco_n_workers=self.config.coco_n_workers,
        )

        return pipeline_detector_config

    def _get_pipeline_segmenter_config(self, boxes_geopackage_path: Path):
        pipeline_segmenter_config = PipelineSegmenterIOConfig(
            **self.config.pipeline_segmenter_config.as_dict(),
            raster_path=self.config.raster_path,
            output_folder=str(self.config.output_folder),
            boxes_geopackage_path=str(boxes_geopackage_path)
        )

        return pipeline_segmenter_config

    def _get_pipeline_classifier_config(self, segmentations_geopackage_path: Path):
        pipeline_classifier_config = PipelineClassifierIOConfig(
            **self.config.pipeline_classifier_config.as_dict(),
            raster_path=self.config.raster_path,
            output_folder=str(self.config.output_folder),
            segmentations_geopackage_path=str(segmentations_geopackage_path)
        )

        return pipeline_classifier_config

