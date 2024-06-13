from pathlib import Path
import time

from geodataset.utils.file_name_conventions import CocoNameConvention

from config.config_parsers.segmenter_parsers import SegmenterInferIOConfig
from config.config_parsers.pipeline_parsers import PipelineSegmenterIOConfig
from engine.pipelines.pipeline_base import BaseRasterPipeline
from mains.tilerizer_mains import tilerizer_main
from mains.aggregator_mains import aggregator_main_with_polygons_input
from mains.coco_to_geopackage_mains import coco_to_geopackage_main
from mains.segmenter_mains import segmenter_infer_main


class PipelineSegmenter(BaseRasterPipeline):
    def __init__(self, pipeline_segmenter_config: PipelineSegmenterIOConfig):
        super().__init__(
            raster_path=pipeline_segmenter_config.raster_path,
            output_folder=pipeline_segmenter_config.output_folder
        )

        self.config = pipeline_segmenter_config
        self.scores_weights_config = self.config.segmenter_aggregator_config.scores_weights

        self.segmenter_tilerizer_output_folder = Path(self.output_folder) / 'segmenter_tilerizer_output'
        self.segmenter_output_folder = Path(self.output_folder) / 'segmenter_output'
        self.segmenter_aggregator_output_folder = Path(self.output_folder) / 'segmenter_aggregator_output'

    @classmethod
    def from_config(cls, pipeline_segmenter_config: PipelineSegmenterIOConfig):
        return cls(pipeline_segmenter_config)

    def run(self):
        start_time = time.time()

        segmenter_tilerizer_config = self._get_tilerizer_config(
            tilerizer_config=self.config.segmenter_tilerizer_config,
            output_folder=self.segmenter_tilerizer_output_folder,
            labels_path=self.config.boxes_geopackage_path,
            main_label_category_column_name=None,
            other_labels_attributes_column_names=['detector_score'] if self.scores_weights_config and 'detector_score' in self.scores_weights_config else None,
        )
        segmenter_tiles_path, segmenter_coco_paths = tilerizer_main(
            config=segmenter_tilerizer_config
        )
        segmenter_coco_path = segmenter_coco_paths[list(self.config.segmenter_tilerizer_config.aois.keys())[0]]

        # Predicting tree instance segmentations
        segmenter_config = self._get_segmenter_infer_config(
            tiles_path=segmenter_tiles_path,
            coco_path=segmenter_coco_path
        )
        segmenter_output = segmenter_infer_main(
            config=segmenter_config
        )
        segmenter_scale_factor = segmenter_tilerizer_config.raster_resolution_config.scale_factor
        segmenter_ground_resolution = segmenter_tilerizer_config.raster_resolution_config.ground_resolution

        if self.config.save_segmenter_intermediate_output:
            segmenter_tiles_paths, segmenter_masks, segmenter_masks_scores, segmenter_boxes_scores, segmenter_output_file = segmenter_output
        else:
            segmenter_tiles_paths, segmenter_masks, segmenter_masks_scores, segmenter_boxes_scores = segmenter_output

        # Aggregating the segmented trees
        segmenter_aggregator_output_file = CocoNameConvention.create_name(
            product_name=self.raster_name,
            fold='infersegmenteraggregator',
            scale_factor=segmenter_scale_factor,
            ground_resolution=segmenter_ground_resolution
        )
        segmenter_aggregator_output_path = self.segmenter_aggregator_output_folder / segmenter_aggregator_output_file

        polygons_scores = {'segmenter_score': segmenter_masks_scores}
        polygons_scores_weights = {'segmenter_score': self.scores_weights_config['segmenter_score'] if self.scores_weights_config and 'segmenter_score' in self.scores_weights_config else 1.0}
        if self.scores_weights_config and 'detector_score' in self.scores_weights_config:
            polygons_scores['detector_score'] = segmenter_boxes_scores
            polygons_scores_weights['detector_score'] = self.scores_weights_config['detector_score'] if self.scores_weights_config and 'detector_score' in self.scores_weights_config else 1.0

        aggregator_main_with_polygons_input(
            config=self.config.segmenter_aggregator_config,
            tiles_paths=segmenter_tiles_paths,
            polygons=segmenter_masks,
            polygons_scores=polygons_scores,
            polygons_scores_weights=polygons_scores_weights,
            output_path=segmenter_aggregator_output_path
        )

        # Converting aggregated trees masks from coco to geopackage
        coco_to_geopackage_config = self._get_coco_to_geopackage_config(
            input_tiles_root=segmenter_tiles_path,
            coco_path=segmenter_aggregator_output_path,
            output_folder=self.segmenter_aggregator_output_folder
        )
        tree_segments_gdf, segmenter_aggregator_geopackage_path = coco_to_geopackage_main(
            config=coco_to_geopackage_config
        )

        end_time = time.time()
        print(f"It took {end_time - start_time} seconds to run the raster through the Segmenter pipeline.")

        return segmenter_aggregator_geopackage_path

    def _get_segmenter_infer_config(self,
                                    tiles_path: Path,
                                    coco_path: Path):

        if self.config.save_segmenter_intermediate_output:
            output_folder = str(self.segmenter_output_folder)
        else:
            output_folder = None

        segmenter_infer_config = SegmenterInferIOConfig(
            **self.config.segmenter_infer_config.as_dict(),
            input_tiles_root=str(tiles_path),
            coco_path=str(coco_path),
            output_folder=output_folder,
        )

        return segmenter_infer_config
