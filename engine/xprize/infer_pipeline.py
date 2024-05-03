from dataclasses import asdict
from pathlib import Path

from geodataset.utils import strip_all_extensions
from geodataset.utils.file_name_conventions import validate_and_convert_product_name

from config.config_parsers.aggregator_parsers import AggregatorIOConfig, AggregatorConfig
from config.config_parsers.coco_to_geojson_parsers import CocoToGeojsonIOConfig
from config.config_parsers.detector_parsers import DetectorInferIOConfig
from config.config_parsers.segmenter_parsers import SegmenterInferIOConfig
from config.config_parsers.tilerizer_parsers import TilerizerIOConfig, TilerizerConfig
from config.config_parsers.xprize_parsers import XPrizeIOConfig
from engine.embedder.siamese.siamese_infer import siamese_classifier
from mains import tilerizer_main
from mains.aggregator_main import aggregator_main
from mains.coco_to_geojson_main import coco_to_geojson_main
from mains.detector_mains import detector_infer_main
from mains.segmenter_main import segmenter_infer_main


class XPrizePipeline:
    def __init__(self, xprize_config: XPrizeIOConfig):
        self.config = xprize_config
        self.raster_name = validate_and_convert_product_name(strip_all_extensions(Path(self.config.raster_path)))
        self.detector_tilerizer_output_folder = Path(self.config.output_folder) / 'detector_tilerizer_output'
        self.detector_output_folder = Path(self.config.output_folder) / 'detector_output'
        self.detector_aggregator_output_folder = Path(self.config.output_folder) / 'detector_aggregator_output'
        self.segmenter_tilerizer_output_folder = Path(self.config.output_folder) / 'segmenter_tilerizer_output'
        self.segmenter_output_folder = Path(self.config.output_folder) / 'segmenter_output'
        self.segmenter_aggregator_output_folder = Path(self.config.output_folder) / 'segmenter_aggregator_output'

    @classmethod
    def from_config(cls, xprize_config: XPrizeIOConfig):
        return cls(xprize_config)

    def run(self):
        # Creating tiles for the detector
        detector_tilerizer_config = self._get_tilerizer_config(
            tilerizer_config=self.config.detector_tilerizer_config,
            output_folder=self.detector_tilerizer_output_folder,
            labels_path=None,
            main_label_category_column_name=None
        )
        detector_tiles_path, _ = tilerizer_main(
            config=detector_tilerizer_config
        )

        # Detecting trees
        detector_config = self._get_detector_infer_config()
        detector_coco_output_path = detector_infer_main(
            config=detector_config
        )

        # Aggregating detected trees
        detector_aggregator_config = self._get_aggregator_config(
            aggregator_config=self.config.detector_aggregator_config,
            tiles_path=detector_tiles_path,
            coco_path=detector_coco_output_path,
            output_path=self.detector_aggregator_output_folder
        )
        detector_aggregator_output_file = aggregator_main(
            config=detector_aggregator_config
        )

        # Converting aggregated trees from coco to geojson
        coco_to_geojson_config = self._get_coco_to_geojson_config(
            input_tiles_root=detector_tiles_path,
            coco_path=detector_aggregator_output_file,
            output_folder=self.detector_aggregator_output_folder
        )
        _, detector_aggregator_geojson_path = coco_to_geojson_main(
            config=coco_to_geojson_config
        )

        if asdict(self.config.detector_tilerizer_config) == asdict(self.config.segmenter_tilerizer_config):
            # The tilerizer configs are the same for the detector and the segmenter, so no need to re-tilerize and re-run the aggregator.
            print("Skipping the segmenter tilerizer as the detector and segmenter tilerizer configs are the same.")

            # Predicting tree instance segmentations
            segmenter_config = self._get_segmenter_infer_config(
                tiles_path=detector_tiles_path,
                coco_path=detector_aggregator_output_file
            )
            segmenter_output_file = segmenter_infer_main(
                config=segmenter_config
            )

            segmenter_tiles_path = detector_tiles_path

        else:
            # Creating tiles for the segmenter as the tilerizer configs are different for the detector and segmenter
            # TODO add a boolean parameter to Tilerizer to avoid duplicating boxes/segments (for each box it should find out what is the appropriate tile based on tiles centroids and borders intersections)
            segmenter_tilerizer_config = self._get_tilerizer_config(
                tilerizer_config=self.config.segmenter_tilerizer_config,
                output_folder=self.segmenter_tilerizer_output_folder,
                labels_path=detector_aggregator_geojson_path,
                main_label_category_column_name=None  # TODO maybe change?
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
            segmenter_output_file = segmenter_infer_main(
                config=segmenter_config
            )

        # Aggregating trees masks       # TODO change the Aggregator logic for segmentations, as it should merge the intersections etc
        segmenter_aggregator_config = self._get_aggregator_config(
            aggregator_config=self.config.segmenter_aggregator_config,
            tiles_path=segmenter_tiles_path,
            coco_path=segmenter_output_file,
            output_path=self.segmenter_aggregator_output_folder
        )
        segmenter_final_output_file = aggregator_main(
            config=segmenter_aggregator_config
        )

        # Converting aggregated trees masks from coco to geojson
        coco_to_geojson_config = self._get_coco_to_geojson_config(
            input_tiles_root=segmenter_tiles_path,
            coco_path=segmenter_final_output_file,
            output_folder=self.segmenter_aggregator_output_folder
        )
        tree_segments_gdf, segmenter_aggregator_geojson_path = coco_to_geojson_main(
            config=coco_to_geojson_config
        )

        # TODO tilerize with PolygonTilerizer

        siamese_classifier(
            data_roots=[segmenter_final_output_file.parent, segmenter_tiles_path],
            fold='' # TODO
        )

    def _get_tilerizer_config(self,
                              tilerizer_config: TilerizerConfig,
                              output_folder: Path,
                              labels_path: Path or None,
                              main_label_category_column_name: str or None):
        assert len(tilerizer_config.aois) == 1, \
            "Only one AOI for the tilerizer is supported for now in the XPrize infer pipeline."

        preprocessor_config = TilerizerIOConfig(
            **tilerizer_config.as_dict(),
            raster_path=self.config.raster_path,
            output_folder=str(output_folder),
            labels_path=labels_path,
            ignore_tiles_without_labels=True,
            main_label_category_column_name=main_label_category_column_name,
        )

        return preprocessor_config

    def _get_detector_infer_config(self):
        detector_infer_config = DetectorInferIOConfig(
            **self.config.detector_infer_config.as_dict(),
            input_tiles_root=str(self.detector_tilerizer_output_folder),
            infer_aoi_name=list(self.config.detector_tilerizer_config.aois.keys())[0],
            output_folder=str(self.detector_output_folder),
            coco_n_workers=self.config.coco_n_workers
        )

        return detector_infer_config

    @staticmethod
    def _get_aggregator_config(aggregator_config: AggregatorConfig,
                               tiles_path: Path,
                               coco_path: Path,
                               output_path: Path):
        aggregator_config = AggregatorIOConfig(
            **aggregator_config.as_dict(),
            input_tiles_root=str(tiles_path),
            coco_path=str(coco_path),
            output_folder=str(output_path),
        )

        return aggregator_config

    @staticmethod
    def _get_coco_to_geojson_config(input_tiles_root: Path,
                                    coco_path: Path,
                                    output_folder: Path):
        coco_to_geojson_config = CocoToGeojsonIOConfig(
            input_tiles_root=str(input_tiles_root),
            coco_path=str(coco_path),
            output_folder=str(output_folder),
        )

        return coco_to_geojson_config

    def _get_segmenter_infer_config(self,
                                    tiles_path: Path,
                                    coco_path: Path):
        segmenter_infer_config = SegmenterInferIOConfig(
            **self.config.segmenter_infer_config.as_dict(),
            input_tiles_root=str(tiles_path),
            coco_path=str(coco_path),
            output_folder=str(self.segmenter_output_folder),
        )

        return segmenter_infer_config
