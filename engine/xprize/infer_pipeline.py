from dataclasses import asdict
from pathlib import Path

from geodataset.utils import strip_all_extensions
from geodataset.utils.file_name_conventions import validate_and_convert_product_name

from config.config_parsers.aggregator_parsers import AggregatorIOConfig
from config.config_parsers.detector_parsers import DetectorInferIOConfig
from config.config_parsers.segmenter_parsers import SegmenterInferIOConfig
from config.config_parsers.tilerizer_parsers import TilerizerIOConfig
from config.config_parsers.xprize_parsers import XPrizeIOConfig
from mains import tilerizer_main
from mains.aggregator_main import aggregator_main
from mains.detector_mains import detector_infer_main
from mains.segmenter_main import segmenter_infer_main


class XPrizePipeline:
    def __init__(self, xprize_config: XPrizeIOConfig):
        self.config = xprize_config
        self.raster_name = validate_and_convert_product_name(strip_all_extensions(Path(self.config.raster_path)))
        self.tilerizer_output_folder = Path(self.config.output_folder) / 'tilerizer_output'
        self.detector_output_folder = Path(self.config.output_folder) / 'detector_output'
        self.aggregator_output_folder = Path(self.config.output_folder) / 'aggregator_output'
        self.segmenter_output_folder = Path(self.config.output_folder) / 'segmenter_output'

    @classmethod
    def from_config(cls, xprize_config: XPrizeIOConfig):
        return cls(xprize_config)

    def run(self):
        # Creating tiles
        tilerizer_config = self._get_tilerizer_config()
        tiles_path = tilerizer_main(config=tilerizer_config)

        # Detecting trees
        detector_config = self._get_detector_infer_config()
        detector_coco_output_path = detector_infer_main(config=detector_config)

        # Aggregating detected trees
        aggregator_config = self._get_aggregator_config(tiles_path=tiles_path,
                                                        detector_coco_output_path=detector_coco_output_path)
        aggregator_output_file = aggregator_main(config=aggregator_config)

        # Predicting tree instance segmentations
        segmenter_config = self._get_segmenter_infer_config(tiles_path=tiles_path,
                                                            aggregator_output_file=aggregator_output_file)
        segmenter_output_file = segmenter_infer_main(config=segmenter_config)

    def _get_tilerizer_config(self):
        assert len(self.config.tilerizer_config.aois) == 1, \
            "Only one AOI for the tilerizer is supported for now in the XPrize infer pipeline."

        preprocessor_config = TilerizerIOConfig(
            **self.config.tilerizer_config.as_dict(),
            raster_path=self.config.raster_path,
            output_folder=str(self.tilerizer_output_folder),
            labels_path=None,
            ignore_tiles_without_labels=True,
            main_label_category_column_name=None,
        )

        return preprocessor_config

    def _get_detector_infer_config(self):
        detector_infer_config = DetectorInferIOConfig(
            **self.config.detector_infer_config.as_dict(),
            input_tiles_root=str(self.tilerizer_output_folder),
            infer_aoi_name=list(self.config.tilerizer_config.aois.keys())[0],
            output_folder=str(self.detector_output_folder),
            coco_n_workers=self.config.coco_n_workers
        )

        return detector_infer_config

    def _get_aggregator_config(self,
                               tiles_path: Path,
                               detector_coco_output_path: Path):
        aggregator_config = AggregatorIOConfig(
            **self.config.aggregator_config.as_dict(),
            input_tiles_root=str(tiles_path),
            coco_path=str(detector_coco_output_path),
            output_folder=str(self.aggregator_output_folder),
        )

        return aggregator_config

    def _get_segmenter_infer_config(self,
                                    tiles_path: Path,
                                    aggregator_output_file: Path):
        segmenter_infer_config = SegmenterInferIOConfig(
            **self.config.sam_infer_config.as_dict(),
            input_tiles_root=str(tiles_path),
            coco_path=str(aggregator_output_file),
            output_folder=str(self.segmenter_output_folder),
        )

        return segmenter_infer_config
