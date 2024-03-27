from dataclasses import asdict
from pathlib import Path
from shapely import box

from geodataset.dataset import UnlabeledRasterDataset, BoxesDataset
from geodataset.utils import strip_all_extensions
from geodataset.utils.file_name_conventions import CocoNameConvention, validate_and_convert_product_name
from geodataset.aggregator import DetectionAggregator

from config.config_parsers.aggregator_parsers import AggregatorCLIConfig
from config.config_parsers.detector_parsers import DetectorInferCLIConfig
from config.config_parsers.segmenter_parsers import SegmenterInferCLIConfig
from config.config_parsers.tilerizer_parsers import TilerizerCLIConfig
from config.config_parsers.xprize_parsers import XPrizeConfig
from engine.detector.detector_pipelines import DetectorInferencePipeline
from engine.detector.utils import collate_fn_images, generate_detector_inference_coco, detector_result_to_lists
from engine.segmenter.sam import SamPredictorWrapper
from mains import tilerizer_main
from mains.aggregator_main import aggregator_main
from mains.detector_mains import detector_infer_main
from mains.segmenter_main import segmenter_infer_main


class XPrizePipeline:
    def __init__(self, xprize_config: XPrizeConfig):
        self.config = xprize_config
        self.raster_name = validate_and_convert_product_name(strip_all_extensions(Path(self.config.raster_path)))
        self.tilerizer_output_folder = Path(self.config.output_folder) / 'tilerizer_output'
        self.detector_output_folder = Path(self.config.output_folder) / 'detector_output'
        self.aggregator_output_folder = Path(self.config.output_folder) / 'aggregator_output'
        self.sam_output_folder = Path(self.config.output_folder) / 'sam_output'

    @classmethod
    def from_config(cls, xprize_config: XPrizeConfig):
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
        segmenter_config = self._get_segmenter_infer_config(aggregator_output_file=aggregator_output_file)
        segmenter_output_file = segmenter_infer_main(config=segmenter_config)

    def _get_tilerizer_config(self):
        preprocessor_config = TilerizerCLIConfig(
            **self.config.tilerizer_config.as_dict(),
            raster_path=self.config.raster_path,
            output_folder=str(self.tilerizer_output_folder),
            labels_path=None,
            ignore_tiles_without_labels=True,
            main_label_category_column_name=None,
        )

        return preprocessor_config

    def _get_detector_infer_config(self):
        detector_infer_config = DetectorInferCLIConfig(
            **self.config.detector_infer_config.as_dict(),
            input_tiles_root=str(self.tilerizer_output_folder),
            infer_aoi_name='infer',
            output_folder=str(self.detector_output_folder),
            coco_n_workers=self.config.coco_n_workers
        )

        return detector_infer_config

    def _get_aggregator_config(self,
                               tiles_path: Path,
                               detector_coco_output_path: Path):
        aggregator_config = AggregatorCLIConfig(
            **self.config.aggregator_config.as_dict(),
            input_tiles_root=str(tiles_path),
            coco_path=str(detector_coco_output_path),
            output_folder=str(self.aggregator_output_folder),
        )

        return aggregator_config

    def _get_segmenter_infer_config(self, aggregator_output_file: Path):
        segmenter_infer_config = SegmenterInferCLIConfig(
            **self.config.sam_infer_config.as_dict(),
            raster_path=self.config.raster_path,
            boxes_path=str(aggregator_output_file),
            output_folder=str(self.sam_output_folder),
        )

        return segmenter_infer_config
