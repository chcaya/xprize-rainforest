import json
from datetime import date
from pathlib import Path
from shapely import box

from geodataset.dataset import UnlabeledRasterDataset
from geodataset.utils import generate_label_coco, strip_all_extensions
from geodataset.utils.file_name_conventions import CocoNameConvention, validate_and_convert_product_name
from geodataset.aggregator import DetectionAggregator

from config.config_parser.config_parsers import XPrizeConfig, PreprocessorConfig, DetectorInferConfig
from engine.detector.detector_pipelines import DetectorInferencePipeline
from engine.detector.utils import collate_fn_images
from mains import preprocessor_main


class XPrizePipeline:
    def __init__(self, xprize_config: XPrizeConfig):
        self.config = xprize_config
        self.raster_name = validate_and_convert_product_name(strip_all_extensions(Path(self.config.raster_path)))
        self.preprocessor_output_folder = Path(self.config.output_folder) / 'preprocessor_output'
        self.detector_output_folder = Path(self.config.output_folder) / 'detector_output'
        self.aggregator_output_folder = Path(self.config.output_folder) / 'aggregator_output'

        self.preprocessor_output_folder.mkdir(parents=True, exist_ok=True)
        self.detector_output_folder.mkdir(parents=True, exist_ok=True)
        self.aggregator_output_folder.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, xprize_config: XPrizeConfig):
        return cls(xprize_config)

    def run(self):
        # Creating tiles
        preprocessor_config = self._get_preprocessor_config()
        preprocessor_main(preprocessor_config)

        # Detecting trees
        print('Detecting trees...')
        infer_ds = UnlabeledRasterDataset(root_path=Path(self.preprocessor_output_folder),
                                          fold="infer",
                                          transform=None)  # No augmentation for inference
        detector_config = self._get_detector_infer_config()
        inferer = DetectorInferencePipeline.from_config(detector_config)

        detector_result = inferer.infer(infer_ds=infer_ds, collate_fn=collate_fn_images)
        detector_result = [{k: v.cpu().numpy() for k, v in x.items()} for x in detector_result]
        for x in detector_result:
            x['boxes'] = [box(*b) for b in x['boxes']]
            x['scores'] = x['scores'].tolist()
        boxes = [x['boxes'] for x in detector_result]
        scores = [x['scores'] for x in detector_result]

        detector_output_name = CocoNameConvention.create_name(product_name=self.raster_name,
                                                              fold="infer",
                                                              scale_factor=self.config.scale_factor,
                                                              ground_resolution=self.config.ground_resolution)
        print('Saving trees to disk into a COCO file...')
        self._generate_detector_inference_coco(tiles_paths=list(infer_ds.tile_paths),  # Important: don't shuffle the infer_ds or the dataloader,
                                                                                       # or it will mess up the tiles_paths order with the detected boxes.
                                               boxes=boxes,
                                               scores=scores,
                                               output_path=self.detector_output_folder / f'{detector_output_name}')

        # Aggregating detected trees
        print('Aggregating (de-duplicating) detected trees...')
        DetectionAggregator.from_boxes(geojson_output_path=self.aggregator_output_folder / f'test_output_08_scores04.geojson',  # TODO change output name
                                       boxes=boxes,
                                       scores=scores,
                                       tiles_paths=list(infer_ds.tile_paths),
                                       min_score_threshold=0.4,         # TODO add these parameters in the config file
                                       intersect_remove_ratio=0.8)

    def _generate_detector_inference_coco(self, tiles_paths: list, boxes: list, scores: list, output_path: Path):
        images_cocos = []
        detections_cocos = []
        for tile_id, (tile_path, tile_boxes, tile_boxes_scores) in enumerate(zip(tiles_paths, boxes, scores)):
            images_cocos.append({
                "id": tile_id,
                "width": self.config.tile_size,
                "height": self.config.tile_size,
                "file_name": str(tile_path.name),
            })

            for i in range(len(tile_boxes)):
                detections_cocos.append(generate_label_coco(
                    polygon=tile_boxes[i],
                    tile_height=self.config.tile_size,
                    tile_width=self.config.tile_size,
                    tile_id=tile_id,
                    use_rle_for_labels=True,
                    category_id=None,
                    other_attributes_dict={'score': float(tile_boxes_scores[i])}
                ))

        # Save the COCO dataset to a JSON file
        with output_path.open('w') as f:
            json.dump({
                "info": {
                    "description": f"Inference for the product '{Path(self.config.raster_path).name}'"
                                   f" with the model architecture '{self.config.detector_architecture}',"
                                   f" backbone '{self.config.detector_rcnn_backbone_model_resnet_name}'"
                                   f" and the checkpoint '{self.config.detector_checkpoint_state_dict_path}'."
                                   f" The scale_factor is '{self.config.scale_factor}'"
                                   f" and ground_resolution is '{self.config.ground_resolution}'.",
                    "dataset_name": str(Path(self.config.raster_path).name),
                    "version": "1.0",
                    "year": str(date.today().year),
                    "date_created": str(date.today())
                },
                "licenses": [
                    # add license?
                ],
                "images": images_cocos,
                "annotations": detections_cocos,
                "categories": None}, f, ensure_ascii=False, indent=2)

    def _get_preprocessor_config(self):
        preprocessor_config = PreprocessorConfig(
            raster_path=self.config.raster_path,
            labels_path=None,
            output_folder=str(self.preprocessor_output_folder),
            tile_size=self.config.tile_size,
            tile_overlap=self.config.tile_overlap,
            scale_factor=self.config.scale_factor,
            ground_resolution=self.config.ground_resolution,
            aoi_config=self.config.aoi_config,
            aoi_type=self.config.aoi_type,
            aois=self.config.aois,
            ignore_black_white_alpha_tiles_threshold=self.config.ignore_black_white_alpha_tiles_threshold,
            ignore_tiles_without_labels=False,
            main_label_category_column_name=None
        )

        return preprocessor_config

    def _get_detector_infer_config(self):
        detector_infer_config = DetectorInferConfig(
            data_root_path=str(self.preprocessor_output_folder),
            architecture=self.config.detector_architecture,
            rcnn_backbone_model_resnet_name=self.config.detector_rcnn_backbone_model_resnet_name,
            batch_size=self.config.detector_batch_size,
            checkpoint_state_dict_path=self.config.detector_checkpoint_state_dict_path,
            box_predictions_per_image=self.config.detector_box_predictions_per_image
        )

        return detector_infer_config
