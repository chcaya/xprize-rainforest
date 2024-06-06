from dataclasses import asdict
from pathlib import Path
import time

from geodataset.dataset.polygon_dataset import SiameseValidationDataset
from geodataset.utils import strip_all_extensions
from geodataset.utils.file_name_conventions import validate_and_convert_product_name, CocoNameConvention

from config.config_parsers.coco_to_geopackage_parsers import CocoToGeopackageIOConfig
from config.config_parsers.detector_parsers import DetectorInferIOConfig
from config.config_parsers.segmenter_parsers import SegmenterInferIOConfig
from config.config_parsers.tilerizer_parsers import TilerizerIOConfig, TilerizerConfig
from config.config_parsers.xprize_parsers import XPrizeIOConfig
from engine.embedder.siamese.siamese_infer import siamese_classifier
from mains import tilerizer_main
from mains.aggregator_main import aggregator_main_with_polygons_input
from mains.coco_to_geopackage_main import coco_to_geopackage_main
from mains.detector_mains import detector_infer_main
from mains.embedder_main import siamese_infer_main
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
        self.classifier_tilerizer_output_folder = Path(self.config.output_folder) / 'classifier_tilerizer_output'
        self.classifier_output_folder = Path(self.config.output_folder) / 'classifier_output'

    @classmethod
    def from_config(cls, xprize_config: XPrizeIOConfig):
        return cls(xprize_config)

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
        aggregator_main_with_polygons_input(
            config=self.config.detector_aggregator_config,
            tiles_paths=detector_tiles_paths,
            polygons=detector_polygons,
            polygons_scores={'detector_score': detector_polygons_scores},
            polygons_scores_weights={'detector_score': 1.0},
            output_path=detector_aggregator_output_path
        )

        # Converting aggregated trees from coco to geojson
        coco_to_geopackage_config = self._get_coco_to_geopackage_config(
            input_tiles_root=detector_tiles_path,
            coco_path=detector_aggregator_output_path,
            output_folder=self.detector_aggregator_output_folder
        )
        _, detector_aggregator_geojson_path = coco_to_geopackage_main(
            config=coco_to_geopackage_config
        )

        if asdict(self.config.detector_tilerizer_config) == asdict(self.config.segmenter_tilerizer_config):
            # The tilerizer configs are the same for the detector and the segmenter, so no need to re-tilerize and re-run the aggregator.
            print("Skipping the segmenter tilerizer as the detector and segmenter tilerizer configs are the same.")

            # Predicting tree instance segmentations
            segmenter_config = self._get_segmenter_infer_config(
                tiles_path=detector_tiles_path,
                coco_path=detector_aggregator_output_path
            )
            segmenter_output = segmenter_infer_main(
                config=segmenter_config
            )

            segmenter_tiles_path = detector_tiles_path
            segmenter_scale_factor = detector_tilerizer_config.raster_resolution_config.scale_factor
            segmenter_ground_resolution = detector_tilerizer_config.raster_resolution_config.ground_resolution

        else:
            # Creating tiles for the segmenter as the tilerizer configs are different for the detector and segmenter
            # TODO add a boolean parameter to Tilerizer to avoid duplicating boxes/segments (for each box it should find out what is the appropriate tile based on tiles centroids and borders intersections)
            segmenter_tilerizer_config = self._get_tilerizer_config(
                tilerizer_config=self.config.segmenter_tilerizer_config,
                output_folder=self.segmenter_tilerizer_output_folder,
                labels_path=detector_aggregator_geojson_path,
                main_label_category_column_name=None,
                other_labels_attributes_column_names=['detector_score']
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

        aggregator_main_with_polygons_input(
            config=self.config.segmenter_aggregator_config,
            tiles_paths=segmenter_tiles_paths,
            polygons=segmenter_masks,
            polygons_scores={'detector_score': segmenter_boxes_scores,
                             'segmenter_score': segmenter_masks_scores},
            polygons_scores_weights={'detector_score': 3.0, 'segmenter_score': 1.0},        # TODO THIS DOESNT WORK AS EXPECTED (3 * 1 * score_detector * score_segmenter scales all the results the same (* 3))
            output_path=segmenter_aggregator_output_path
        )

        # Converting aggregated trees masks from coco to geojson
        coco_to_geopackage_config = self._get_coco_to_geopackage_config(
            input_tiles_root=segmenter_tiles_path,
            coco_path=segmenter_aggregator_output_path,
            output_folder=self.segmenter_aggregator_output_folder
        )
        tree_segments_gdf, segmenter_aggregator_geojson_path = coco_to_geopackage_main(
            config=coco_to_geopackage_config
        )

        # Tilerizing the final dataset for the siamese classifier, with one tree per tile
        embedder_tilerizer_config = self._get_tilerizer_config(
            tilerizer_config=self.config.classifier_tilerizer_config,
            output_folder=self.classifier_tilerizer_output_folder,
            labels_path=segmenter_aggregator_geojson_path,
            main_label_category_column_name=None,
            other_labels_attributes_column_names=['detector_score', 'segmenter_score']
        )

        embedder_tiles_path, coco_paths = tilerizer_main(
            config=embedder_tilerizer_config
        )

        # Getting embeddings of each objects
        embeddings_df = siamese_infer_main(
            config=self.config.classifier_embedder_config,
            siamese_dataset=SiameseValidationDataset(
                fold='infer',
                root_path=[coco_paths['infer'].parent, embedder_tiles_path]
            )
        )

        classifier_coco_path = siamese_classifier(
            data_roots=[coco_paths['infer'].parent, embedder_tiles_path],
            fold='infer',
            siamese_checkpoint=self.config.classifier_embedder_config.checkpoint_path,
            scaler_checkpoint=self.config.classifier_infer_config.scaler_checkpoint_path,
            svc_checkpoint=self.config.classifier_infer_config.classifier_checkpoint_path,
            batch_size=self.config.classifier_embedder_config.batch_size,
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
        tree_segments_classified_gdf, classifier_geojson_path = coco_to_geopackage_main(
            config=coco_to_geopackage_config
        )

        end_time = time.time()
        print(f"It took {end_time - start_time} seconds to run the raster through the whole pipeline.")

    def _get_tilerizer_config(self,
                              tilerizer_config: TilerizerConfig,
                              output_folder: Path,
                              labels_path: Path or None,
                              main_label_category_column_name: str or None,
                              other_labels_attributes_column_names: list or None):
        assert len(tilerizer_config.aois) == 1, \
            "Only one AOI for the tilerizer is supported for now in the XPrize infer pipeline."

        preprocessor_config = TilerizerIOConfig(
            **tilerizer_config.as_dict(),
            raster_path=self.config.raster_path,
            output_folder=str(output_folder),
            labels_path=labels_path,
            ignore_tiles_without_labels=True,
            main_label_category_column_name=main_label_category_column_name,
            other_labels_attributes_column_names=other_labels_attributes_column_names
        )

        return preprocessor_config

    def _get_detector_infer_config(self):
        if self.config.save_detector_intermediate_output:
            output_folder = str(self.segmenter_output_folder)
        else:
            output_folder = None

        detector_infer_config = DetectorInferIOConfig(
            **self.config.detector_infer_config.as_dict(),
            input_tiles_root=str(self.detector_tilerizer_output_folder),
            infer_aoi_name=list(self.config.detector_tilerizer_config.aois.keys())[0],
            output_folder=output_folder,
            coco_n_workers=self.config.coco_n_workers
        )

        return detector_infer_config

    @staticmethod
    def _get_coco_to_geopackage_config(input_tiles_root: Path,
                                       coco_path: Path,
                                       output_folder: Path):
        coco_to_geopackage_config = CocoToGeopackageIOConfig(
            input_tiles_root=str(input_tiles_root),
            coco_path=str(coco_path),
            output_folder=str(output_folder),
        )

        return coco_to_geopackage_config

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
