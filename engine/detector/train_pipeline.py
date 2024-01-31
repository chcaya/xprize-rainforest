import os
from rastervision.core.data import make_od_scene, ClassConfig
from rastervision.pytorch_learner.dataset import ObjectDetectionSlidingWindowGeoDataset
from rastervision.pytorch_learner.dataset.visualizer import ObjectDetectionVisualizer

from engine.detector.utils import display_train_valid_test_aoi


class DetectorTrainingPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.working_folder = str(os.path.join(config["OUTPUT_FOLDER"], config["OUTPUT_NAME"]))

        if not os.path.exists(self.working_folder):
            os.mkdir(self.working_folder)

        self.rgb_path = str(os.path.join(config["DATA_ROOT"], config["TIF_NAME"]))
        self.trees_bbox_path = str(os.path.join(config["DATA_ROOT"], config["BBOX_TREES_NAME"]))
        self.train_bbox_path = str(os.path.join(config["DATA_ROOT"], config["BBOX_TRAIN_NAME"]))
        self.valid_bbox_path = str(os.path.join(config["DATA_ROOT"], config["BBOX_VALID_NAME"]))
        self.test_bbox_path = str(os.path.join(config["DATA_ROOT"], config["BBOX_TEST_NAME"]))

        self.chip_size = config["CHIP_SIZE"]
        self.chip_stride = config["CHIP_STRIDE"]

        self.class_config = ClassConfig(names=['tree'], colors=['white'])

        self.train_scene = self._create_scene(aoi_uri=self.train_bbox_path)
        self.valid_scene = self._create_scene(aoi_uri=self.valid_bbox_path)
        self.test_scene = self._create_scene(aoi_uri=self.test_bbox_path)

        self.train_dataset = ObjectDetectionSlidingWindowGeoDataset(
            scene=self.train_scene,
            size=self.chip_size,
            stride=self.chip_stride)

    def _create_scene(self, aoi_uri: str):
        scene = make_od_scene(
            class_config=self.class_config,
            image_uri=self.rgb_path,
            aoi_uri=aoi_uri,
            label_vector_uri=self.trees_bbox_path,
            label_vector_default_class_id=self.class_config.get_class_id('tree'),
            image_raster_source_kw=dict(allow_streaming=True))

        return scene

    def plot_data_to_disk(self, show_images: bool):
        display_train_valid_test_aoi(train_scene=self.train_scene,
                                     valid_scene=self.valid_scene,
                                     test_scene=self.test_scene,
                                     show_image=show_images,
                                     output_file=str(os.path.join(self.working_folder, "detector_aois.png")))

        vis = ObjectDetectionVisualizer(
            class_names=self.class_config.names, class_colors=self.class_config.colors)

        x, y = vis.get_batch(self.train_dataset, 10)
        vis.plot_batch(x, y, output_path=str(os.path.join(self.working_folder, "detector_sample_tiles.png")), show=show_images)


