from rastervision.core.data import make_od_scene
from pathlib import Path
from rastervision.core.data import ClassConfig

from engine.detector.utils import display_train_valid_test_aoi


def train_pipeline(config):
    rgb_path = str(Path.joinpath(Path(config["DATA_ROOT"]), Path(config["TIF_NAME"])))
    trees_path = str(Path.joinpath(Path(config["DATA_ROOT"]), Path(config["BBOX_TREES_NAME"])))
    train_box_path = str(Path.joinpath(Path(config["DATA_ROOT"]), Path(config["BBOX_TRAIN_NAME"])))
    valid_box_path = str(Path.joinpath(Path(config["DATA_ROOT"]), Path(config["BBOX_VALID_NAME"])))
    test_box_path = str(Path.joinpath(Path(config["DATA_ROOT"]), Path(config["BBOX_TEST_NAME"])))

    class_config = ClassConfig(names=['tree'], colors=['white'])

    train_scene = make_od_scene(
        class_config=class_config,
        image_uri=rgb_path,
        aoi_uri=train_box_path,
        label_vector_uri=trees_path,
        label_vector_default_class_id=class_config.get_class_id('tree'),
        image_raster_source_kw=dict(allow_streaming=True))

    valid_scene = make_od_scene(
        class_config=class_config,
        image_uri=rgb_path,
        aoi_uri=valid_box_path,
        label_vector_uri=trees_path,
        label_vector_default_class_id=class_config.get_class_id('tree'),
        image_raster_source_kw=dict(allow_streaming=True))

    test_scene = make_od_scene(
        class_config=class_config,
        image_uri=rgb_path,
        aoi_uri=test_box_path,
        label_vector_uri=trees_path,
        label_vector_default_class_id=class_config.get_class_id('tree'),
        image_raster_source_kw=dict(allow_streaming=True))

    display_train_valid_test_aoi(train_scene=train_scene, valid_scene=valid_scene, test_scene=test_scene)

