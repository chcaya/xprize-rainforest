import numpy as np
import shapely
from segment_anything import SamPredictor, sam_model_registry


class SamPredictorWrapper:
    def __init__(self):
        sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")     #TODO set model_type and download checkpoint
        self.predictor = SamPredictor(sam)

    def infer(self, image: np.ndarray, box: shapely.box):
        self.predictor.set_image(image)
        box_array = np.array(box.bounds)
        masks, _, _ = self.predictor.predict(box_array)
        return masks
