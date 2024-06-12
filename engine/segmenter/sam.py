import time
from typing import List

import numpy as np
import psutil
import torch
from geodataset.dataset import DetectionLabeledRasterCocoDataset
import multiprocessing

from segment_anything import SamPredictor, sam_model_registry

from torch.utils.data import DataLoader
from tqdm import tqdm

from engine.segmenter.utils import mask_to_polygon, sam_collate_fn


def get_memory_usage():
    memory_info = psutil.virtual_memory()
    memory_percentage = memory_info.percent

    return memory_percentage


def process_masks(queue, output_dict, simplify_tolerance):
    results = []
    while True:
        item = queue.get()
        if item is None:
            break
        tile_idx, masks, scores = item
        masks_polygons = [mask_to_polygon(mask.squeeze(), simplify_tolerance=simplify_tolerance) for mask in masks]
        results.append((tile_idx, masks_polygons, scores.squeeze().tolist()))
        queue.task_done()  # Indicate that the task is complete

    for result in results:
        output_dict[result[0]] = (result[1], result[2])


class SamPredictorWrapper:
    def __init__(self,
                 model_type: str,
                 checkpoint_path: str,
                 simplify_tolerance: float,
                 n_postprocess_workers: int,
                 box_batch_size: int):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.simplify_tolerance = simplify_tolerance
        self.n_postprocess_workers = n_postprocess_workers
        self.box_batch_size = box_batch_size
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        sam.to(self.device)
        self.predictor = SamPredictor(sam)

    def _infer(self, image: np.ndarray, boxes: List[np.array]):
        self.predictor.set_image(image)
        box_array = np.array(boxes)
        box_tensor = torch.Tensor(box_array).to(torch.long)

        all_masks = []
        all_scores = []

        for i in range(0, len(box_tensor), self.box_batch_size):
            box_batch = box_tensor[i:i + self.box_batch_size].to(self.device)
            masks, scores, low_res_masks = self.predictor.predict_torch(point_coords=None,
                                                                        point_labels=None,
                                                                        boxes=box_batch,
                                                                        multimask_output=False)
            masks, scores = masks.cpu(), scores.cpu()

            all_masks.append(masks)
            all_scores.append(scores)

        all_masks = torch.cat(all_masks, dim=0)
        all_scores = torch.cat(all_scores, dim=0)

        return all_masks, all_scores

    def infer_on_multi_box_dataset(self, dataset: DetectionLabeledRasterCocoDataset):
        infer_dl = DataLoader(dataset, batch_size=1, shuffle=False,
                              collate_fn=sam_collate_fn,
                              num_workers=3, persistent_workers=True)

        dataset_with_progress = tqdm(infer_dl,
                                     desc="Inferring SAM...",
                                     leave=True)

        tiles_paths = []
        tiles_masks_polygons = []
        tiles_masks_scores = []
        queue = multiprocessing.JoinableQueue()  # Create a JoinableQueue

        # Create a manager to share data across processes
        manager = multiprocessing.Manager()
        output_dict = manager.dict()

        # Start post-processing processes
        post_process_processes = []
        for _ in range(self.n_postprocess_workers):
            p = multiprocessing.Process(target=process_masks, args=(queue, output_dict, self.simplify_tolerance))
            p.start()
            post_process_processes.append(p)

        for tile_idx, sample in enumerate(dataset_with_progress):
            if get_memory_usage() > 90:
                print("Memory usage is high. Pausing for 10 seconds...")
                time.sleep(10)

            image, boxes_data = sample
            image = image[:3, :, :]
            image_hwc = image.transpose((1, 2, 0))
            image_hwc = (image_hwc * 255).astype(np.uint8)
            masks, scores = self._infer(image=image_hwc, boxes=boxes_data['boxes'])
            masks = masks.numpy()
            scores = scores.numpy()
            tiles_paths.append(dataset.tiles[tile_idx]['path'])

            # Put masks and scores into the queue for post-processing
            queue.put((tile_idx, masks, scores))

        # Wait for all tasks in the queue to be completed
        queue.join()

        # Signal the end of input to the queue
        for _ in range(self.n_postprocess_workers):
            queue.put(None)

        # Wait for post-processing processes to finish
        for p in post_process_processes:
            p.join()

        # Close the queue
        queue.close()

        # Assemble the results into tiles_masks_polygons
        for tile_idx in sorted(output_dict.keys()):
            masks_polygons, scores = output_dict[tile_idx]
            tiles_masks_polygons.append(masks_polygons)
            tiles_masks_scores.append(scores)

        return tiles_paths, tiles_masks_polygons, tiles_masks_scores
