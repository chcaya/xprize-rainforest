import numpy as np
import torch
from shapely import box


def collate_fn_detection(batch):
    if type(batch[0][0]) is np.ndarray:
        data = np.array([item[0] for item in batch])
        data = torch.tensor(data, dtype=torch.float32)
    else:
        data = torch.tensor([item[0] for item in batch], dtype=torch.float32)

    labels = [{'boxes': torch.tensor(item[1]['boxes'], dtype=torch.float32),
               'labels': torch.tensor(item[1]['labels'], dtype=torch.long)} for item in batch]

    return data, labels


def collate_fn_images(batch):
    if type(batch[0]) is np.ndarray:
        data = np.array([item for item in batch])
        data = torch.tensor(data, dtype=torch.float32)
    else:
        data = torch.tensor([item for item in batch], dtype=torch.float32)

    return data


def detector_result_to_lists(detector_result):
    detector_result = [{k: v.cpu().numpy() for k, v in x.items()} for x in detector_result]
    for x in detector_result:
        x['boxes'] = [box(*b) for b in x['boxes']]
        x['scores'] = x['scores'].tolist()
    boxes = [x['boxes'] for x in detector_result]
    scores = [x['scores'] for x in detector_result]

    return boxes, scores
