from pathlib import Path

import joblib
import torch
from geodataset.utils import COCOGenerator, decode_rle_to_polygon, CocoNameConvention

from engine.embedder.siamese.siamese_model import SiameseNetwork2
from engine.embedder.siamese.siamese_train import infer_model, valid_collate_fn
from geodataset.dataset.polygon_dataset import SiameseValidationDataset


def siamese_classifier(data_roots, fold, siamese_checkpoint, scaler_checkpoint, svc_checkpoint, batch_size: int, product_name: str, ground_resolution: float, scale_factor: float, output_path: Path):
    dataset = SiameseValidationDataset(
        fold=fold,
        root_path=[Path(root) for root in data_roots]
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=3,
                                         collate_fn=valid_collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseNetwork2()
    print(f'Loading model from {siamese_checkpoint}')
    weights = torch.load(siamese_checkpoint)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    labels, embeddings = infer_model(model, loader, device)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print(labels.shape, embeddings.shape)

    scaler = joblib.load(scaler_checkpoint)
    svc = joblib.load(svc_checkpoint)

    X = scaler.transform(embeddings)
    preds = svc.predict(X)

    tiles_paths = [value["path"] for key, value in sorted(dataset.tiles.items(), key=lambda item: item[0])]

    segmentations = [[decode_rle_to_polygon(value["labels"][0]['segmentation'])] for key, value in sorted(dataset.tiles.items(), key=lambda item: item[0])]

    output_path.mkdir(parents=True, exist_ok=True)

    coco_name = CocoNameConvention.create_name(
        product_name=product_name,
        fold='inferclassifier',
        ground_resolution=ground_resolution,
        scale_factor=scale_factor,
    )

    coco_json_output_path = output_path / coco_name

    coco_generator = COCOGenerator(
        description=f"Aggregated boxes from multiple tiles.",
        tiles_paths=tiles_paths,
        polygons=segmentations,
        scores=None,
        categories=[[int(x)] for x in preds],
        other_attributes=None,
        output_path=coco_json_output_path,
        use_rle_for_labels=True,
        n_workers=5,  # TODO make this a parameter to the class
        coco_categories_list=None  # TODO make this a parameter to the class
    )

    coco_generator.generate_coco()

    return coco_json_output_path
