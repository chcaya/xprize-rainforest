from pathlib import Path

from geodataset.dataset import SegmentationLabeledRasterCocoDataset
from geodataset.dataset.polygon_dataset import SiameseValidationDataset
from geodataset.utils import CocoNameConvention

from config.config_parsers.embedder_parsers import EmbedderInferIOConfig, DINOv2InferConfig, SiameseInferConfig, \
    DINOv2InferIOConfig, SiameseInferIOConfig
# from engine.embedder.dinov2.dinov2 import DINOv2Inference
# from engine.embedder.siamese.siamese_infer import siamese_infer

from engine.embedder.bioclip.dataset import BioClipDataset
from engine.embedder.bioclip.file_loader import BioClipFileLoader
from engine.embedder.bioclip.bioclip_model import BioCLIPModel
from engine.embedder.bioclip.downstream_trainer import DownstreamModelTrainer


# this file needs to do:
# 1) load drone imagery for inference
# 2) run the tilerization of inference imagery
# 3) call the bioclip embedder to generate embeddings
# 4) perform prediction using chosen downstream model
# 5) convert results into csv or geodataframe output

def bioclip_main():
    """
    # 1) load drone imagery for inference
    # 2) run the tilerization of inference imagery
    # 3) call the bioclip embedder to generate embeddings
    # 4) perform prediction using chosen downstream model
    # 5) convert results into csv or geodataframe output
    """
    dataset = None
    config = None
    embeddings =  bioclip_embeddings_infer_main(config, dataset)

    raise NotImplementedError


def bioclip_downstream_infer_main(config, embeddings):
    #config must contain pretrain path
    # todo: update bioclip to accept config and dataset/image path
    classifier = DownstreamModelTrainer(config)
    # embeddings = classifier.infer
    return embeddings

def bioclip_embeddings_infer_main(config, dataset):
    #config must contain pretrain path
    # todo: update bioclip to accept config and dataset/image path
    embedder = BioCLIPModel(config)
    embeddings = embedder.generate_embeddings(dataset)
    return embeddings

#
# def dino_v2_infer_main(config: DINOv2InferConfig, segmentation_dataset: SegmentationLabeledRasterCocoDataset):
#     embedder = DINOv2Inference(config)
#
#     embeddings = embedder.infer_on_segmentation_dataset(segmentation_dataset)
#
#     return embeddings


# def siamese_infer_main(config: SiameseInferConfig, siamese_dataset: SiameseValidationDataset):
#     embeddings_df = siamese_infer(
#         siamese_dataset=siamese_dataset,
#         siamese_checkpoint=config.checkpoint_path,
#         batch_size=config.batch_size
#     )
#
#     return embeddings_df

# todo: adapt this
def bioclip_pipeline_infer_main(config):
    output_folder = Path(config.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)        # TODO change exist_ok back to False

    # 1) create tiles
    # product_name, scale_factor, ground_resolution, fold = CocoNameConvention.parse_name(Path(config.coco_path).name)

    tiles_path = Path(config.input_tiles_root)
    assert tiles_path.is_dir() and tiles_path.name == "tiles", \
        "The tiles_path must be the path of a directory named 'tiles'."

    if isinstance(config, DINOv2InferIOConfig):
        segmentation_dataset = SegmentationLabeledRasterCocoDataset(
            root_path=[
                Path(config.coco_path).parent,
                tiles_path.parent
            ],
            fold=fold
        )
        embeddings_df = dino_v2_infer_main(config=config, segmentation_dataset=segmentation_dataset)
    elif isinstance(config, SiameseInferIOConfig):
        siamese_dataset = SiameseValidationDataset(
            root_path=[
                Path(config.coco_path).parent,
                tiles_path.parent
            ],
            fold=fold
        )

        embeddings_df = siamese_infer_main(
            config=config,
            siamese_dataset=siamese_dataset
        )
    else:
        raise NotImplementedError

    output_path = output_folder / f"{product_name}_embeddings_{fold}.csv"
    embeddings_df.to_csv(output_path, index=False)

    config.save_yaml_config(output_path=output_folder / "embedder_infer_config.yaml")

    return output_path

def tilerize_dataset_main(config_path):
    raise NotImplementedError


# todo: use the below ref to finalize the bioclip_pipeline_main
def reference_main(config_path):
    config = load_config(config_path)

    bioclip_model = BioCLIPModel(config['model_name'], config['pretrained_path'])
    trainer = DownstreamModelTrainer(config)

    file_loader = BioClipFileLoader(
        dir_path=Path('/Users/daoud/PycharmAssets/xprize/'),
        taxonomy_file='photos_exif_taxo.csv'
    )

    taxonomy_data = file_loader.get_taxonomy_data()

    folders = file_loader.get_folders("dji/zoomed_out/cropped/*")
    if num_folders is not None:
        folders = folders[:num_folders]

    image_paths = []
    for folder in folders:
        image_paths.extend(file_loader.get_image_paths(folder))

    dataset = BioClipDataset(image_paths, taxonomy_data, bioclip_model.preprocess_val)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    all_embeddings, all_labels = [], []
    for images, labels in data_loader:
        embeddings = bioclip_model.generate_embeddings(images)
        all_embeddings.append(embeddings)
        all_labels.extend(labels)

    all_embeddings = torch.cat(all_embeddings)


def bioclip_pipeline_infer_main(config_path: str, model_path: str, data_dir: str, model_type: str, output_file: str = None):
    config = load_config(config_path)
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_folder = Path(config['output_folder'])
    output_folder.mkdir(exist_ok=True, parents=True)

    # 1) Load drone imagery for inference

    file_loader = FileLoader(
        dir_path=Path(data_dir),
        taxonomy_file=config['taxonomy_file']
    )

    taxonomy_data = file_loader.get_taxonomy_data()
    folders = file_loader.get_folders("dji/zoomed_out/cropped/*")

    image_paths = []
    for folder in folders:
        image_paths.extend(file_loader.get_image_paths(folder))

    # 2) Run the tilerization of inference imagery (assumed to be handled externally)
    # todo: call tilerize code

    # 3) Load


    # 3) Call the BioCLIP embedder to generate embeddings
    bioclip_model = BioCLIPModel(config['model_name'], config['pretrained_path']).to(config['device'])
    dataset = BioClipDataset(image_paths, taxonomy_data, bioclip_model.preprocess_val)
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    all_embeddings, file_paths = [], []
    for images, _ in data_loader:
        images = images.to(config['device'])
        embeddings = bioclip_model.generate_embeddings(images)
        all_embeddings.append(embeddings.cpu())
        file_paths.extend(image_paths)

    all_embeddings = torch.cat(all_embeddings)

    # 4) Perform prediction using chosen downstream model
    inference = BioClipInference(config_path, model_path, model_type)
    predictions_df = inference.__call__(data_dir)

    # 5) Convert results into CSV or GeoDataFrame output
    if output_file:
        predictions_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    else:
        print(predictions_df)

    return predictions_df
