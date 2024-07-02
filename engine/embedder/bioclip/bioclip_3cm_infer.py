from pathlib import Path

import torch
from PIL import Image


from file_loader import BioClipFileLoader
from utils.config_utils import load_config
from bioclip_model import BioCLIPModel
from utils.visualization import plot_embeddings

if __name__ == '__main__':

    config = load_config("./3cm_config.yaml")

    bioclip_model = BioCLIPModel(config['training']['model_name'], config['training']['pretrained_path'])


    dir_path = Path(config['data']['dir_path'])
    taxonomy_file = config['data']['taxonomy_file']
    folder_pattern = config['data']['folder_pattern']
    num_folders = config['data'].get('num_folders')

    file_loader = BioClipFileLoader(
        dir_path=dir_path,
        taxonomy_file=taxonomy_file
    )

    taxonomy_data = file_loader.get_taxonomy_data()
    folders = file_loader.get_folders(folder_pattern)

    if num_folders is not None:
        folders = folders[:num_folders]

    image_paths = []
    folder_to_image_paths = {}
    for folder in folders:
        folder_name = folder.split('/')[-1]
        folder_to_image_paths[folder_name] = file_loader.get_image_paths(folder)
        image_paths.extend(file_loader.get_image_paths(folder))

    image_tensors, labels = [], []
    for folder, image_paths in folder_to_image_paths.items():
        if folder in ["type_1", "type_2", "type_3"]:
            for image_path in image_paths:
                image = Image.open(image_path).convert("RGB")
                image_tensor = bioclip_model.preprocess_val(image).squeeze(0)
                image_tensors.append(image_tensor)
                labels.append(folder)

    image_tensors = torch.stack(image_tensors).unsqueeze(dim = 1)
    embeddings = bioclip_model.generate_embeddings(image_tensors)
    plot_embeddings(embeddings, labels)

    # todo: check finetuned downstream model embedding on this
    print (image_paths)