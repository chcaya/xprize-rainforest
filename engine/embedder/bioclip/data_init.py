import yaml
from pathlib import Path
from torch.utils.data import DataLoader

from file_loader import BioClipFileLoader
from dataset import BioClipDataset
from bioclip_model import BioCLIPModel
# Assuming BioClipFileLoader and BioClipDataset are already defined

def data_loader_init_main(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

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
    for folder in folders:
        image_paths.extend(file_loader.get_image_paths(folder))

    # Replace 'bioclip_model' with your actual model loading code
    bioclip_model = BioCLIPModel(config['training']['model_name'], config['training']['pretrained_path'])
    dataset = BioClipDataset(image_paths, taxonomy_data, bioclip_model.preprocess_val)
    data_loader = DataLoader(dataset, batch_size=config['training']['batch_size'],
                             shuffle=config['data']['shuffle'], num_workers=config['training']['num_workers'])

    return data_loader

if __name__ == "__main__":
    data_loader = data_loader_init_main('configs/config.yaml')
    # You can now use the data_loader as needed
    for idx, (image_tensors, labels) in enumerate(data_loader):
        print (idx, image_tensors)
