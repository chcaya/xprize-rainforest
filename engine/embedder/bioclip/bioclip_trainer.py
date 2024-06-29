import argparse
from pathlib import Path
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

from bioclip_model import BioCLIPModel
from downstream_trainer import DownstreamModelTrainer
from dataset import BioClipDataset
from file_loader import FileLoader
from utils.visualization import plot_embeddings


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main(config_path, visualize_embeddings, num_folders, downstream):
    config = load_config(config_path)

    bioclip_model = BioCLIPModel(config['model_name'], config['pretrained_path'])
    trainer = DownstreamModelTrainer(config)

    file_loader = FileLoader(
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

    if visualize_embeddings:
        plot_embeddings(all_embeddings.numpy(), all_labels)

    X_train, X_test, y_train, y_test, label_encoder = trainer.preprocess_data(
        all_embeddings.numpy(), np.array(all_labels), test_size=config['test_size'], random_state=config['random_state'])

    if downstream == 'knn':
        model, classifier_preds, accuracy, report, cm = trainer.train_knn(X_train, X_test, y_train, y_test)
    elif downstream == 'svc':
        model, classifier_preds, accuracy, report, cm = trainer.train_svc(X_train, X_test, y_train, y_test)
    elif downstream == 'nn':
        model, classifier_preds, accuracy, report, cm = trainer.train_nn(X_train, X_test, y_train, y_test)
    else:
        raise ValueError("Invalid downstream model type. Choose from 'knn', 'svc', or 'nn'.")

    class_names = label_encoder.inverse_transform(np.unique(y_test))
    trainer.plot_confusion_matrix(cm, class_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioCLIP Model Training and Inference")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    parser.add_argument('--visualize_embeddings', action='store_true', help='Flag to visualize embeddings')
    parser.add_argument('--num_folders', type=int, help='Number of folders to process (default is all)')
    parser.add_argument('--downstream', type=str, choices=['knn', 'svc', 'nn'], required=True,
                        help='Type of downstream model to train')

    args = parser.parse_args()

    main(args.config, args.visualize_embeddings, args.num_folders, args.downstream)
