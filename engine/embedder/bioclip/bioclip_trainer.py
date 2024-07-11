import argparse
from pathlib import Path

import joblib
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

from bioclip_model import BioCLIPModel
from downstream_trainer import DownstreamModelTrainer
from dataset import BioClipDataset
from file_loader import BioClipFileLoader
from utils.visualization import plot_embeddings
from utils.config_utils import load_config
from engine.embedder.bioclip.data_init import data_loader_init_main



def main(config_path, visualize_embeddings, num_folders, downstream):
    config = load_config(config_path)

    bioclip_model = BioCLIPModel(config['training']['model_name'], config['training']['pretrained_path'])
    trainer = DownstreamModelTrainer(config)
    data_loader = data_loader_init_main('configs/config.yaml')


    all_embeddings, all_labels = [], []
    for batch_idx, (images, labels) in enumerate(data_loader):
        print (f'{batch_idx}/{len(data_loader)}')
        embeddings = bioclip_model.generate_embeddings(images)
        all_embeddings.append(embeddings)
        all_labels.extend(labels)

    all_embeddings = torch.cat(all_embeddings)
    torch.save(all_embeddings, 'archive/train_embeddings.pt')
    if visualize_embeddings:
        plot_embeddings(all_embeddings.numpy(), all_labels)

    X_train, X_test, y_train, y_test, label_encoder = trainer.preprocess_data(
        all_embeddings.numpy(), np.array(all_labels), test_size=config['training']['test_size'], random_state=config['training']['random_state'])

    joblib.dump(label_encoder, 'archive/label_encoder.pkl')


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
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration file')
    parser.add_argument('--visualize_embeddings', action='store_true', help='Flag to visualize embeddings')
    parser.add_argument('--num_folders', type=int, help='Number of folders to process (default is all)')
    parser.add_argument('--downstream', type=str, choices=['knn', 'svc', 'nn'], required=True,
                        help='Type of downstream model to train')

    args = parser.parse_args()

    main(args.config, args.visualize_embeddings, args.num_folders, args.downstream)
