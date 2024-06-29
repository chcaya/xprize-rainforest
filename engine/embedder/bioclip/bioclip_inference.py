# this file is a WIP

import argparse
from pathlib import Path
import torch
import yaml
import pandas as pd
from torch.utils.data import DataLoader
from dataset import BioClipDataset
from file_loader import FileLoader
from bioclip_model import BioCLIPModel
from downstream_trainer import DownstreamModelTrainer

class BioClipInference:
    SUPPORTED_MODELS = ['knn', 'svc', 'nn']

    def __init__(self, config_path: str, downstream_model_path: str, model_type: str):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.bioclip_model = self._load_bioclip_model()
        self.downstream_model = self._load_downstream_model(downstream_model_path)

    def load_config(self, config_path: str):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def _load_bioclip_model(self):
        return BioCLIPModel(self.config['model_name'], self.config['pretrained_path']).to(self.device)

    def _load_downstream_model(self, model_path: str):
        assert self.model_type in self.SUPPORTED_MODELS, \
            f"Invalid downstream model type: '{self.model_type}'. Valid values are {self.SUPPORTED_MODELS}."
        #todo: simplify model loading
        trainer = DownstreamModelTrainer(self.config)
        if self.model_type == 'knn':
            raise NotImplementedError
        elif self.model_type == 'svc':
            raise NotImplementedError
        elif self.model_type == 'nn':
            raise NotImplementedError
        else:
            raise ValueError("Invalid downstream model type. Choose from 'knn', 'svc', or 'nn'.")

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def __call__(self, image_dir: str, save_predictions: bool = False, output_file: str = None):
        file_loader = FileLoader(
            dir_path=Path(image_dir),
            taxonomy_file=self.config['taxonomy_file']
        )

        taxonomy_data = file_loader.get_taxonomy_data()
        # todo: move this to config or argparse
        folders = file_loader.get_folders("dji/zoomed_out/cropped/*")

        image_paths = []
        for folder in folders:
            image_paths.extend(file_loader.get_image_paths(folder))

        dataset = BioClipDataset(image_paths, taxonomy_data, self.bioclip_model.preprocess_val)
        data_loader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=4)

        all_embeddings, file_paths = [], []
        for images, labels in data_loader:
            images = images.to(self.device)
            embeddings = self.bioclip_model.generate_embeddings(images)
            all_embeddings.append(embeddings)
            file_paths.extend(dataset.image_paths)

        all_embeddings = torch.cat(all_embeddings).cpu()

        if self.model_type == 'nn':
            self.downstream_model.eval()
            with torch.no_grad():
                outputs = self.downstream_model(all_embeddings)
                predictions = torch.argmax(outputs, dim=1)
        else:
            predictions = self.downstream_model.predict(all_embeddings)

        predictions = predictions.cpu().numpy()

        df = pd.DataFrame({
            'file_path': file_paths,
            'prediction': predictions
        })

        if save_predictions and output_file:
            df.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")

        print(f"Inference completed. Predictions: {predictions}")
        return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioCLIP Model Inference")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    parser.add_argument('--model', type=str, required=True, help='Path to the saved downstream model state dict')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the data for inference')
    parser.add_argument('--model_type', type=str, choices=['knn', 'svc', 'nn'], required=True, help='Type of downstream model to load')
    parser.add_argument('--save_predictions', action='store_true', help='Flag to save predictions to a file')
    parser.add_argument('--output_file', type=str, help='File path to save predictions')

    args = parser.parse_args()

    inference = BioClipInference(args.config, args.model, args.model_type)
    predictions_df = inference(args.data_dir, save_predictions=args.save_predictions, output_file=args.output_file)
