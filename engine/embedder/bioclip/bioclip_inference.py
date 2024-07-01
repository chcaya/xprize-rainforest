# this file is a WIP

import argparse
import torch
import yaml
import pandas as pd

from engine.embedder.bioclip.bioclip_model import BioCLIPModel
from engine.embedder.bioclip.downstream_trainer import DownstreamModelTrainer
from engine.embedder.bioclip.data_init import data_loader_init_main

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

    def save_predictions(self, file_paths, predictions, output_file = None):
        df = pd.DataFrame({
            'file_path': file_paths,
            'prediction': predictions
        })

        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")

        print(f"Inference completed. Predictions: {predictions}")
        return df

    def __call__(self, image_dir: str, save_predictions: bool = False, output_file: str = None):

        data_loader = data_loader_init_main('config.yaml')

        all_embeddings, file_paths = [], []

        # todo: for inference loader file path needs to be sent
        for images, labels in data_loader:
            images = images.to(self.device)
            embeddings = self.bioclip_model.generate_embeddings(images)
            all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings).cpu()

        # predict on downstream
        # todo: give confidence score from logits
        if self.model_type == 'nn':
            self.downstream_model.eval()
            with torch.no_grad():
                outputs = self.downstream_model(all_embeddings)
                predictions = torch.argmax(outputs, dim=1)
        else:
            predictions = self.downstream_model.predict(all_embeddings)

        predictions = predictions.cpu().numpy()
        predictions_df = self.save_predictions(file_paths, predictions, output_file)
        return predictions_df

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
