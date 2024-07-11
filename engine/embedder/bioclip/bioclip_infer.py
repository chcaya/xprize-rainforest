import argparse
import os.path
import torch
import joblib
import numpy as np
import pandas as pd
import yaml

from engine.embedder.bioclip.data_init import data_loader_init_main
from engine.embedder.bioclip.two_layer_nn import TwoLayerNN
from engine.embedder.bioclip.bioclip_model import BioCLIPModel
from utils.config_utils import load_config

class BioClipInference:
    SUPPORTED_MODELS = ['nn']

    def __init__(self, config_path: str, downstream_model_path: str, model_type: str):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.bioclip_model = self._load_bioclip_model()
        self.downstream_model = self._load_downstream_model(downstream_model_path)
        self.label_encoder = joblib.load('archive/label_encoder.pkl')

    def load_config(self, config_path: str):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def _load_bioclip_model(self):
        return BioCLIPModel(self.config['training']['model_name'], self.config['training']['pretrained_path'])

    def _load_downstream_model(self, model_path: str):
        assert self.model_type in self.SUPPORTED_MODELS, \
            f"Invalid downstream model type: '{self.model_type}'. Valid values are {self.SUPPORTED_MODELS}."

        if self.model_type == 'nn':
            model = TwoLayerNN(512, self.config['training']['hidden_dim'], 46).to(self.device)
        else:
            raise ValueError("Invalid downstream model type. Choose 'nn'.")

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def save_predictions(self, final_labels, final_predictions, final_conf, output_file=None):
        results = pd.DataFrame({
            "image": final_labels,
            "predictions": final_predictions,
            "confidence": final_conf
        })

        if output_file:
            results.to_csv(output_file, index=False)
            print(f"Predictions saved to {output_file}")

        print(f"Inference completed.")
        return results

    def __call__(self, config_file: str,
                 save_predictions: bool = False,
                 output_file: str = None,
                 confidence_threshold: float = 0.95):
        config = self.load_config(config_file)
        set_id = config['data']['folder_pattern'].split('/')[-3]

        embeddings_path = f'embeddings/embeddings_{set_id}.pt'

        if os.path.exists(embeddings_path):
            all_embeddings = torch.load(embeddings_path)
            all_labels = np.load(f"embeddings/labels_{set_id}.npy")
            all_folder_names = np.load(f"embeddings/image_names_{set_id}.npy")
        else:
            all_embeddings = None

        if all_embeddings is None:
            data_loader = data_loader_init_main(config_file)
            all_embeddings, all_labels, all_folder_names = [], [], []
            for batch_idx, (image_tensors, batch_labels, folders_names) in enumerate(data_loader):
                print(f'batch: {batch_idx}/{len(data_loader)}')
                batch_embeddings = self.bioclip_model.generate_embeddings(image_tensors)
                all_embeddings.append(batch_embeddings)
                all_labels.extend(batch_labels)
                all_folder_names.extend(folders_names)

            all_embeddings = torch.cat(all_embeddings)
            all_labels = np.array(all_labels)
            all_folder_names = np.array(all_folder_names)

            torch.save(all_embeddings, f'embeddings/embeddings_{set_id}.pt')
            np.save(f"embeddings/labels_{set_id}.npy", all_labels)
            np.save(f"embeddings/image_names_{set_id}.npy", all_folder_names)

        final_labels, final_predictions, final_conf = [], [], []

        all_folder_names = np.array(all_folder_names)
        unique_folders = np.unique(all_folder_names)
        for folder in unique_folders:
            folder_indices = all_folder_names == folder
            folder_embeddings = all_embeddings[all_folder_names == folder]

            logits = self.downstream_model(folder_embeddings)

            probabilities = torch.softmax(logits, dim=1)
            confidence_scores = torch.max(probabilities, dim=1)[0]
            confidence_scores = torch.where(confidence_scores < confidence_threshold, torch.tensor(0.0), confidence_scores)

            confidence_scores = confidence_scores.view(-1, 1)
            weighted_probabilities = torch.matmul(confidence_scores.T, probabilities)
            normalized_probabilities = torch.softmax(weighted_probabilities, dim=1)

            top3_probabilities, top3_indices = torch.topk(normalized_probabilities, 3)
            top3_class_labels = self.label_encoder.inverse_transform(top3_indices.cpu().numpy()[0])
            top3_confidences = top3_probabilities.cpu().detach().numpy()[0].tolist()

            print("Probabilities Matrix:", probabilities)
            print("Confidence Scores:", confidence_scores)
            print("Weighted Probabilities:", weighted_probabilities)
            print("Normalized Probabilities:", normalized_probabilities)
            print("Top 3 Predicted Class Indices:", top3_indices)
            print("Top 3 Predicted Class Labels:", top3_class_labels)
            print("Top 3 Confidence Scores:", top3_confidences)

            final_labels.append(folder)
            final_predictions.append(top3_class_labels)
            final_conf.append(top3_confidences)

        output_file = f'{set_id}_' + output_file
        self.save_predictions(final_labels, final_predictions, final_conf, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioCLIP Model Inference")
    parser.add_argument('--config', type=str, default='configs/config_test.yaml', help='Path to the configuration file')
    parser.add_argument('--model', type=str, required=True, help='Path to the saved downstream model state dict')
    parser.add_argument('--model_type', type=str, choices=['nn'], required=True, help='Type of downstream model to load')
    parser.add_argument('--save_predictions', action='store_true', help='Flag to save predictions to a file')
    parser.add_argument('--output_file', type=str, help='File path to save predictions')

    args = parser.parse_args()

    inference = BioClipInference(args.config, args.model, args.model_type)
    inference(args.config, save_predictions=args.save_predictions, output_file=args.output_file)
