import os.path

import joblib
import numpy as np
import pandas as pd
import torch

from engine.embedder.bioclip.data_init import data_loader_init_main
from engine.embedder.bioclip.two_layer_nn import TwoLayerNN
from utils.config_utils import load_config
from bioclip_model import BioCLIPModel

if __name__ == '__main__':

    config = load_config("./config_test.yaml")
    confidence_threshold = 0.95
    set_id = config['data']['folder_pattern'].split('/')[-3]

    bioclip_model = BioCLIPModel(config['training']['model_name'], config['training']['pretrained_path'])
    data_loader = data_loader_init_main('./config_test.yaml')
    downstream_model = TwoLayerNN(512, config['training']['hidden_dim'], 46)
    downstream_model.load_state_dict(torch.load('downstream/preds/downstream_nn.pth'))
    label_encoder = joblib.load('archive/label_encoder.pkl')

    embeddings_path = f'embeddings/embeddings_{set_id}.pt'

    if os.path.exists(embeddings_path):
        all_embeddings = torch.load(embeddings_path)
        all_labels = np.load(f"embeddings/labels_{set_id}.npy")
        all_folder_names = np.load(f"embeddings/image_names_{set_id}.npy")
    else:
        all_embeddings = None

    # if you were unable to load generate embeddings
    if all_embeddings is None:
        # Extract features using the model
        all_embeddings, all_labels, all_folder_names = [], [], []
        for batch_idx, (image_tensors, batch_labels, folders_names) in enumerate(data_loader):
            print(f'batch: {batch_idx}/{len(data_loader)}')
            batch_embeddings = bioclip_model.generate_embeddings(image_tensors)
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

    # downstream_model = DownstreamModelTrainer(config)
    all_folder_names = np.array(all_folder_names)
    unique_folders = np.unique(all_folder_names)
    for folder in unique_folders:
        folder_indices = all_folder_names == folder
        folder_embeddings = all_embeddings[all_folder_names == folder]

        logits = downstream_model(folder_embeddings)

        # Convert logits to probabilities using softmax
        probabilities = torch.softmax(logits, dim=1)
        # Get the confidence scores (maximum probabilities) for each image
        confidence_scores = torch.max(probabilities, dim=1)[0]
        confidence_scores = torch.where(confidence_scores < confidence_threshold, torch.tensor(0.0), confidence_scores)

        # Reshape the confidence scores to enable matrix multiplication
        confidence_scores = confidence_scores.view(-1, 1)

        # Weigh the class probabilities by their confidence scores using matrix multiplication
        weighted_probabilities = torch.matmul(confidence_scores.T, probabilities)

        # Apply softmax to the weighted probabilities to get normalized probabilities
        normalized_probabilities = torch.softmax(weighted_probabilities, dim=1)

        # Get the top 3 predicted class indices and their confidence scores
        top3_probabilities, top3_indices = torch.topk(normalized_probabilities, 3)

        # Map the predicted class indices to class labels
        top3_class_labels = label_encoder.inverse_transform(top3_indices.cpu().numpy()[0])

        # Convert the top 3 probabilities to a list of floats
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

    results = pd.DataFrame({
        "image": final_labels,
        "predictions": final_predictions,
        "confidence": final_conf
    })

    results.to_csv(f"./{set_id}_results_top3.csv")