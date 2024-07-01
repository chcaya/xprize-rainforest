import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

import torch
from torch.nn.functional import normalize
from bioclip_model import BioCLIPModel
from utils.config_utils import load_config
from data_init import data_loader_init_main
from utils.visualization import plot_embeddings, plot_confusion_matrix
import matplotlib.cm as cm


class BioClipActiveLearner:
    def __init__(self, model: BioCLIPModel, dbscan_eps=0.2, dbscan_min_samples=5, n_neighbors=5,
                 uncertainty_threshold=0.2,
                 n_clusters=10, rf_estimators=100):
        self.model = model
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.n_neighbors = n_neighbors
        self.uncertainty_threshold = uncertainty_threshold
        self.n_clusters = n_clusters
        self.rf_estimators = rf_estimators
        self.scaler = StandardScaler()

        self.db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.rf = RandomForestClassifier(n_estimators=rf_estimators, random_state=42)
        self.knn = NearestNeighbors(n_neighbors=n_neighbors)
        self.tsne = TSNE(n_components=2)
        self.umap = None

        self.labels = None
        self.knn_labels = None
        self.labeled_data = []
        self.labeled_labels = []

    def extract_features(self, image_tensors):
        return self.model.generate_embeddings(image_tensors)

    def reduce_dimensions(self, embeddings, method='tsne'):
        if method == 'tsne':
            return self.tsne.fit_transform(embeddings)
        elif method == 'umap':
            return self.umap.fit_transform(embeddings)
        else:
            raise ValueError("Method must be either 'tsne' or 'umap'")

    def cluster_data(self, features, method='dbscan'):
        features = self.scaler.fit_transform(features)
        if method == 'dbscan':
            self.labels = self.db.fit_predict(features)
        elif method == 'kmeans':
            self.labels = self.kmeans.fit_predict(features)
        else:
            raise ValueError("Method must be either 'dbscan' or 'kmeans'")
        return self.labels

    def fit_knn(self, features, labels):
        self.knn.fit(features, labels)
        self.knn_labels = labels

    def classify_new_data(self, new_data_features):
        # Transform the new data features using the same scaler
        new_data_features = self.scaler.transform(new_data_features)
        # Find the k nearest neighbors for each new data point
        distances, indices = self.knn.kneighbors(new_data_features)
        # Predict labels based on the majority vote from the nearest neighbors
        predicted_labels = []
        for idx in indices:
            neighbor_labels = self.knn_labels[idx]
            predicted_label = self.most_frequent_element(neighbor_labels)
            predicted_labels.append(predicted_label)
        return np.array(predicted_labels)

    def most_frequent_element(self, array):
        unique_elements, counts = np.unique(array, return_counts=True)
        max_index = counts.argmax()
        return unique_elements[max_index]

    def train_rf_classifier(self, labeled_indices, features, labels):
        labeled_features = features[labeled_indices]
        self.rf.fit(labeled_features, labels)

    def propagate_labels(self, features):
        return self.rf.predict(features)

    def uncertainty_estimation(self, new_data_features):
        new_data_features = self.scaler.transform(new_data_features)
        distances, _ = self.knn.kneighbors(new_data_features)
        uncertainties = np.mean(distances, axis=1)
        return uncertainties

    def select_samples_for_labeling(self, new_data_features, num_samples):
        uncertainties = self.uncertainty_estimation(new_data_features)
        selected_indices = np.argsort(uncertainties)[-num_samples:]
        return selected_indices

    def visualize_dbscan_output(self, features):
        # Reduce dimensionality for visualization
        reduced_features = self.reduce_dimensions(features, method='tsne')

        # Plot the clusters
        plt.figure(figsize=(10, 7))
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            if label == -1:
                # Noise
                plt.scatter(reduced_features[self.labels == label, 0], reduced_features[self.labels == label, 1], s=50,
                            c='k', marker='x', label='Noise')
            else:
                plt.scatter(reduced_features[self.labels == label, 0], reduced_features[self.labels == label, 1], s=50,
                            label=f'Cluster {label}')

        plt.title('DBSCAN Clusters')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.show()

    def visualize_clusters(self, labeled_features, unlabeled_features, labeled_labels, new_labels, iteration):
        # Reduce dimensionality for visualization
        all_features = np.concatenate([labeled_features, unlabeled_features], axis=0)
        all_labels = np.concatenate([labeled_labels, new_labels], axis=0)
        reduced_features = self.reduce_dimensions(all_features, method='tsne')

        # Ensure labels are numpy arrays
        labeled_labels = np.array(labeled_labels)
        new_labels = np.array(new_labels)

        # Separate labeled and unlabeled data
        num_labeled = len(labeled_features)
        labeled_reduced = reduced_features[:num_labeled]
        unlabeled_reduced = reduced_features[num_labeled:]

        # Create a color map
        unique_labels = np.unique(np.concatenate((labeled_labels, new_labels)))
        colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: color for label, color in zip(unique_labels, colors)}

        # Plot the clusters
        plt.figure(figsize=(10, 7))
        for label in np.unique(all_labels):
            labeled_points = labeled_reduced[labeled_labels == label]
            unlabeled_points = unlabeled_reduced[new_labels == label]
            plt.scatter(labeled_points[:, 0], labeled_points[:, 1], s=50, color=color_map[label],
                        label=f'GT Labelled Class {label}: {len(labeled_points)}')
            plt.scatter(unlabeled_points[:, 0], unlabeled_points[:, 1], s=100, facecolors='none',
                        edgecolors=color_map[label], label=f'Unseen point predicted Class {label}: {len(unlabeled_points)}')

        plt.title(f'Clusters after iteration {iteration}, Labelled Points: {len(labeled_labels)}/{len(all_labels)}')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

    def align_clusters_with_labels(self, dbscan_labels, true_labels, num_samples=5):
        label_map = {}
        all_sampled_idx = []
        for cluster_label in np.unique(dbscan_labels):
            if cluster_label != -1:  # Exclude noise points
                # all indices where dbcan labels match current cluster (note: dbscan labels are in order of true_labels)
                cluster_indices = np.where(dbscan_labels == cluster_label)[0]
                sampled_indices = random.sample(list(cluster_indices), min(num_samples, len(cluster_indices)))
                all_sampled_idx.extend(sampled_indices)
                sampled_true_labels = true_labels[sampled_indices]
                most_common_label = self.most_frequent_element(sampled_true_labels)
                label_map[cluster_label] = most_common_label
        aligned_labels = np.array([label_map[label] if label != -1 else -1 for label in dbscan_labels])
        return aligned_labels, all_sampled_idx

### Main Method

def main():
    # Assuming `YourPretrainedModel` is a PyTorch model and `dataloader` is a PyTorch DataLoader
    config = load_config(config_path='config.yaml')
    data_loader = data_loader_init_main('config.yaml')

    model = BioCLIPModel(config['training']['model_name'], config['training']['pretrained_path'])
    clusterer = BioClipActiveLearner(model)

    all_embeddings, all_labels = [], []
    # Extract features using the model
    for idx, (image_tensors, labels) in enumerate(data_loader):
        features = clusterer.extract_features(image_tensors)
        all_embeddings.append(features)
        all_labels.extend(labels)

    all_embeddings = torch.cat(all_embeddings)
    all_labels = np.array(all_labels)
    # Dimensionality reduction (optional, for visualization)

    all_embeddings_reduced = clusterer.reduce_dimensions(all_embeddings, method='tsne')
    # Perform initial clustering with DBSCAN and visualize the output
    dbscan_labels = clusterer.cluster_data(all_embeddings_reduced, method='dbscan')
    clusterer.visualize_dbscan_output(all_embeddings)
    # reset scaler to correct dims
    clusterer.scaler.fit(all_embeddings)


    # Align DBSCAN clusters with true labels by sampling
    sample_points_per_cluster = 10
    aligned_labels, sampled_idx = clusterer.align_clusters_with_labels(dbscan_labels, all_labels, num_samples=10)

    # Use aligned labels to train k-NN
    dbscan_indices = np.where(dbscan_labels != -1)[0]  # Exclude noise points
    dbscan_features = all_embeddings[dbscan_indices]
    dbscan_labels_filtered = aligned_labels[dbscan_indices]

    # Train k-NN with aligned labels
    clusterer.fit_knn(dbscan_features, dbscan_labels_filtered)

    # Set initial labeled and unlabeled data
    labeled_features = all_embeddings[sampled_idx]
    labeled_labels = all_labels[sampled_idx]
    unlabeled_indices = list(set(range(len(all_embeddings))) - set(sampled_idx))
    unlabeled_features = all_embeddings[unlabeled_indices]

    silhouette_avg = silhouette_score(all_embeddings, dbscan_labels)
    print(f"Silhouette Score DBSCAN: {silhouette_avg}")

    num_iterations = 20
    num_samples = 20

    for iteration in range(num_iterations):
        # 1) fit a knn on the labelled examples we know
        clusterer.fit_knn(labeled_features, labeled_labels)

        # 2) propogate belief: get new labels on the whole unseen dataset via the knn
        knn_labels = clusterer.classify_new_data(unlabeled_features)

        accuracy = accuracy_score(all_labels[unlabeled_indices], knn_labels)
        class_report = classification_report(all_labels[unlabeled_indices], knn_labels)
        cm = confusion_matrix(all_labels[unlabeled_indices], knn_labels)
        print (f"accuracy: {accuracy}")
        # print (class_report)
        plot_confusion_matrix(cm, class_names=range(len(np.unique(knn_labels))))

        # 3) select new samples to label
        selected_indices = clusterer.select_samples_for_labeling(unlabeled_features, num_samples)

        # 4) get true labels for uncertain points

        # Simulate manual labeling by using the ground truth labels
        true_indices = [unlabeled_indices[idx] for idx in selected_indices]
        true_labels = all_labels[true_indices]

        # update sets of labelled
        # update labelled features with new ground truth for the sample images
        labeled_features = np.concatenate([labeled_features, all_embeddings[true_indices]], axis=0)
        labeled_labels = np.concatenate([labeled_labels, true_labels], axis=0)

        # update unlabelled now that we have seen a new sample
        unlabeled_indices = list(set(unlabeled_indices) - set(true_indices))
        unlabeled_features = all_embeddings[unlabeled_indices]

        # 5) visualize the clusters
        propagated_labels = clusterer.classify_new_data(unlabeled_features)
        clusterer.labels = np.concatenate([labeled_labels, propagated_labels], axis=0)

        clusterer.visualize_clusters(labeled_features, unlabeled_features, labeled_labels, propagated_labels, iteration)

        # todo: update self.labels with each iteration

        # Calculate and print the silhouette score
        silhouette_avg = silhouette_score(all_embeddings, clusterer.labels)
        # print(f"Silhouette Score: {silhouette_avg}")


if __name__ == "__main__":
    main()
