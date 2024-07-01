import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from engine.embedder.bioclip.bioclip_model import BioCLIPModel
from engine.embedder.bioclip.data_init import data_loader_init_main
from engine.embedder.bioclip.utils.config_utils import load_config
from engine.embedder.bioclip.utils.visualization import plot_confusion_matrix, visualize_clusters, visualize_dbscan_output

class BioClipActiveLearner:
    def __init__(self,
                 model: BioCLIPModel,
                 dbscan_eps: float =0.2,
                 dbscan_min_samples: int =5,
                 uncertainty_threshold: float = 0.3,
                 rf_estimators=100):
        self.model = model
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.uncertainty_threshold = uncertainty_threshold # todo: can this be used better
        self.rf_estimators = rf_estimators
        self.scaler = StandardScaler()

        self.db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        self.rf = RandomForestClassifier(n_estimators=rf_estimators, random_state=42)
        self.tsne = TSNE(n_components=2)

        self.labels = None
        self.classifier_labels = None
        self.labeled_data = []
        self.labeled_labels = []

    def extract_features(self, image_tensors):
        return self.model.generate_embeddings(image_tensors)

    def reduce_dimensions(self, embeddings):
        return self.tsne.fit_transform(embeddings)

    def cluster_data(self, features):
        features = self.scaler.fit_transform(features)
        self.labels = self.db.fit_predict(features)
        return self.labels

    def train_rf_classifier(self, features, labels):
        self.rf.fit(features, labels)
        self.classifier_labels = labels

    def propagate_labels(self, features):
        return self.rf.predict(features)

    def uncertainty_estimation(self, embeddings):
        proba = self.rf.predict_proba(embeddings)
        uncertainties = 1 - np.max(proba, axis=1)
        return uncertainties

    def select_samples_for_labeling(self, embeddings, num_samples, method='rf'):
        uncertainties = self.uncertainty_estimation(embeddings)
        selected_indices = np.argsort(uncertainties)[-num_samples:]
        return selected_indices


    @staticmethod
    def most_frequent_element(array):
        unique_elements, counts = np.unique(array, return_counts=True)
        max_index = counts.argmax()
        return unique_elements[max_index]


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


def main():
    config = load_config(config_path='config.yaml')
    data_loader = data_loader_init_main('config.yaml')

    model = BioCLIPModel(config['training']['model_name'], config['training']['pretrained_path'])
    active_learner = BioClipActiveLearner(model)

    # Extract features using the model
    all_embeddings, all_labels = [], []
    for idx, (image_tensors, labels) in enumerate(data_loader):
        features = active_learner.extract_features(image_tensors)
        all_embeddings.append(features)
        all_labels.extend(labels)

    all_embeddings = torch.cat(all_embeddings)
    all_labels = np.array(all_labels)

    # reduce dims for dbscan
    all_embeddings_reduced = active_learner.reduce_dimensions(all_embeddings)
    dbscan_labels = active_learner.cluster_data(all_embeddings_reduced)

    if config['active_learner']['visualize_dbscan']:
        visualize_dbscan_output(all_embeddings, dbscan_labels)

    # todo: find a better way than this hack
    # reset scaler back to correct dims
    active_learner.scaler.fit(all_embeddings)

    # Align DBSCAN clusters with true labels by sampling
    sample_points_per_cluster = config['active_learner']['sample_dbscan_points_per_cluster']
    aligned_labels, sampled_idx = active_learner.align_clusters_with_labels(dbscan_labels, all_labels,
                                                                            num_samples=sample_points_per_cluster)
    # Use aligned labels to train k-NN
    dbscan_indices = np.where(dbscan_labels != -1)[0]  # Exclude noise points
    dbscan_features = all_embeddings[dbscan_indices]
    dbscan_labels_filtered = aligned_labels[dbscan_indices]
    active_learner.train_rf_classifier(dbscan_features, dbscan_labels_filtered)

    # Set initial labeled and unlabeled data
    labeled_features = all_embeddings[sampled_idx]
    labeled_labels = all_labels[sampled_idx]
    unlabeled_indices = list(set(range(len(all_embeddings))) - set(sampled_idx))
    unlabeled_features = all_embeddings[unlabeled_indices]

    silhouette_avg = silhouette_score(all_embeddings, dbscan_labels)
    print(f"Silhouette Score DBSCAN: {silhouette_avg}")

    # todo: move choice of model to config

    for iteration in range(config['active_learner']['max_iterations']):
        # 1) fit a classifer(RF) on the labelled examples we know
        active_learner.train_rf_classifier(labeled_features, labeled_labels)

        # 2) propogate belief: get new labels on the whole unseen dataset via the knn
        classifier_labels = active_learner.propagate_labels(unlabeled_features)

        # eval classifier
        accuracy = accuracy_score(all_labels[unlabeled_indices], classifier_labels)
        print(f"accuracy: {accuracy}")

        cm = confusion_matrix(all_labels[unlabeled_indices], classifier_labels)
        plot_confusion_matrix(cm, class_names=range(len(np.unique(classifier_labels))))

        # 3) select new samples to label
        num_samples = config['active_learner']['sample_size']
        selected_indices = active_learner.select_samples_for_labeling(unlabeled_features, num_samples)

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
        propagated_labels = active_learner.propagate_labels(unlabeled_features)
        active_learner.labels = np.concatenate([labeled_labels, propagated_labels], axis=0)

        if config['active_learner']['visualize_clusters']:
            visualize_clusters(labeled_features, unlabeled_features, labeled_labels, propagated_labels, iteration)


        # Calculate and print the silhouette score
        silhouette_avg = silhouette_score(all_embeddings, active_learner.labels)
        print(f"Silhouette Score: {silhouette_avg}")


if __name__ == "__main__":
    main()
