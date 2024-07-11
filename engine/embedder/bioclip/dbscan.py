import glob

import pandas as pd
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.cluster import DBSCAN, HDBSCAN
from utils.visualization import visualize_dbscan_output
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

def plot_pairwise_distance_histogram(X, bins=50):
    """
    Compute and plot the histogram of pairwise distances.

    Parameters:
    - X: np.ndarray, shape (n_samples, n_features)
        The input data.
    - bins: int
        Number of bins to use for the histogram.
    """
    # Compute pairwise distances
    distances = pairwise_distances(X)

    # Flatten the distance matrix and remove zero distances
    distances = distances[np.triu_indices_from(distances, k=1)]

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=bins, edgecolor='k')
    plt.title('Histogram of Pairwise Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':

    embeddings_filepaths = glob.glob("archive/embeddings_with_labels/embeddings_*")
    labels_filepaths = glob.glob("archive/embeddings_with_labels/labels_*")

    all_embeddings = []
    all_labels = []

    for embeddings_filepath, label_filepath in zip(embeddings_filepaths, labels_filepaths):
        if '_last' in embeddings_filepath or '_last' in label_filepath: continue

        embeddings = torch.load(embeddings_filepath)
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.cat(embeddings)
        labels = np.load(label_filepath)

        all_embeddings.append(embeddings)
        all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings)
    all_labels = np.concatenate(all_labels)

    embeddings_with_labels = pd.DataFrame({
        "embeddings": [e.numpy().tolist() for e in all_embeddings],
        "labels": all_labels,
    })

    embeddings_with_labels.to_csv('bioclip_embeddings_with_filepaths.csv')
    # reduce dims for dbscan
    print("reducing dims")
    scaler = StandardScaler()
    # tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    pca = PCA(n_components=5, random_state=42)
    reduced_embeddings = pca.fit_transform(all_embeddings)
    scaled_embeddings = scaler.fit_transform(reduced_embeddings)

    plot_pairwise_distance_histogram(scaled_embeddings, bins = 50)

    print ("running dbscan")
    for db_eps in [1, 1.5, 2, 2.5, 3]:
        for db_num_samp in [5, 10, 15, 20, 25, 30, 40]:
            db = DBSCAN(eps=db_eps, min_samples=db_num_samp, n_jobs=-1)
            print("running dbscan")
            dbscan_labels = db.fit_predict(scaled_embeddings)

            if len(set(dbscan_labels)) > 1:  # Avoid silhouette score if only one cluster

                visualize_dbscan_output(scaled_embeddings, dbscan_labels,
                                        title_suffix=f' eps: {db_eps}, num_samples: {db_num_samp}')

    for db_num_samp in [10, 15, 20, 30, 50, 100, 200]:

        hdbscan = HDBSCAN(min_cluster_size=20, min_samples=db_num_samp)

        hdbscan_labels = hdbscan.fit_predict(scaled_embeddings)
        if len(set(hdbscan_labels)) > 1:  # Avoid silhouette score if only one cluster

                    visualize_dbscan_output(scaled_embeddings, hdbscan_labels,
                                        title_suffix=f' HDBSCAN, num_samples: {db_num_samp}')
