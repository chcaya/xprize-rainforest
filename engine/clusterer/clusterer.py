import itertools
import os

import numpy as np
import pandas as pd
import umap
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


class Clusterer:
    def __init__(self,
                 embeddings_df: pd.DataFrame,
                 embeddings_column_name: str,
                 scale_embeddings: bool,
                 metric: str,
                 reduce_algo_name1: str or None,
                 reduce_algo_name2: str or None,
                 visualize_algo_name: str,
                 use_reduced1_for_visualization: bool,
                 tsne_perplexity: int,
                 umap_n_neighbors: int,
                 umap_min_dist: float,
                 min_cluster_size: int,
                 n_components1: int or None,
                 n_components2: int or None,
                 output_root_dir: str,
                 n_cpus: int = 10):

        self.embeddings_df = embeddings_df
        self.embeddings_column_name = embeddings_column_name

        self.scale_embeddings = scale_embeddings
        self.metric = metric
        self.reduce_algo_name1 = reduce_algo_name1
        self.reduce_algo_name2 = reduce_algo_name2
        self.visualize_algo_name = visualize_algo_name
        self.use_reduced1_for_visualization = use_reduced1_for_visualization
        self.tsne_perplexity = tsne_perplexity
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.min_cluster_size = min_cluster_size
        self.n_components1 = n_components1
        self.n_components2 = n_components2
        self.output_root_dir = output_root_dir
        self.n_cpus = n_cpus

        assert self.metric in ['cosine', 'euclidean']

        self.reduce_algo_1 = self.get_reduce_algo(reduce_algo_name1, n_components1)
        self.reduce_algo_2 = self.get_reduce_algo(reduce_algo_name2, n_components2)

        if self.visualize_algo_name == 'tsne':
            self.visualize_algo = TSNE(n_components=2, random_state=42,
                                       metric=self.metric, perplexity=self.tsne_perplexity)
        elif self.visualize_algo_name == 'umap':
            self.visualize_algo = umap.UMAP(n_components=2, n_neighbors=self.umap_n_neighbors,
                                            min_dist=self.umap_min_dist, metric=self.metric, random_state=42)
        else:
            raise ValueError(f"Unknown visualization algorithm: {self.visualize_algo_name}")

        embeddings = np.array(self.embeddings_df[self.embeddings_column_name].to_list())

        if self.scale_embeddings:
            scaler = StandardScaler()
            embeddings = scaler.fit_transform(embeddings)

        if self.reduce_algo_1 is not None and self.reduce_algo_2 is not None:
            self.reduced_embeddings1 = self.reduce_algo_1.fit_transform(embeddings)
            self.reduced_embeddings = self.reduce_algo_2.fit_transform(self.reduced_embeddings1)
        elif self.reduce_algo_1 is not None:
            self.reduced_embeddings1 = self.reduce_algo_1.fit_transform(embeddings)
            self.reduced_embeddings = self.reduced_embeddings1
        else:
            self.reduced_embeddings1 = embeddings
            self.reduced_embeddings = embeddings

        if self.use_reduced1_for_visualization:
            self.visualize_embeddings = self.visualize_algo.fit_transform(self.reduced_embeddings1)
        else:
            self.visualize_embeddings = self.visualize_algo.fit_transform(embeddings)

    def get_reduce_algo(self, reduce_algo_name: str, n_components: int):
        if reduce_algo_name == 'tsne':
            reduce_algo = TSNE(n_components=n_components, random_state=42,
                               metric=self.metric, perplexity=self.tsne_perplexity)
        elif reduce_algo_name == 'umap':
            reduce_algo = umap.UMAP(n_components=n_components, n_neighbors=self.umap_n_neighbors,
                                    min_dist=self.umap_min_dist, metric=self.metric, random_state=42)
        elif reduce_algo_name == 'pca':
            reduce_algo = PCA(n_components=n_components)
        elif reduce_algo_name is None:
            reduce_algo = None
        else:
            raise ValueError(f"Unknown reduction algorithm: {reduce_algo_name}")

        return reduce_algo

    def search_best_params(self,
                           cluster_algo: str,
                           dbscan_eps_list: list,
                           dbscan_min_samples_list: list,
                           hdbscan_min_samples_list: list,
                           hdbscan_cluster_selection_epsilon_list: list,
                           hdbscan_cluster_selection_method_list: list):

        if cluster_algo == 'dbscan':
            combinations = itertools.product(dbscan_eps_list,
                                             dbscan_min_samples_list)
            for (eps, min_samples) in combinations:
                self._run_dbscan(eps=eps, min_samples=min_samples, display_results=True)
        elif cluster_algo == 'hdbscan':
            combinations = itertools.product(hdbscan_cluster_selection_epsilon_list,
                                             hdbscan_min_samples_list,
                                             hdbscan_cluster_selection_method_list)
            for (cluster_selection_epsilon, min_samples, cluster_selection_method) in combinations:
                self._run_hdbscan(min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon,
                                  cluster_selection_method=cluster_selection_method, display_results=True)
        else:
            raise ValueError(f"Unknown clustering algorithm: {cluster_algo}")

    def get_clusters(self,
                     cluster_algo: str,
                     dbscan_eps: float,
                     dbscan_min_samples: int,
                     hdbscan_min_samples: int,
                     hdbscan_cluster_selection_epsilon: float,
                     hdbscan_cluster_selection_method: str):
        if cluster_algo == 'dbscan':
            cluster_labels = self._run_dbscan(eps=dbscan_eps, min_samples=dbscan_min_samples, display_results=False)
        elif cluster_algo == 'hdbscan':
            cluster_labels = self._run_hdbscan(min_samples=hdbscan_min_samples,
                                               cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
                                               cluster_selection_method=hdbscan_cluster_selection_method,
                                               display_results=False)
        else:
            raise ValueError(f"Unknown clustering algorithm: {cluster_algo}")

        return cluster_labels

    def _get_output_name_prefix(self, cluster_algo: str):
        output_name_prefix = f"{cluster_algo}_{self.metric}_{self.reduce_algo_name1}_{self.n_components1}_{self.reduce_algo_name2}_{self.n_components2}_{self.visualize_algo_name}_{self.min_cluster_size}"
        return output_name_prefix

    def _run_dbscan(self,
                    eps: float,
                    min_samples: int,
                    display_results: bool):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=self.metric, n_jobs=self.n_cpus)
        cluster_labels = dbscan.fit_predict(self.reduced_embeddings)
        cluster_labels, n_clusters = self.parse_clusters(cluster_labels)
        print(f'eps={eps}, min_samples={min_samples},'
              f' n_samples_without_cluster={len([x for x in cluster_labels if x == -1])},'
              f' n_clusters (with over {self.min_cluster_size} samples)={n_clusters}')
        if display_results:
            output_string_prefix = self._get_output_name_prefix('dbscan')
            output_string_suffix = f"eps{str(eps).replace('.', 'p')}_minsamples{min_samples}"
            self.display_clusters(cluster_labels=cluster_labels, n_clusters=n_clusters,
                                  output_string_prefix=output_string_prefix, output_string_suffix=output_string_suffix)
        return cluster_labels

    def _run_hdbscan(self,
                     min_samples: int,
                     cluster_selection_epsilon: float,
                     cluster_selection_method: str,
                     display_results: bool):
        hdbscan = HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=min_samples,
                          cluster_selection_epsilon=cluster_selection_epsilon, metric=self.metric,
                          cluster_selection_method=cluster_selection_method, copy=True,
                          n_jobs=self.n_cpus)
        cluster_labels = hdbscan.fit_predict(self.reduced_embeddings)
        cluster_labels, n_clusters = self.parse_clusters(cluster_labels)
        print(f'cluster_selection_epsilon={cluster_selection_epsilon}, min_samples={min_samples},'
              f' method={cluster_selection_method},'
              f' n_samples_without_cluster={len([x for x in cluster_labels if x == -1])},'
              f' n_clusters (with over {self.min_cluster_size} samples)={n_clusters}')
        if display_results:
            output_string_prefix = self._get_output_name_prefix('hdbscan')
            output_string_suffix = f"method{cluster_selection_method}_clusterselectionepsilon{str(cluster_selection_epsilon).replace('.', 'p')}_minsamples{min_samples}"
            self.display_clusters(cluster_labels=cluster_labels, n_clusters=n_clusters,
                                  output_string_prefix=output_string_prefix, output_string_suffix=output_string_suffix)

        return cluster_labels

    def parse_clusters(self, cluster_labels: np.array):
        unique, counts = np.unique(cluster_labels, return_counts=True)
        cluster_counts = dict(zip(unique, counts))

        # Set the labels of small clusters to -1
        for cluster_id, count in cluster_counts.items():
            if count < self.min_cluster_size and cluster_id != -1:
                cluster_labels[cluster_labels == cluster_id] = -1

        large_unique, large_counts = np.unique(cluster_labels, return_counts=True)
        large_cluster_counts = dict(zip(large_unique, large_counts))

        # reordering clusters ids from [1...m] to [1...n] by decreasing size
        new_cluster_labels = cluster_labels.copy()
        for new_cluster_id, (cluster_label, _) in enumerate(
                sorted(large_cluster_counts.items(), key=lambda x: x[1], reverse=True)):
            if cluster_label == -1:
                continue
            new_cluster_labels[cluster_labels == cluster_label] = new_cluster_id + 1

        new_cluster_labels = np.array(new_cluster_labels)
        n_clusters = len(set(new_cluster_labels)) - (1 if -1 in new_cluster_labels else 0)
        return new_cluster_labels, n_clusters

    def display_clusters(self,
                         cluster_labels: np.array,
                         n_clusters: int,
                         output_string_prefix: str,
                         output_string_suffix: str):
        if n_clusters > 0:
            # Create a large figure
            plt.figure(figsize=(20, 20))
            cmap = plt.get_cmap('jet', n_clusters)

            color_indexes = np.linspace(0, 1, n_clusters)
            color_indexes = np.random.permutation(color_indexes)

            # Plot noise points
            if -1 in cluster_labels:
                idx = cluster_labels == -1
                plt.scatter(self.visualize_embeddings[idx, 0], self.visualize_embeddings[idx, 1], color='black',
                            label='Noise',
                            alpha=0.5)

            # Plot each cluster
            for color_i, k in enumerate(set(cluster_labels)):
                if k == -1:
                    continue
                idx = cluster_labels == k
                plt.scatter(self.visualize_embeddings[idx, 0], self.visualize_embeddings[idx, 1],
                            color=cmap(color_indexes[color_i]),
                            label=f'Cluster {k}',
                            alpha=0.5)

            # Add colorbar, titles, and labels
            plt.colorbar(ticks=range(n_clusters), label='Cluster index',
                         boundaries=np.arange(n_clusters + 1) - 0.5,
                         spacing='proportional')
            plt.clim(-0.5, n_clusters - 0.5)
            plt.title(f'{output_string_prefix}_{output_string_suffix}')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.legend(title="Cluster IDs")

            # Save the figure
            output_folder = os.path.join(self.output_root_dir, output_string_prefix)
            os.makedirs(output_folder, exist_ok=True)
            plt.savefig(os.path.join(output_folder, f"{output_string_prefix}_{output_string_suffix}.png"))
            plt.close()
