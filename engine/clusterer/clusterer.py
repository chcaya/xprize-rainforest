import joblib
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from sklearn.cluster import KMeans, DBSCAN


class Clusterer:
    def __init__(self,
                 clustering_algorithm: str,
                 clustering_algorithm_params: dict):

        self.clustering_algorithm = clustering_algorithm

        if self.clustering_algorithm == 'kmeans':
            clusterer = KMeans
        else:
            raise ValueError(f"Clusterer algorithm '{self.clustering_algorithm}' hasn't been added yet.")

        self.clusterer = clusterer(**clustering_algorithm_params)

    def fit_predict(self, dataset: pd.DataFrame):
        X = np.ndarray(dataset['embeddings'])
        y = np.ndarray(dataset['labels'])

        preds = self.clusterer.fit_predict(X, y)

        return preds
