import pandas as pd

from config.config_parsers.clusterer_parsers import ClustererInferConfig, ClustererInferIOConfig
from engine.clusterer.clusterer import Clusterer


def clusterer_main(config: ClustererInferConfig, dataset: pd.DataFrame):
    assert len(dataset) > 0
    assert 'embeddings' in dataset
    assert 'labels' in dataset

    clusterer = Clusterer(
        clustering_algorithm=config.clustering_algorithm,
        clustering_algorithm_params=config.clustering_algorithm_params
    )

    preds = clusterer.fit_predict(dataset=dataset)

    output_df = dataset.copy()
    output_df.drop('embeddings')
    output_df['cluster'] = preds

    return output_df


def clusterer_io_main(config: ClustererInferIOConfig):
    dataset = pd.read_csv



