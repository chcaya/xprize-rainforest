from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from engine.embedder.contrastive.contrastive_dataset import ContrastiveInternalDataset, ContrastiveDataset
from engine.embedder.contrastive.contrastive_model import XPrizeTreeEmbedder, XPrizeTreeEmbedder2, \
    XPrizeTreeEmbedder2NoDate
from engine.embedder.contrastive.contrastive_train import infer_model_with_labels
from engine.embedder.contrastive.contrastive_utils import FOREST_QPEB_MEAN, FOREST_QPEB_STD, contrastive_collate_fn

if __name__ == "__main__":
    # source_data_root = Path('/home/hugo/Documents/xprize/data/FINAL_polygon_dataset_1536px_gr0p03')
    source_data_root = Path('D:/XPrize/Data/pre_processed/FINAL_polygon_dataset_1536px_gr0p03')
    min_level = 'family'
    n_clusters = 15  # Adjust this value based on your needs
    # phylogenetic_tree_distances_path = '/home/hugo/Documents/xprize/data/pairs_with_dist.csv'
    phylogenetic_tree_distances_path = 'D:/XPrize/Data/phylogeny/pairs_with_dist.csv'
    metric = 'cosine' #'euclidean'
    # checkpoint = '/home/hugo/Documents/xprize/trainings/contrastive_resnet50_256_1024_144_mpt_1719085279/checkpoint_7.pth'
    # checkpoint = '/home/hugo/Documents/xprize/trainings_2/contrastive_resnet50_768_1024_64_genus_1719470300/checkpoint_37_62700.pth'
    checkpoint = 'D:/XPrize/models/embedder_quebec_equator_brazil/checkpoint_37_62700.pth'
    # checkpoint = '/home/hugo/Documents/xprize/training_alliance_canada/min_family/checkpoint_29.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 768
    # model = XPrizeTreeEmbedder2(
    #     resnet_model='resnet50',
    #     final_embedding_size=1024,
    #     dropout=0.5
    # ).to(device)
    model = XPrizeTreeEmbedder2NoDate.from_checkpoint(checkpoint).to(device)
    # model.load_state_dict(torch.load(checkpoint))
    model.eval()

    brazil_date_pattern = r'^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})'
    equator_date_pattern = r'^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})'
    panama_date_pattern = r'^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})|bci_50ha_(?P<year2>\d{4})_(?P<month2>\d{2})_(?P<day2>\d{2})_'
    quebec_date_pattern = r'^(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_'

    # test data
    test_dataset_panama = ContrastiveInternalDataset(
        fold='test',
        root_path=source_data_root / 'panama',
        date_pattern=panama_date_pattern
    )
    test_dataset_quebec = ContrastiveInternalDataset(
        fold='test',
        root_path=source_data_root / 'quebec_trees',
        date_pattern=quebec_date_pattern
    )
    test_dataset_brazil = ContrastiveInternalDataset(
        fold='test',
        root_path=source_data_root / 'brazil_zf2',
        date_pattern=brazil_date_pattern
    )
    test_dataset_equator = ContrastiveInternalDataset(
        fold='test',
        root_path=source_data_root / 'equator',
        date_pattern=equator_date_pattern
    )

    # valid data
    test_dataset_brazil_valid = ContrastiveInternalDataset(
        fold='valid',
        root_path=source_data_root / 'brazil_zf2',
        date_pattern=brazil_date_pattern
    )
    test_dataset_equator_valid = ContrastiveInternalDataset(
        fold='valid',
        root_path=source_data_root / 'equator',
        date_pattern=equator_date_pattern
    )

    dataset = ContrastiveDataset(
        dataset_config={
                        # 'brazil': test_dataset_brazil,
                        'equator': test_dataset_equator,
                        # 'panama': test_dataset_panama,
                        # 'quebec': test_dataset_quebec,
                        # 'brazil_valid': test_dataset_brazil_valid,
                        'equator_valid': test_dataset_equator_valid,
        },
        min_level=min_level,
        image_size=image_size,
        transform=None,
        normalize=True,
        mean=FOREST_QPEB_MEAN,
        std=FOREST_QPEB_STD,
        taxa_distances_df=pd.read_csv(phylogenetic_tree_distances_path),
        max_resampling_times=0
    )

    with torch.no_grad():
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=contrastive_collate_fn)
        labels, labels_ids, families, families_ids, embeddings, predicted_families, _ = infer_model_with_labels(model, loader, device,
                                                                                        use_mixed_precision=False)

    embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else np.array(embeddings)

    df = pd.DataFrame({'embeddings': list(embeddings_np), 'labels': labels, 'family': families})
    df.to_csv('./embeddings_random.csv', index=False)

    # # Apply K-Means clustering
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # cluster_labels_1024 = kmeans.fit_predict(embeddings_np)

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, metric=metric)
    reduced_embeddings = tsne.fit_transform(embeddings_np)
    top_labels = list(set(labels))

    # # Compute UMAP
    # reducer = umap.UMAP(n_components=2, random_state=42)
    # reduced_embeddings = reducer.fit_transform(embeddings_np)
    # top_labels = list(set(labels))

    plt.figure(figsize=(27, 12))
    # eps_list = [0.03, 0.05, 0.08, 0.12, 0.15, 0.2]
    # eps_list = [1.5, 2, 2.5, 3, 3.5, 4]
    # min_samples_list = [5, 7, 10, 12, 15, 17, 20, 30, 40]
    eps_list = [2.2, 2.5, 3, 3.5]
    min_samples_list = [16, 17, 18]
    for i, eps in enumerate(eps_list):
        for j, min_samples in enumerate(min_samples_list):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(reduced_embeddings)
            print(i * len(min_samples_list) + j + 1, f'eps={eps}, min_samples={min_samples}, n_clusters={len(set(cluster_labels))}')
            plt.subplot(len(eps_list) + 1, len(min_samples_list), i * len(min_samples_list) + j + 1)
            cmap = plt.get_cmap('jet', n_clusters)

            for k in range(n_clusters):
                idx = cluster_labels == k
                plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], color=cmap(k), label=f'Cluster {k}', alpha=0.5)

            plt.colorbar(ticks=range(n_clusters), label='Cluster index', boundaries=np.arange(n_clusters + 1) - 0.5,
                         spacing='proportional')
            plt.clim(-0.5, n_clusters - 0.5)
            plt.title(f't-SNE eps={eps}, min_samples={min_samples}')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.legend(title="Cluster IDs")

    plt.subplot(len(eps_list) + 1, len(min_samples_list), i * len(min_samples_list) + j + 2)
    cmap = plt.get_cmap('jet', len(top_labels))  # Get a colormap with as many colors as top labels

    for i, label in enumerate(top_labels):
        idx = labels == label
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], color=cmap(i), label=f'{label}', alpha=0.5)

    plt.colorbar(ticks=range(len(top_labels)), label='Label index', boundaries=np.arange(len(top_labels) + 1) - 0.5,
                 spacing='proportional')
    plt.clim(-0.5, len(top_labels) - 0.5)
    plt.title('t-SNE of Embeddings with True Labels')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title="True Labels")
    plt.show()

    plt.figure(figsize=(10, 10))

    cmap = plt.get_cmap('jet', len(top_labels))  # Get a colormap with as many colors as top labels

    for i, label in enumerate(top_labels):
        idx = labels == label
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], color=cmap(i), label=f'{label}', alpha=0.8)

    plt.colorbar(ticks=range(len(top_labels)), label='Label index', boundaries=np.arange(len(top_labels) + 1) - 0.5,
                 spacing='proportional')
    plt.clim(-0.5, len(top_labels) - 0.5)
    plt.title('t-SNE of Embeddings with True Labels')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title="True Labels")

    plt.show()


