from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from engine.embedder.contrastive.contrastive_dataset import ContrastiveInternalDataset, ContrastiveDataset, \
    ContrastiveInferDataset
from engine.embedder.contrastive.contrastive_model import XPrizeTreeEmbedder, XPrizeTreeEmbedder2, \
    XPrizeTreeEmbedder2NoDate
from engine.embedder.contrastive.contrastive_infer import infer_model_without_labels, infer_model_with_labels
from engine.embedder.contrastive.contrastive_utils import FOREST_QPEB_MEAN, FOREST_QPEB_STD, contrastive_collate_fn, \
    contrastive_infer_collate_fn

if __name__ == "__main__":
    infer_path = Path('C:/Users/Hugo/Documents/XPrize/infer/20240521_zf2100ha_highres_m3m_rgb/classifier_tilerizer_output/20240521_zf2100ha_highres_m3m_rgb')
    # source_data_root = Path('/media/hugobaudchon/4 TB/XPrize/Data/pre_processed/FINAL_polygon_dataset_1536px_gr0p03')
    source_data_root = Path('D:/XPrize/Data/pre_processed/FINAL_polygon_dataset_1536px_gr0p03')
    min_level = 'genus'
    n_clusters = 15  # Adjust this value based on your needs
    phylogenetic_tree_distances_path = 'D:/XPrize/Data/phylogeny/pairs_with_dist.csv'
    metric = 'cosine' #'euclidean'
    # checkpoint = '/home/hugo/Documents/xprize/trainings/contrastive_resnet50_256_1024_144_mpt_1719085279/checkpoint_7.pth'
    checkpoint = 'D:/XPrize/models/embedder_quebec_equator_brazil/checkpoint_37_62700.pth'
    # checkpoint = '/home/hugo/Documents/xprize/training_alliance_canada/min_family/checkpoint_29.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 768

    brazil_date_pattern = r'^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})'

    train_dataset_brazil = ContrastiveInternalDataset(
        fold='train',
        root_path=source_data_root / 'brazil_zf2',
        date_pattern=brazil_date_pattern
    )
    valid_dataset_brazil = ContrastiveInternalDataset(
        fold='valid',
        root_path=source_data_root / 'brazil_zf2',
        date_pattern=brazil_date_pattern
    )
    test_dataset_brazil = ContrastiveInternalDataset(
        fold='test',
        root_path=source_data_root / 'brazil_zf2',
        date_pattern=brazil_date_pattern
    )

    train_dataset_equator = ContrastiveInternalDataset(
        fold='train',
        root_path=source_data_root / 'equator',
        date_pattern=brazil_date_pattern
    )
    valid_dataset_equator = ContrastiveInternalDataset(
        fold='valid',
        root_path=source_data_root / 'equator',
        date_pattern=brazil_date_pattern
    )
    test_dataset_equator = ContrastiveInternalDataset(
        fold='test',
        root_path=source_data_root / 'equator',
        date_pattern=brazil_date_pattern
    )

    infer_dataset = ContrastiveInferDataset(
        fold='infer',
        root_path=infer_path,
        date_pattern=brazil_date_pattern,
        image_size=image_size,
        transform=None,
        normalize=True,
        mean=FOREST_QPEB_MEAN,
        std=FOREST_QPEB_STD
    )

    dataset = ContrastiveDataset(
        dataset_config={
            'equator_test': test_dataset_equator,
            'equator_valid': valid_dataset_equator,
            'equator_train': train_dataset_equator,
            'brazil_test': test_dataset_brazil,
            'brazil_valid': valid_dataset_brazil,
            'brazil_train': train_dataset_brazil
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

    infer_tiles_paths = [str(infer_dataset.tiles[idx]['path']) for idx in range(len(infer_dataset))]

    print(len(dataset))

    model = XPrizeTreeEmbedder2NoDate.from_checkpoint(checkpoint).to(device)
    # model.load_state_dict(torch.load(checkpoint))
    model.eval()

    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=contrastive_collate_fn)
        infer_loader = torch.utils.data.DataLoader(infer_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=contrastive_infer_collate_fn)

        test_labels, test_labels_ids, test_families, test_families_ids, test_embeddings, test_predicted_families, test_predicted_families_scores = infer_model_with_labels(
            model,
            test_loader,
            device,
            use_mixed_precision=False,
            as_numpy=True
        )

        print(test_predicted_families_scores.tolist())
        print(test_predicted_families_scores.shape)

        df_test = pd.DataFrame({'embeddings': test_embeddings.tolist(), 'labels': test_labels, 'family': test_families})
        df_test.to_csv('./embeddings_equator_brazil.csv', index=False)
        df_test.to_pickle('./embeddings_equator_brazil.pkl')

        infer_embeddings, infer_predicted_families, infer_predicted_families_scores = infer_model_without_labels(
            model,
            infer_loader,
            device,
            use_mixed_precision=False,
            as_numpy=True
        )
        infer_labels = [None] * len(infer_embeddings)

        df_infer = pd.DataFrame({'embeddings': infer_embeddings.tolist(), 'labels': None, 'family': None,
                                 'predicted_family': infer_predicted_families,
                                 'predicted_family_scores': infer_predicted_families_scores.tolist(),
                                 'tile_path': infer_tiles_paths})
        df_infer.to_csv('./embeddings_100ha.csv', index=False)
        df_infer.to_pickle('./embeddings_100ha.pkl')

    embeddings_np = np.concatenate((test_embeddings, infer_embeddings), axis=0)
    labels = np.concatenate((test_labels, infer_labels), axis=0)

    indices = np.arange(embeddings_np.shape[0])
    np.random.shuffle(indices)
    shuffled_embeddings = embeddings_np[indices]
    shuffled_labels = labels[indices]

    # # Apply K-Means clustering
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # cluster_labels_1024 = kmeans.fit_predict(embeddings_np)

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, metric=metric)
    reduced_embeddings = tsne.fit_transform(shuffled_embeddings)
    top_labels = list(set(shuffled_labels))

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
        idx = shuffled_labels == label
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
