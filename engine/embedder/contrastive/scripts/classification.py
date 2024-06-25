from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

from engine.embedder.contrastive.contrastive_dataset import ContrastiveInternalDataset, ContrastiveDataset
from engine.embedder.contrastive.contrastive_model import XPrizeTreeEmbedder
from engine.embedder.contrastive.contrastive_train import infer_model
from engine.embedder.contrastive.contrastive_utils import FOREST_QPEB_MEAN, FOREST_QPEB_STD, contrastive_collate_fn


if __name__ == "__main__":
    source_data_root = Path('/home/hugo/Documents/xprize/data/FINAL_polygon_dataset_1536px_gr0p03')
    min_level = 'family'
    n_neighbors = 5  # Adjust this value based on your needs
    phylogenetic_tree_distances_path = '/home/hugo/Documents/xprize/data/pairs_with_dist.csv'
    metric = 'cosine' #'euclidean'
    # checkpoint = '/home/hugo/Documents/xprize/trainings/contrastive_resnet50_256_1024_144_mpt_1719085279/checkpoint_7.pth'
    checkpoint = '/home/hugo/Documents/xprize/training_alliance_canada/min_genus/checkpoint_27.pth'
    # checkpoint = '/home/hugo/Documents/xprize/training_alliance_canada/min_family/checkpoint_29.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 768
    model = XPrizeTreeEmbedder(
        resnet_model='resnet101',
        final_embedding_size=1024,
        dropout=0.5
    ).to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    brazil_date_pattern = r'^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})'
    equator_date_pattern = r'^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})'
    panama_date_pattern = r'^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})|bci_50ha_(?P<year2>\d{4})_(?P<month2>\d{2})_(?P<day2>\d{2})_'
    quebec_date_pattern = r'^(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_'

    train_dataset_panama = ContrastiveInternalDataset(
        fold='train',
        root_path=source_data_root / 'panama',
        date_pattern=panama_date_pattern
    )
    train_dataset_quebec = ContrastiveInternalDataset(
        fold='train',
        root_path=source_data_root / 'quebec_trees',
        date_pattern=quebec_date_pattern
    )
    train_dataset_brazil = ContrastiveInternalDataset(
        fold='train',
        root_path=source_data_root / 'brazil_zf2',
        date_pattern=brazil_date_pattern
    )
    train_dataset_equator = ContrastiveInternalDataset(
        fold='train',
        root_path=source_data_root / 'equator',
        date_pattern=equator_date_pattern
    )

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

    train_dataset = ContrastiveDataset(
        dataset_config={
                        # 'brazil': train_dataset_brazil,
                        # 'equator': train_dataset_equator,
                        'panama': train_dataset_panama,
                        # 'quebec': train_dataset_quebec
        },
        min_level=min_level,
        image_size=image_size,
        transform=None,
        normalize=True,
        mean=FOREST_QPEB_MEAN,
        std=FOREST_QPEB_STD,
        taxa_distances_df=pd.read_csv(phylogenetic_tree_distances_path)
    )

    test_dataset = ContrastiveDataset(
        dataset_config={
                        # 'brazil': test_dataset_brazil,
                        # 'equator': test_dataset_equator,
                        'panama': test_dataset_panama,
                        # 'quebec': test_dataset_quebec
        },
        min_level=min_level,
        image_size=image_size,
        transform=None,
        normalize=True,
        mean=FOREST_QPEB_MEAN,
        std=FOREST_QPEB_STD,
        taxa_distances_df=pd.read_csv(phylogenetic_tree_distances_path)
    )

    with torch.no_grad():
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=contrastive_collate_fn)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=contrastive_collate_fn)

        train_labels, train_labels_ids, train_embeddings = infer_model(model, train_loader, device,
                                                                       use_multi_gpu=False,
                                                                       use_mixed_precision=False)
        test_labels, test_labels_ids, test_embeddings = infer_model(model, test_loader, device,
                                                                    use_multi_gpu=False,
                                                                    use_mixed_precision=False)

    train_embeddings_np = train_embeddings.cpu().numpy() if isinstance(train_embeddings, torch.Tensor) else np.array(train_embeddings)
    test_embeddings_np = test_embeddings.cpu().numpy() if isinstance(test_embeddings, torch.Tensor) else np.array(test_embeddings)

    # train a k neighbors classifier
    k_neighbors = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    k_neighbors.fit(train_embeddings_np, train_labels_ids)
    test_predictions = k_neighbors.predict(test_embeddings_np)

    print(classification_report(test_labels_ids, test_predictions))




