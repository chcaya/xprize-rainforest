import argparse
import os
import time
from pathlib import Path
import albumentations as A
import numpy as np
import pandas as pd
import torch

import yaml
from pytorch_metric_learning import losses, distances, reducers, miners
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.samplers import MPerClassSampler, FixedSetOfTriplets
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from engine.embedder.contrastive.contrastive_dataset import ContrastiveInternalDataset, ContrastiveDataset
from engine.embedder.contrastive.contrastive_model import XPrizeTreeEmbedder, XPrizeTreeEmbedder2
from engine.embedder.contrastive.contrastive_utils import FOREST_QPEB_MEAN, FOREST_QPEB_STD, save_model, \
    contrastive_collate_fn
from engine.embedder.transforms import embedder_transforms


def infer_model(model, dataloader, device, use_mixed_precision, use_multi_gpu, desc='Infering...', as_numpy=True):
    all_labels = []
    all_labels_ids = []
    all_families = []
    all_families_ids = []
    all_embeddings = []
    all_predicted_families = []

    for images, months, days, labels_ids, labels, families_ids, families in tqdm(dataloader, total=len(dataloader), desc=desc):
        data = torch.Tensor(images).to(device)
        months = torch.Tensor(months).to(device)
        days = torch.Tensor(days).to(device)
        if len(data.shape) == 3:
            data = data.unsqueeze(0)

        if use_mixed_precision:
            with torch.cuda.amp.autocast():
                output = model(data, months, days)
        else:
            output = model(data, months, days)

        if isinstance(model, nn.DataParallel):
            actual_model = model.module
        else:
            actual_model = model

        if isinstance(actual_model, XPrizeTreeEmbedder):
            embeddings = output
            predicted_families = [None] * len(labels)
        elif isinstance(actual_model, XPrizeTreeEmbedder2):
            embeddings, classifier_logits = output[0], output[1]
            predicted_families_ids = torch.argmax(classifier_logits, dim=1)
            predicted_families = [model.ids_to_families_mapping[int(family_id)] for family_id in predicted_families_ids]
        else:
            raise ValueError(f'Unknown model type: {actual_model.__class__}')

        all_labels.append(labels)
        all_labels_ids.append(labels_ids.cpu())
        all_families.append(families)
        all_families_ids.append(families_ids.cpu())
        all_embeddings.append(embeddings.detach().cpu())
        all_predicted_families.append(predicted_families)

    final_labels = sum(all_labels, [])
    final_labels_ids = torch.cat(all_labels_ids, dim=0)
    final_families = sum(all_families, [])
    final_families_ids = torch.cat(all_families_ids, dim=0)
    final_embeddings = torch.cat(all_embeddings, dim=0)
    final_predicted_families = sum(all_predicted_families, [])

    if as_numpy:
        final_labels = np.array(final_labels)
        final_embeddings = final_embeddings.numpy()
        final_labels_ids = final_labels_ids.numpy()

    return final_labels, final_labels_ids, final_families, final_families_ids, final_embeddings, final_predicted_families


def train(model: XPrizeTreeEmbedder or XPrizeTreeEmbedder2,
          train_dataset: ContrastiveDataset,
          valid_dataset: ContrastiveDataset,
          valid_train_dataset_for_classification: ContrastiveDataset,
          valid_valid_dataset_for_classification: ContrastiveDataset,
          train_batch_size: int,
          valid_batch_size: int,
          loss_weight_triplet: float,
          loss_weight_classification: float,
          use_multi_gpu: bool,
          distance: str,
          mining_func: miners.BaseMiner,
          criterion_metric: losses.BaseMetricLossFunction,
          criterion_classification: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          n_grad_accumulation_steps: int,
          writer: SummaryWriter,
          output_dir: Path,
          num_epochs: int,
          data_loader_num_workers: int):

    if use_multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    scaler = torch.cuda.amp.GradScaler()  # Initialize the GradScaler for mixed precision training

    sampler = MPerClassSampler(train_dataset.all_samples_labels, m=4, batch_size=train_batch_size, length_before_new_iter=len(train_dataset))
    data_loader = DataLoader(train_dataset,
                             batch_size=train_batch_size,
                             collate_fn=contrastive_collate_fn,
                             pin_memory=use_multi_gpu,
                             num_workers=data_loader_num_workers,
                             sampler=sampler)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        loss_since_last_log = 0
        loss_classification_since_last_log = 0
        loss_triplet_since_last_log = 0
        step_since_last_log = 0
        # re-instantiate the dataloader at the start of each epoch as the sampling was re-generated at the end of every epoch
        overall_step = epoch * len(data_loader) // n_grad_accumulation_steps
        accumulated_steps = 0

        for data in tqdm(data_loader, desc=f'Epoch {epoch}...'):
            imgs, months, days, labels_ids, labels, families_ids, families = data
            imgs, labels_ids, families_ids = imgs.to(device), labels_ids.to(device), families_ids.to(device)
            months, days = months.to(device), days.to(device),

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # Enable autocasting for mixed precision
                if isinstance(model, nn.DataParallel):
                    actual_model = model.module
                else:
                    actual_model = model

                if isinstance(actual_model, XPrizeTreeEmbedder):
                    embeddings = model(imgs, months, days)
                    indices_tuple = mining_func(embeddings, labels_ids)
                    loss = criterion_metric(embeddings=embeddings, labels=labels_ids, indices_tuple=indices_tuple)
                elif isinstance(actual_model, XPrizeTreeEmbedder2):
                    embeddings, classifier_logits = model(imgs, months, days)
                    indices_tuple = mining_func(embeddings, labels_ids)
                    loss_triplet = criterion_metric(embeddings=embeddings, labels=labels_ids, indices_tuple=indices_tuple)
                    model_compatible_labels = torch.Tensor([model.families_to_id_mapping[family] for family in families]).long().to(device)
                    loss_classification = criterion_classification(classifier_logits, model_compatible_labels)
                    loss = loss_weight_triplet * loss_triplet + loss_weight_classification * loss_classification

                    loss_classification_since_last_log += loss_weight_classification * loss_classification.item()
                    loss_triplet_since_last_log += loss_weight_triplet * loss_triplet.item()
                else:
                    raise ValueError(f'Unknown model type: {actual_model.__class__}')

            scaler.scale(loss).backward()  # Backward pass with scaled loss

            total_loss += loss.item()
            loss_since_last_log += loss.item()

            step_since_last_log += 1
            accumulated_steps += 1

            if accumulated_steps == n_grad_accumulation_steps:
                scaler.step(optimizer)  # Update the model parameters with scaled optimizer step
                scaler.update()  # Update the scaler for next iteration
                optimizer.zero_grad()
                accumulated_steps = 0
                overall_step += 1

                if overall_step != 0 and overall_step % 10 == 0:
                    writer.add_scalar('Loss', loss_since_last_log / step_since_last_log, overall_step)
                    writer.add_scalar('Loss_Classification', loss_classification_since_last_log / step_since_last_log, overall_step)
                    writer.add_scalar('Loss_Triplet', loss_triplet_since_last_log / step_since_last_log, overall_step)
                    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], overall_step)
                    loss_since_last_log = 0
                    loss_classification_since_last_log = 0
                    loss_triplet_since_last_log = 0
                    step_since_last_log = 0

                if overall_step != 0 and overall_step % 50 == 0:
                    writer.flush()

        # Check if there are remaining accumulated gradients
        if accumulated_steps > 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        model.eval()
        validate_for_classification(
            model=model,
            valid_train_dataset_for_classification=valid_train_dataset_for_classification,
            valid_valid_dataset_for_classification=valid_valid_dataset_for_classification,
            valid_batch_size=valid_batch_size,
            distance=distance,
            epoch=epoch,
            overall_step=overall_step,
            writer=writer,
            data_loader_num_workers=data_loader_num_workers,
            use_multi_gpu=use_multi_gpu
        )

        valid_loss = validate_for_loss(
            model=model,
            valid_dataset=valid_dataset,
            valid_batch_size=valid_batch_size,
            criterion_metric=criterion_metric,
            epoch=epoch,
            overall_step=overall_step,
            writer=writer,
            data_loader_num_workers=data_loader_num_workers,
            use_multi_gpu=use_multi_gpu
        )

        checkpoint_output_file = os.path.join(output_dir, f'checkpoint_{epoch}.pth')
        save_model(model, checkpoint_output_file=checkpoint_output_file)
        scheduler.step()
        model.train()

        print(f'Epoch {epoch}, Train Loss: {total_loss / len(data_loader)}, Valid Loss: {valid_loss}')


def validate_for_classification(model,
                                valid_train_dataset_for_classification,
                                valid_valid_dataset_for_classification,
                                valid_batch_size,
                                distance,
                                overall_step,
                                epoch,
                                writer,
                                data_loader_num_workers,
                                use_multi_gpu):
    print(f'Validating for epoch {epoch}...')

    valid_train_loader_for_classification = torch.utils.data.DataLoader(valid_train_dataset_for_classification,
                                                                        batch_size=valid_batch_size, shuffle=False,
                                                                        num_workers=data_loader_num_workers,
                                                                        collate_fn=contrastive_collate_fn)
    valid_valid_loader_for_classification = torch.utils.data.DataLoader(valid_valid_dataset_for_classification,
                                                                        batch_size=valid_batch_size, shuffle=False,
                                                                        num_workers=data_loader_num_workers,
                                                                        collate_fn=contrastive_collate_fn)

    model.eval()
    with torch.no_grad():
        train_labels, train_labels_ids, train_families, train_families_ids, train_embeddings, train_predicted_families = infer_model(model, valid_train_loader_for_classification, device, use_multi_gpu=use_multi_gpu, use_mixed_precision=False, desc='Inferring train samples...')
        valid_labels, valid_labels_ids, valid_families, valid_families_ids, valid_embeddings, valid_predicted_families = infer_model(model, valid_valid_loader_for_classification, device, use_multi_gpu=use_multi_gpu, use_mixed_precision=False, desc='Inferring valid samples...')

        train_to_delete = train_labels == -1
        valid_to_delete = valid_labels == -1

        train_embeddings = train_embeddings[~train_to_delete]
        train_labels = train_labels[~train_to_delete]
        valid_embeddings = valid_embeddings[~valid_to_delete]
        valid_labels = valid_labels[~valid_to_delete]

        # Standardize the embeddings
        scaler_standard = StandardScaler()
        X_train = scaler_standard.fit_transform(train_embeddings)
        X_test = scaler_standard.transform(valid_embeddings)

        # Train the SVC
        k_neighbors = KNeighborsClassifier(n_neighbors=5, metric=distance)
        k_neighbors.fit(X_train, train_labels)
        predictions = k_neighbors.predict(X_test)

        accuracy = accuracy_score(valid_labels, predictions)
        metrics = precision_recall_fscore_support(valid_labels, predictions, average='macro')
        metrics_weighted = precision_recall_fscore_support(valid_labels, predictions, average='weighted')
        writer.add_scalar('Valid/KNN/Accuracy', accuracy, overall_step)
        writer.add_scalar('Valid/KNN/Macro_Precision', metrics[0], overall_step)
        writer.add_scalar('Valid/KNN/Macro_Recall', metrics[1], overall_step)
        writer.add_scalar('Valid/KNN/Macro_F1', metrics[2], overall_step)
        writer.add_scalar('Valid/KNN/Weighted_Precision', metrics_weighted[0], overall_step)
        writer.add_scalar('Valid/KNN/Weighted_Recall', metrics_weighted[1], overall_step)
        writer.add_scalar('Valid/KNN/Weighted_F1', metrics_weighted[2], overall_step)
        print(f'Validation Accuracy: KNN: {accuracy}')

        if isinstance(model, nn.DataParallel):
            actual_model = model.module
        else:
            actual_model = model

        if isinstance(actual_model, XPrizeTreeEmbedder2):
            accuracy_train = accuracy_score(train_families, train_predicted_families)
            accuracy_valid = accuracy_score(valid_families, valid_predicted_families)
            f1_macro = precision_recall_fscore_support(valid_families, valid_predicted_families, average='macro')[2]
            f1_weighted = precision_recall_fscore_support(valid_families, valid_predicted_families, average='weighted')[2]
            writer.add_scalar('Valid/Family_XPrizeTreeEmbedder2/Accuracy', accuracy_valid, overall_step)
            writer.add_scalar('Valid/Family_XPrizeTreeEmbedder2/Macro_F1', f1_macro, overall_step)
            writer.add_scalar('Valid/Family_XPrizeTreeEmbedder2/Weighted_F1', f1_weighted, overall_step)
            print(f'Validation Accuracy: Family_XPrizeTreeEmbedder2: {accuracy_valid}')
            print(f'Train Accuracy: Family_XPrizeTreeEmbedder2: {accuracy_train}')


def validate_for_loss(model,
                      valid_dataset,
                      valid_batch_size,
                      criterion_metric,
                      epoch,
                      overall_step,
                      writer,
                      data_loader_num_workers,
                      use_multi_gpu):
    print(f'Validating for epoch {epoch}...')

    # sampler = MPerClassSampler(valid_dataset.all_samples_labels, m=4, batch_size=valid_batch_size, length_before_new_iter=len(valid_dataset))
    sampler = FixedSetOfTriplets(valid_dataset.all_samples_labels, len(valid_dataset) * 10)
    triplets_tuples = (sampler.fixed_set_of_triplets[:, 0], sampler.fixed_set_of_triplets[:, 1], sampler.fixed_set_of_triplets[:, 2])
    valid_loader = DataLoader(valid_dataset,
                              shuffle=False,
                              batch_size=valid_batch_size,
                              collate_fn=contrastive_collate_fn,
                              pin_memory=use_multi_gpu,
                              num_workers=data_loader_num_workers,
                              )

    total_loss = 0

    model.eval()
    with torch.no_grad():
        valid_labels, valid_labels_ids, valid_families, valid_families_ids, valid_embeddings, valid_predicted_families = infer_model(model, valid_loader, device, use_multi_gpu=use_multi_gpu, use_mixed_precision=False, as_numpy=False)
        total_loss = criterion_metric(embeddings=valid_embeddings, labels=valid_labels_ids, indices_tuple=triplets_tuples)

        writer.add_scalar('Valid/Loss', total_loss / len(valid_loader), overall_step)

        return total_loss / len(valid_loader)


if __name__ == '__main__':
    print('====Contrastive training main====')
    parser = argparse.ArgumentParser(description="Script to train the Contrastive network for XPrize Rainforest.")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--data_root', type=str, required=False)
    args = parser.parse_args()

    yaml_config = yaml.safe_load(open(args.config_path, 'r'))

    source_data_root = Path(yaml_config['source_data_root'])
    if args.data_root:
        source_data_root = Path(args.data_root)

    use_datasets = yaml_config['use_datasets']
    min_level = yaml_config['min_level']
    distance = yaml_config['distance']
    quebec_classification_dates = yaml_config['quebec_classification_dates']
    resnet_model = yaml_config['resnet_model']
    use_multi_gpu = yaml_config['use_multi_gpu']
    start_from_checkpoint = yaml_config['start_from_checkpoint']
    learning_rate = yaml_config['learning_rate']
    train_batch_size = yaml_config['train_batch_size']
    n_grad_accumulation_steps = yaml_config['n_grad_accumulation_steps']
    valid_batch_size = train_batch_size * yaml_config['valid_batch_size_multiplier']
    image_size = yaml_config['image_size']
    loss_weight_triplet = yaml_config['loss_weight_triplet']
    loss_weight_classification = yaml_config['loss_weight_classification']
    triplet_margin = yaml_config['triplet_margin']
    triplet_type = yaml_config['triplet_type']
    dropout = yaml_config['dropout']
    final_embedding_size = yaml_config['final_embedding_size']
    scheduler_T = yaml_config['scheduler_T']
    num_epochs = yaml_config['num_epochs']
    data_loader_num_workers = yaml_config['data_loader_num_workers']
    phylogenetic_tree_distances_path = yaml_config['phylogenetic_tree_distances_path']
    output_folder_root = Path(yaml_config['output_folder_root'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_model_name = f'contrastive_{resnet_model}_{image_size}_{final_embedding_size}_{train_batch_size * n_grad_accumulation_steps}_{min_level}'

    # Loading datasets
    print('Loading datasets...')
    brazil_date_pattern = r'^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})'
    equator_date_pattern = r'^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})'
    panama_date_pattern = r'^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})|bci_50ha_(?P<year2>\d{4})_(?P<month2>\d{2})_(?P<day2>\d{2})_'
    quebec_date_pattern = r'^(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_'

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

    valid_dataset_brazil = ContrastiveInternalDataset(
        fold='valid',
        root_path=source_data_root / 'brazil_zf2',
        date_pattern=brazil_date_pattern
    )
    valid_dataset_equator = ContrastiveInternalDataset(
        fold='valid',
        root_path=source_data_root / 'equator',
        date_pattern=equator_date_pattern
    )
    valid_dataset_panama = ContrastiveInternalDataset(
        fold='valid',
        root_path=source_data_root / 'panama',
        date_pattern=panama_date_pattern
    )
    valid_dataset_quebec = ContrastiveInternalDataset(
        fold='valid',
        root_path=source_data_root / 'quebec_trees',
        date_pattern=quebec_date_pattern
    )

    train_dataset_config = {
        'brazil': train_dataset_brazil,
        'equator': train_dataset_equator,
        'panama': train_dataset_panama,
        'quebec': train_dataset_quebec
    }
    valid_dataset_config = {
        'brazil': train_dataset_brazil,
        'equator': train_dataset_equator,
        'panama': valid_dataset_panama,
        'quebec': valid_dataset_quebec
    }

    if 'brazil' not in use_datasets:
        del train_dataset_config['brazil']
        del valid_dataset_config['brazil']
    if 'equator' not in use_datasets:
        del train_dataset_config['equator']
        del valid_dataset_config['equator']
    if 'panama' not in use_datasets:
        del train_dataset_config['panama']
        del valid_dataset_config['panama']
    if 'quebec' not in use_datasets:
        del train_dataset_config['quebec']
        del valid_dataset_config['quebec']

    siamese_sampler_dataset_train = ContrastiveDataset(
        dataset_config=train_dataset_config,
        min_level=min_level,
        image_size=image_size,
        transform=A.Compose(embedder_transforms),
        normalize=True,
        mean=FOREST_QPEB_MEAN,
        std=FOREST_QPEB_STD,
        taxa_distances_df=pd.read_csv(phylogenetic_tree_distances_path)
    )

    siamese_sampler_dataset_valid = ContrastiveDataset(
        dataset_config=valid_dataset_config,
        min_level=min_level,
        image_size=image_size,
        transform=None,
        normalize=True,
        mean=FOREST_QPEB_MEAN,
        std=FOREST_QPEB_STD,
        taxa_distances_df=pd.read_csv(phylogenetic_tree_distances_path)
    )
    siamese_sampler_dataset_valid.shuffle(seed=0)       # shuffling once to make sure species labels are well mixed for mini-batches

    # Preparing the Quebec datasets for classification validation
    quebec_detection_root = source_data_root / 'quebec_trees/from_detector'
    quebec_folders = [x for x in quebec_detection_root.iterdir() if x.is_dir()]
    if '2021_09_02' in quebec_classification_dates:
        quebec_folders_keep = [source_data_root / 'quebec_trees/from_annotations']
        quebec_classification_dates = [x for x in quebec_classification_dates if x != '2021_09_02']
    else:
        quebec_folders_keep = []
    quebec_folders_keep += [x for x in quebec_folders if any(date in x.name for date in quebec_classification_dates)]

    valid_train_dataset_quebec = ContrastiveInternalDataset(
        fold='train',
        root_path=quebec_folders_keep,
        date_pattern=quebec_date_pattern
    )
    valid_valid_dataset_quebec = ContrastiveInternalDataset(
        fold='valid',
        root_path=quebec_folders_keep,
        date_pattern=quebec_date_pattern
    )

    valid_train_dataset_quebec_for_classification = ContrastiveDataset(
        dataset_config={'quebec': valid_train_dataset_quebec},
        min_level=min_level,
        image_size=image_size,
        transform=A.Compose(embedder_transforms),
        normalize=True,
        mean=FOREST_QPEB_MEAN,
        std=FOREST_QPEB_STD,
        taxa_distances_df=pd.read_csv(phylogenetic_tree_distances_path)
    )

    valid_valid_dataset_quebec_for_classification = ContrastiveDataset(
        dataset_config={'quebec': valid_valid_dataset_quebec},
        min_level=min_level,
        image_size=image_size,
        transform=None,
        normalize=True,
        mean=FOREST_QPEB_MEAN,
        std=FOREST_QPEB_STD,
        taxa_distances_df=pd.read_csv(phylogenetic_tree_distances_path)
    )

    print('Datasets loaded successfully.')

    model = XPrizeTreeEmbedder2(
        resnet_model=resnet_model,
        final_embedding_size=final_embedding_size,
        dropout=dropout,
        date_embedding_dim=32,
        families=list(siamese_sampler_dataset_train.families_set),
    ).to(device)

    if start_from_checkpoint:
        if isinstance(model, XPrizeTreeEmbedder):
            model.load_state_dict(torch.load(start_from_checkpoint))
        elif isinstance(model, XPrizeTreeEmbedder2):
            model = model.from_checkpoint(start_from_checkpoint)
        else:
            raise ValueError(f'Unknown model type: {model.__class__}')

    if distance == 'cosine':
        distance_f = distances.CosineSimilarity()
    elif distance == 'euclidean':
        distance_f = LpDistance(normalize_embeddings=True, p=2, power=1)
    else:
        raise ValueError(f'Unknown distance metric: {distance}')

    reducer = reducers.AvgNonZeroReducer()
    # criterion = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    criterion_metric = losses.TripletMarginLoss(margin=triplet_margin, distance=distance_f, reducer=reducer)
    criterion_classification = nn.CrossEntropyLoss()
    mining_func = miners.TripletMarginMiner(
        margin=triplet_margin, distance=distance_f, type_of_triplets=triplet_type
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_T)

    output_dir = output_folder_root / f'{output_model_name}_{int(time.time())}'
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(str(output_dir))

    with open(output_dir / 'contrastive_train_config.yaml', 'w') as config_file:
        yaml.safe_dump(yaml_config, config_file)

    train(
        model=model,
        train_dataset=siamese_sampler_dataset_train,
        valid_dataset=siamese_sampler_dataset_valid,
        valid_train_dataset_for_classification=valid_train_dataset_quebec_for_classification,
        valid_valid_dataset_for_classification=valid_valid_dataset_quebec_for_classification,
        train_batch_size=train_batch_size,
        valid_batch_size=valid_batch_size,
        loss_weight_triplet=loss_weight_triplet,
        loss_weight_classification=loss_weight_classification,
        use_multi_gpu=use_multi_gpu,
        distance=distance,
        mining_func=mining_func,
        criterion_metric=criterion_metric,
        criterion_classification=criterion_classification,
        optimizer=optimizer,
        scheduler=scheduler,
        n_grad_accumulation_steps=n_grad_accumulation_steps,
        writer=writer,
        output_dir=output_dir,
        num_epochs=num_epochs,
        data_loader_num_workers=data_loader_num_workers
    )

