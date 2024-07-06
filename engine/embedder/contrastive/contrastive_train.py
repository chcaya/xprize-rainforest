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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

from engine.embedder.contrastive.contrastive_dataset import ContrastiveInternalDataset, ContrastiveDataset
from engine.embedder.contrastive.contrastive_infer import infer_model_with_labels
from engine.embedder.contrastive.contrastive_model import XPrizeTreeEmbedder, XPrizeTreeEmbedder2, \
    XPrizeTreeEmbedder2NoDate, DinoV2Embedder
from engine.embedder.contrastive.contrastive_utils import FOREST_QPEB_MEAN, FOREST_QPEB_STD, save_model, \
    contrastive_collate_fn
from engine.embedder.transforms import embedder_transforms_v2, embedder_simple_transforms_v2


def train(model: XPrizeTreeEmbedder or XPrizeTreeEmbedder2 or XPrizeTreeEmbedder2NoDate or DinoV2Embedder,
          train_dataloader: DataLoader,
          valid_train_dataloader: DataLoader,
          valid_dataloaders: dict[str, DataLoader],
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
          valid_knn_k: int):

    if use_multi_gpu:
        model = nn.DataParallel(model)
    model.to(device)

    model.eval()
    with torch.no_grad():
        validate(
            model=model,
            train_loader=valid_train_dataloader,
            valid_dataloaders=valid_dataloaders,
            distance=distance,
            overall_step=0,
            writer=writer,
            valid_knn_k=valid_knn_k
        )

    scaler = torch.cuda.amp.GradScaler()  # Initialize the GradScaler for mixed precision training

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        loss_since_last_log = 0
        loss_classification_since_last_log = 0
        loss_triplet_since_last_log = 0
        step_since_last_log = 0
        # re-instantiate the dataloader at the start of each epoch as the sampling was re-generated at the end of every epoch
        overall_step = epoch * len(train_dataloader) // n_grad_accumulation_steps
        accumulated_steps = 0

        for data in tqdm(train_dataloader, desc=f'Epoch {epoch}...'):
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
                    model_compatible_labels = torch.Tensor([actual_model.families_to_id_mapping[family] for family in families]).long().to(device)
                    loss_classification = criterion_classification(classifier_logits, model_compatible_labels)
                    loss = loss_weight_triplet * loss_triplet + loss_weight_classification * loss_classification
                    loss_classification_since_last_log += loss_weight_classification * loss_classification.item()
                    loss_triplet_since_last_log += loss_weight_triplet * loss_triplet.item()
                elif isinstance(actual_model, XPrizeTreeEmbedder2NoDate):
                    embeddings, classifier_logits = model(imgs)
                    indices_tuple = mining_func(embeddings, labels_ids)
                    loss_triplet = criterion_metric(embeddings=embeddings, labels=labels_ids, indices_tuple=indices_tuple)
                    model_compatible_labels = torch.Tensor([actual_model.families_to_id_mapping[family] for family in families]).long().to(device)
                    loss_classification = criterion_classification(classifier_logits, model_compatible_labels)
                    loss = loss_weight_triplet * loss_triplet + loss_weight_classification * loss_classification
                    loss_classification_since_last_log += loss_weight_classification * loss_classification.item()
                    loss_triplet_since_last_log += loss_weight_triplet * loss_triplet.item()
                elif isinstance(actual_model, DinoV2Embedder):
                    embeddings = model(imgs)
                    indices_tuple = mining_func(embeddings, labels_ids)
                    loss = criterion_metric(embeddings=embeddings, labels=labels_ids, indices_tuple=indices_tuple)
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
        with torch.no_grad():
            valid_loss = validate(
                model=model,
                train_loader=valid_train_dataloader,
                valid_dataloaders=valid_dataloaders,
                distance=distance,
                overall_step=overall_step,
                writer=writer,
                valid_knn_k=valid_knn_k
            )

        checkpoint_output_file = os.path.join(output_dir, f'checkpoint_{epoch}_{overall_step}.pth')
        save_model(model, checkpoint_output_file=checkpoint_output_file)
        scheduler.step()
        model.train()

        print(f'Epoch {epoch}, Train Loss: {total_loss / len(train_dataloader)}, Valid Loss: {valid_loss}')


def validate(model,
             train_loader,
             valid_dataloaders,
             distance,
             overall_step,
             writer,
             valid_knn_k):

    train_labels, train_labels_ids, train_families, train_families_ids, train_embeddings, train_predicted_families, _ = infer_model_with_labels(
        model, train_loader, device, use_mixed_precision=True,
        desc='Inferring train samples...')
    train_to_delete = train_labels == -1
    train_embeddings = train_embeddings[~train_to_delete]
    train_labels = train_labels[~train_to_delete]
    scaler_standard = StandardScaler()
    X_train = scaler_standard.fit_transform(train_embeddings)

    all_valid_embeddings = []
    all_valid_labels_ids = []
    total_knn_accuracy = 0
    total_knn_macro_f1 = 0
    total_knn_samples = 0

    total_classification_accuracy = 0
    total_classification_macro_f1 = 0
    total_classification_samples = 0

    for dataset_name, valid_dataloader in valid_dataloaders.items():
        valid_labels, valid_labels_ids, valid_families, valid_families_ids, valid_embeddings, valid_predicted_families, _ = infer_model_with_labels(model, valid_dataloader, device, use_mixed_precision=True, desc=f'Inferring valid samples for {dataset_name}...')
        valid_to_delete = valid_labels == -1
        valid_embeddings = valid_embeddings[~valid_to_delete]
        valid_labels = valid_labels[~valid_to_delete]
        X_test = scaler_standard.transform(valid_embeddings)
        k_neighbors = KNeighborsClassifier(n_neighbors=valid_knn_k, metric=distance)
        k_neighbors.fit(X_train, train_labels)
        predictions = k_neighbors.predict(X_test)

        all_valid_embeddings.append(valid_embeddings)
        all_valid_labels_ids.append(valid_labels_ids)

        accuracy = accuracy_score(valid_labels, predictions)
        metrics = precision_recall_fscore_support(valid_labels, predictions, average='macro')
        writer.add_scalar(f'Valid/{dataset_name}/KNN/Accuracy', accuracy, overall_step)
        writer.add_scalar(f'Valid/{dataset_name}/KNN/Macro_F1', metrics[2], overall_step)
        print(f'Validation Accuracy: KNN {dataset_name}: {accuracy}')

        total_knn_accuracy += accuracy * len(valid_labels)
        total_knn_macro_f1 += metrics[2] * len(valid_labels)
        total_knn_samples += len(valid_labels)

        if isinstance(model, nn.DataParallel):
            actual_model = model.module
        else:
            actual_model = model

        if isinstance(actual_model, (XPrizeTreeEmbedder2, XPrizeTreeEmbedder2NoDate)):
            accuracy_valid = accuracy_score(valid_families, valid_predicted_families)
            f1_macro = precision_recall_fscore_support(valid_families, valid_predicted_families, average='macro')[2]
            writer.add_scalar(f'Valid/{dataset_name}/Family_XPrizeTreeEmbedder2/Accuracy', accuracy_valid, overall_step)
            writer.add_scalar(f'Valid/{dataset_name}/Family_XPrizeTreeEmbedder2/Macro_F1', f1_macro, overall_step)
            print(f'Validation Accuracy: Family_XPrizeTreeEmbedder2 {dataset_name}: {accuracy_valid}')

            total_classification_accuracy += accuracy_valid * len(valid_families)
            total_classification_macro_f1 += f1_macro * len(valid_families)
            total_classification_samples += len(valid_families)

    accuracy_train = accuracy_score(train_families, train_predicted_families)
    writer.add_scalar(f'Train/Family_XPrizeTreeEmbedder2/Accuracy', accuracy_train, overall_step)
    print(f'Train Accuracy: Family_XPrizeTreeEmbedder2: {accuracy_train}')

    writer.add_scalar('Valid/KNN/Accuracy', total_knn_accuracy / total_knn_samples, overall_step)
    writer.add_scalar('Valid/KNN/Macro_F1', total_knn_macro_f1 / total_knn_samples, overall_step)
    writer.add_scalar('Valid/Family_XPrizeTreeEmbedder2/Accuracy', total_classification_accuracy / total_classification_samples, overall_step)
    writer.add_scalar('Valid/Family_XPrizeTreeEmbedder2/Macro_F1', total_classification_macro_f1 / total_classification_samples, overall_step)

    all_valid_embeddings = np.concatenate(all_valid_embeddings, axis=0)
    all_valid_labels_ids = np.concatenate(all_valid_labels_ids, axis=0)
    shuffled_indices = np.random.permutation(len(all_valid_labels_ids))
    all_valid_embeddings = all_valid_embeddings[shuffled_indices]
    all_valid_labels_ids = all_valid_labels_ids[shuffled_indices]

    sampler = FixedSetOfTriplets(all_valid_labels_ids, len(all_valid_labels_ids) * 10)
    triplets_tuples = (sampler.fixed_set_of_triplets[:, 0],
                       sampler.fixed_set_of_triplets[:, 1],
                       sampler.fixed_set_of_triplets[:, 2])
    valid_loss = criterion_metric(embeddings=torch.Tensor(all_valid_embeddings).to(device), labels=torch.Tensor(all_valid_labels_ids).to(device), indices_tuple=triplets_tuples)
    final_valid_loss = valid_loss / triplets_tuples[0].shape[0]
    writer.add_scalar('Valid/Loss_Triplet', final_valid_loss, overall_step)

    return final_valid_loss


def validate_for_classification(model,
                                train_dataset,
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
        train_labels, train_labels_ids, train_families, train_families_ids, train_embeddings, train_predicted_families, _ = infer_model_with_labels(model, valid_train_loader_for_classification, device, use_mixed_precision=False, desc='Inferring train samples...')
        valid_labels, valid_labels_ids, valid_families, valid_families_ids, valid_embeddings, valid_predicted_families, _ = infer_model_with_labels(model, valid_valid_loader_for_classification, device, use_mixed_precision=False, desc='Inferring valid samples...')

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

        if isinstance(actual_model, (XPrizeTreeEmbedder2, XPrizeTreeEmbedder2NoDate)):
            accuracy_train = accuracy_score(train_families, train_predicted_families)
            accuracy_valid = accuracy_score(valid_families, valid_predicted_families)
            f1_macro = precision_recall_fscore_support(valid_families, valid_predicted_families, average='macro')[2]
            f1_weighted = precision_recall_fscore_support(valid_families, valid_predicted_families, average='weighted')[2]
            writer.add_scalar('Train/Family_XPrizeTreeEmbedder2/Accuracy', accuracy_valid, overall_step)
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
        valid_labels, valid_labels_ids, valid_families, valid_families_ids, valid_embeddings, valid_predicted_families, _ = infer_model_with_labels(model, valid_loader, device, use_mixed_precision=False, as_numpy=False)
        total_loss = criterion_metric(embeddings=valid_embeddings, labels=valid_labels_ids, indices_tuple=triplets_tuples)

        writer.add_scalar('Valid/Loss_Triplet', total_loss / triplets_tuples[0].shape[0], overall_step)

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
    max_resampling_times_train = yaml_config['max_resampling_times_train']
    max_resampling_times_valid_train = yaml_config['max_resampling_times_valid_train']
    valid_knn_k = yaml_config['valid_knn_k']
    min_level = yaml_config['min_level']
    distance = yaml_config['distance']
    resnet_model = yaml_config['resnet_model']
    use_multi_gpu = yaml_config['use_multi_gpu']
    start_from_checkpoint = yaml_config['start_from_checkpoint']
    learning_rate = yaml_config['learning_rate']
    train_batch_size = yaml_config['train_batch_size']
    n_grad_accumulation_steps = yaml_config['n_grad_accumulation_steps']
    valid_batch_size = train_batch_size * yaml_config['valid_batch_size_multiplier']
    image_size = yaml_config['image_size']
    random_crop = yaml_config['random_crop']
    loss_weight_triplet = yaml_config['loss_weight_triplet']
    loss_weight_classification = yaml_config['loss_weight_classification']
    triplet_margin = yaml_config['triplet_margin']
    triplet_type = yaml_config['triplet_type']
    dropout = yaml_config['dropout']
    final_embedding_size = yaml_config['final_embedding_size']
    scheduler_T = yaml_config['scheduler_T']
    n_warmup_epochs = yaml_config['n_warmup_epochs']
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

    train_dataset_config = {}

    if 'brazil' in use_datasets:
        train_dataset_brazil = ContrastiveInternalDataset(
            fold='train',
            root_path=source_data_root / 'brazil_zf2',
            date_pattern=brazil_date_pattern
        )
        train_dataset_config['brazil'] = train_dataset_brazil
    if 'equator' in use_datasets:
        train_dataset_equator = ContrastiveInternalDataset(
            fold='train',
            root_path=source_data_root / 'equator',
            date_pattern=equator_date_pattern
        )
        train_dataset_config['equator'] = train_dataset_equator
    if 'panama' in use_datasets:
        train_dataset_panama = ContrastiveInternalDataset(
            fold='train',
            root_path=source_data_root / 'panama',
            date_pattern=panama_date_pattern
        )
        train_dataset_config['panama'] = train_dataset_panama
    if 'quebec' in use_datasets:
        train_dataset_quebec = ContrastiveInternalDataset(
            fold='train',
            root_path=source_data_root / 'quebec_trees',
            date_pattern=quebec_date_pattern
        )
        train_dataset_config['quebec'] = train_dataset_quebec

    siamese_sampler_dataset_train = ContrastiveDataset(
        dataset_config=train_dataset_config,
        min_level=min_level,
        image_size=image_size,
        random_crop=random_crop,
        transform=A.Compose(embedder_transforms_v2),
        normalize=True,
        mean=FOREST_QPEB_MEAN,
        std=FOREST_QPEB_STD,
        taxa_distances_df=pd.read_csv(phylogenetic_tree_distances_path),
        max_resampling_times=max_resampling_times_train
    )

    siamese_sampler_dataset_valid_knn_train = ContrastiveDataset(
        dataset_config=train_dataset_config,
        min_level=min_level,
        image_size=image_size,
        random_crop=random_crop,
        transform=A.Compose(embedder_simple_transforms_v2),
        normalize=True,
        mean=FOREST_QPEB_MEAN,
        std=FOREST_QPEB_STD,
        taxa_distances_df=pd.read_csv(phylogenetic_tree_distances_path),
        max_resampling_times=max_resampling_times_valid_train
    )

    valid_datasets = {}
    if 'brazil' in use_datasets:
        valid_dataset_brazil = ContrastiveInternalDataset(
            fold='valid',
            root_path=source_data_root / 'brazil_zf2',
            date_pattern=brazil_date_pattern
        )
        valid_datasets['brazil'] = ContrastiveDataset(
            dataset_config={'brazil': valid_dataset_brazil},
            min_level=min_level,
            image_size=image_size,
            transform=None,
            normalize=True,
            mean=FOREST_QPEB_MEAN,
            std=FOREST_QPEB_STD,
            taxa_distances_df=pd.read_csv(phylogenetic_tree_distances_path),
            max_resampling_times=0
        )
        valid_datasets['brazil'].shuffle(seed=0)       # shuffling once to make sure species labels are well mixed for mini-batches
    if 'equator' in use_datasets:
        valid_dataset_equator = ContrastiveInternalDataset(
            fold='valid',
            root_path=source_data_root / 'equator',
            date_pattern=equator_date_pattern
        )
        valid_datasets['equator'] = ContrastiveDataset(
            dataset_config={'equator': valid_dataset_equator},
            min_level=min_level,
            image_size=image_size,
            random_crop=False,
            transform=None,
            normalize=True,
            mean=FOREST_QPEB_MEAN,
            std=FOREST_QPEB_STD,
            taxa_distances_df=pd.read_csv(phylogenetic_tree_distances_path),
            max_resampling_times=0
        )
        valid_datasets['equator'].shuffle(seed=0)       # shuffling once to make sure species labels are well mixed for mini-batches
    if 'panama' in use_datasets:
        valid_dataset_panama = ContrastiveInternalDataset(
            fold='valid',
            root_path=source_data_root / 'panama',
            date_pattern=panama_date_pattern
        )
        valid_datasets['panama'] = ContrastiveDataset(
            dataset_config={'panama': valid_dataset_panama},
            min_level=min_level,
            image_size=image_size,
            random_crop=False,
            transform=None,
            normalize=True,
            mean=FOREST_QPEB_MEAN,
            std=FOREST_QPEB_STD,
            taxa_distances_df=pd.read_csv(phylogenetic_tree_distances_path),
            max_resampling_times=0
        )
        valid_datasets['panama'].shuffle(seed=0)       # shuffling once to make sure species labels are well mixed for mini-batches
    if 'quebec' in use_datasets:
        valid_dataset_quebec = ContrastiveInternalDataset(
            fold='valid',
            root_path=source_data_root / 'quebec_trees',
            date_pattern=quebec_date_pattern
        )
        valid_datasets['quebec'] = ContrastiveDataset(
            dataset_config={'quebec': valid_dataset_quebec},
            min_level=min_level,
            image_size=image_size,
            random_crop=False,
            transform=None,
            normalize=True,
            mean=FOREST_QPEB_MEAN,
            std=FOREST_QPEB_STD,
            taxa_distances_df=pd.read_csv(phylogenetic_tree_distances_path),
            max_resampling_times=0
        )
        valid_datasets['quebec'].shuffle(seed=0)       # shuffling once to make sure species labels are well mixed for mini-batches

    print('Datasets loaded successfully.')

    # model = XPrizeTreeEmbedder2(
    #     resnet_model=resnet_model,
    #     final_embedding_size=final_embedding_size,
    #     dropout=dropout,
    #     date_embedding_dim=32,
    #     families=list(siamese_sampler_dataset_train.families_set),
    # ).to(device)

    # model = XPrizeTreeEmbedder2NoDate(
    #     resnet_model=resnet_model,
    #     final_embedding_size=final_embedding_size,
    #     dropout=dropout,
    #     families=list(siamese_sampler_dataset_train.families_set),
    # ).to(device)

    model = DinoV2Embedder(
        size='small',
        final_embedding_size=final_embedding_size,
        dropout=dropout
    ).to(device)

    if start_from_checkpoint:
        if isinstance(model, XPrizeTreeEmbedder):
            model.load_state_dict(torch.load(start_from_checkpoint))
        elif isinstance(model, (XPrizeTreeEmbedder2, XPrizeTreeEmbedder2NoDate)):
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
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=n_warmup_epochs, after_scheduler=scheduler)
    optimizer.step()  # Step once to avoid lr of 0 from scheduler
    scheduler.step()  # Step once to avoid lr of 0 from scheduler

    output_dir = output_folder_root / f'{output_model_name}_{int(time.time())}'
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(str(output_dir))

    with open(output_dir / 'contrastive_train_config.yaml', 'w') as config_file:
        yaml.safe_dump(yaml_config, config_file)

    train_sampler = MPerClassSampler(siamese_sampler_dataset_train.all_samples_labels, m=4, batch_size=train_batch_size, length_before_new_iter=len(siamese_sampler_dataset_train))
    train_dataloader = DataLoader(
        siamese_sampler_dataset_train,
        batch_size=train_batch_size,
        collate_fn=contrastive_collate_fn,
        pin_memory=use_multi_gpu,
        num_workers=data_loader_num_workers,
        sampler=train_sampler,
    )

    valid_train_dataloader = DataLoader(
        siamese_sampler_dataset_valid_knn_train,
        batch_size=train_batch_size,
        collate_fn=contrastive_collate_fn,
        pin_memory=use_multi_gpu,
        num_workers=data_loader_num_workers
    )

    valid_dataloaders = {}
    for valid_dataset_name, valid_dataset in valid_datasets.items():
        valid_dataloaders[valid_dataset_name] = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=valid_batch_size,
            shuffle=False,
            num_workers=data_loader_num_workers,
            collate_fn=contrastive_collate_fn
        )

    train(
        model=model,
        train_dataloader=train_dataloader,
        valid_train_dataloader=valid_train_dataloader,
        valid_dataloaders=valid_dataloaders,
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
        valid_knn_k=valid_knn_k
    )

