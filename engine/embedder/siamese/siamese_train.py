print('Script started')
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
print('First imports done')

from engine.embedder.siamese.siamese_dataset import SiameseSamplerDataset, SiameseSamplerInternalDataset, \
    SiameseValidationDataset, SingleItemsSiameseSamplerDatasetWrapper

from engine.embedder.siamese.siamese_model import ContrastiveLoss, SiameseNetwork2
from engine.embedder.siamese.siamese_utils import train_collate_fn2, valid_collate_fn2, FOREST_QPEB_MEAN, \
    FOREST_QPEB_STD, valid_collate_fn_string_labels
from engine.embedder.siamese.transforms import embedder_transforms

print('Other imports done')

def infer_model(model, dataloader, device, use_mixed_precision, use_multi_gpu, desc='Infering...'):
    all_labels = []
    all_embeddings = []

    for batch_images, batch_months, batch_days, batch_labels in tqdm(dataloader, total=len(dataloader), desc=desc):
        data = torch.Tensor(batch_images).to(device)
        batch_months = torch.Tensor(batch_months).to(device)
        batch_days = torch.Tensor(batch_days).to(device)
        if len(data.shape) == 3:
            data = data.unsqueeze(0)

        if use_multi_gpu:
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    output = model.module.infer(data, batch_months, batch_days)
            else:
                output = model.module.infer(data, batch_months, batch_days)
        else:
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    output = model.infer(data, batch_months, batch_days)
            else:
                output = model.infer(data, batch_months, batch_days)

        output = output.detach().cpu().numpy()

        if type(batch_labels) is list:
            all_labels.append(batch_labels)
        else:
            all_labels.append(batch_labels.cpu().numpy())
        all_embeddings.append(output)

    final_labels = np.concatenate(all_labels, axis=0)
    final_embeddings = np.concatenate(all_embeddings, axis=0)

    return final_labels, final_embeddings


def save_model(model, checkpoint_output_file, use_multi_gpu):
    if use_multi_gpu:
        # Save the original model which is wrapped inside `.module`
        torch.save(model.module.state_dict(), checkpoint_output_file)
    else:
        # Directly save the model
        torch.save(model.state_dict(), checkpoint_output_file)


def train(model: SiameseNetwork2,
          train_dataset: SiameseSamplerDataset,
          valid_dataset: SiameseSamplerDataset,
          valid_train_dataset_for_classification: SiameseValidationDataset,
          valid_valid_dataset_for_classification: SiameseValidationDataset,
          use_multi_gpu: bool,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.StepLR,
          n_grad_accumulation_steps: int,
          writer: SummaryWriter,
          output_dir: Path,
          save_every_n_updates: int,
          num_epochs: int,
          data_loader_num_workers: int):

    if use_multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    scaler = torch.cuda.amp.GradScaler()  # Initialize the GradScaler for mixed precision training

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        loss_since_last_log = 0
        step_since_last_log = 0
        # re-instantiate the dataloader at the start of each epoch as the sampling was re-generated at the end of every epoch
        data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=train_collate_fn2,
                                 pin_memory=use_multi_gpu, num_workers=data_loader_num_workers)
        overall_step = epoch * len(data_loader) // n_grad_accumulation_steps
        accumulated_steps = 0

        for data in tqdm(data_loader, desc=f'Epoch {epoch}...'):
            imgs1, imgs2, months1, months2, days1, days2, labels, margins = data
            imgs1, imgs2, labels, margins = imgs1.to(device), imgs2.to(device), labels.to(device), margins.to(device)
            months1, months2, days1, days2 = months1.to(device), months2.to(device), days1.to(device), days2.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  # Enable autocasting for mixed precision
                output1, output2 = model(imgs1, imgs2, months1, months2, days1, days2)
                loss = criterion(output1, output2, labels, margins)

            scaler.scale(loss).backward()  # Backward pass with scaled loss

            total_loss += loss.item()
            loss_since_last_log += loss.item()
            step_since_last_log += 1
            accumulated_steps += 1

            if accumulated_steps == n_grad_accumulation_steps:
                scaler.step(optimizer)  # Update the model parameters with scaled optimizer step
                scheduler.step()
                scaler.update()  # Update the scaler for next iteration
                optimizer.zero_grad()
                accumulated_steps = 0
                overall_step += 1

                if overall_step != 0 and overall_step % 100 == 0:
                    writer.add_scalar('Loss', loss_since_last_log / step_since_last_log, overall_step)
                    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], overall_step)
                    loss_since_last_log = 0
                    step_since_last_log = 0

                if overall_step != 0 and overall_step % save_every_n_updates == 0:
                    model.eval()
                    validate_for_classification(
                        model=model,
                        valid_train_dataset_for_classification=valid_train_dataset_for_classification,
                        valid_valid_dataset_for_classification=valid_valid_dataset_for_classification,
                        overall_step=overall_step,
                        writer=writer,
                        data_loader_num_workers=data_loader_num_workers,
                        use_multi_gpu=use_multi_gpu
                    )

                    checkpoint_output_file = os.path.join(output_dir, f'checkpoint_{epoch}_{overall_step}.pth')
                    save_model(model, checkpoint_output_file=checkpoint_output_file, use_multi_gpu=use_multi_gpu)
                    model.train()

        # Check if there are remaining accumulated gradients
        if accumulated_steps > 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        model.eval()
        valid_loss = validate_for_loss(
            model=model,
            valid_dataset=valid_dataset,
            epoch=epoch,
            overall_step=overall_step,
            writer=writer,
            data_loader_num_workers=data_loader_num_workers,
            use_multi_gpu=use_multi_gpu
        )

        train_dataset = find_optimal_pairs(
            model=model,
            dataset=train_dataset,
            data_loader_num_workers=data_loader_num_workers,
            use_multi_gpu=use_multi_gpu
        )
        model.train()

        print(f'Epoch {epoch}, Train Loss: {total_loss / len(data_loader)}, Valid Loss: {valid_loss}')

        checkpoint_output_file = os.path.join(output_dir, f'checkpoint_{epoch}.pth')
        save_model(model, checkpoint_output_file=checkpoint_output_file, use_multi_gpu=use_multi_gpu)


def find_optimal_pairs(model, dataset, data_loader_num_workers, use_multi_gpu):
    with torch.no_grad():
        single_item_dataset = SingleItemsSiameseSamplerDatasetWrapper(
            siamese_sampler_dataset=dataset
        )

        data_loader = DataLoader(
            single_item_dataset,
            batch_size=valid_batch_size,
            shuffle=False,
            collate_fn=valid_collate_fn_string_labels,
            num_workers=data_loader_num_workers,
        )

        _, embeddings = infer_model(model, data_loader, device, use_mixed_precision=True, use_multi_gpu=use_multi_gpu, desc='Inferring to find optimal pairs...')

        print(embeddings.shape)

        dataset.find_optimal_siamese_pairs(
            embeddings=torch.Tensor(embeddings),
            compute_device=device,
            distance_compute_batch_size=100000000
        )

        return dataset


def validate_for_loss(model, valid_dataset, epoch, overall_step, writer, data_loader_num_workers, use_multi_gpu):
    print(f'Validating for update {overall_step}...')

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False,
                                               num_workers=data_loader_num_workers, collate_fn=train_collate_fn2)

    total_loss = 0

    model.eval()
    with torch.no_grad():
        for data in tqdm(valid_loader, desc=f'Valid for epoch {epoch}...'):
            imgs1, imgs2, months1, months2, days1, days2, labels, margins = data
            imgs1, imgs2, labels, margins = imgs1.to(device), imgs2.to(device), labels.to(device), margins.to(device)
            months1, months2, days1, days2 = months1.to(device), months2.to(device), days1.to(device), days2.to(device)

            with torch.cuda.amp.autocast():  # Enable autocasting for mixed precision
                output1, output2 = model(imgs1, imgs2, months1, months2, days1, days2)
                loss = criterion(output1, output2, labels, margins)

            total_loss += loss.item()

        writer.add_scalar('Valid/Loss', total_loss / len(valid_loader), overall_step)

        return total_loss / len(valid_loader)


def validate_for_classification(model,
                                valid_train_dataset_for_classification,
                                valid_valid_dataset_for_classification,
                                overall_step,
                                writer,
                                data_loader_num_workers,
                                use_multi_gpu):
    print(f'Validating for update {overall_step}...')

    valid_train_loader_for_classification = torch.utils.data.DataLoader(valid_train_dataset_for_classification,
                                                                        batch_size=valid_batch_size, shuffle=False,
                                                                        num_workers=data_loader_num_workers,
                                                                        collate_fn=valid_collate_fn2)
    valid_valid_loader_for_classification = torch.utils.data.DataLoader(valid_valid_dataset_for_classification,
                                                                        batch_size=valid_batch_size, shuffle=False,
                                                                        num_workers=data_loader_num_workers,
                                                                        collate_fn=valid_collate_fn2)

    model.eval()
    with torch.no_grad():
        train_labels, train_embeddings = infer_model(model, valid_train_loader_for_classification, device, use_multi_gpu=use_multi_gpu, use_mixed_precision=False)
        valid_labels, valid_embeddings = infer_model(model, valid_valid_loader_for_classification, device, use_multi_gpu=use_multi_gpu, use_mixed_precision=False)

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
        svc = SVC(kernel='linear')
        svc.fit(X_train, train_labels)

        # Evaluate the SVC
        y_pred = svc.predict(X_test)

        accuracy = accuracy_score(valid_labels, y_pred)
        metrics = precision_recall_fscore_support(valid_labels, y_pred, average='macro')
        metrics_weighted = precision_recall_fscore_support(valid_labels, y_pred,
                                                           average='weighted')
        writer.add_scalar('Valid/Accuracy', accuracy, overall_step)
        writer.add_scalar('Valid/Macro_Precision', metrics[0], overall_step)
        writer.add_scalar('Valid/Macro_Recall', metrics[1], overall_step)
        writer.add_scalar('Valid/Macro_F1', metrics[2], overall_step)
        writer.add_scalar('Valid/Weighted_Precision', metrics_weighted[0], overall_step)
        writer.add_scalar('Valid/Weighted_Recall', metrics_weighted[1], overall_step)
        writer.add_scalar('Valid/Weighted_F1', metrics_weighted[2], overall_step)
        print(f'Validation Accuracy: {accuracy}')


if __name__ == '__main__':
    print('====Siamese training main====')
    parser = argparse.ArgumentParser(description="Script to train the Siamese network for XPrize Rainforest.")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--data_root', type=str, required=False)
    args = parser.parse_args()

    yaml_config = yaml.safe_load(open(args.config_path, 'r'))

    source_data_root = Path(yaml_config['source_data_root'])
    if args.data_root:
        source_data_root = Path(args.data_root)

    use_datasets = yaml_config['use_datasets']
    resnet_model = yaml_config['resnet_model']
    use_multi_gpu = yaml_config['use_multi_gpu']
    start_from_checkpoint = yaml_config['start_from_checkpoint']
    learning_rate = yaml_config['learning_rate']
    train_batch_size = yaml_config['train_batch_size']
    n_grad_accumulation_steps = yaml_config['n_grad_accumulation_steps']
    valid_batch_size = train_batch_size * yaml_config['valid_batch_size_multiplier']
    image_size = yaml_config['image_size']
    dropout = yaml_config['dropout']
    final_embedding_size = yaml_config['final_embedding_size']
    save_every_n_updates = yaml_config['save_every_n_updates']
    scheduler_step_every_n_updates = yaml_config['scheduler_step_every_n_updates']
    scheduler_gamma = yaml_config['scheduler_gamma']
    num_epochs = yaml_config['num_epochs']
    n_positive_pairs_train = yaml_config['n_positive_pairs_train']
    n_negative_pairs_train = yaml_config['n_negative_pairs_train']
    n_positive_pairs_valid = yaml_config['n_positive_pairs_valid']
    n_negative_pairs_valid = yaml_config['n_negative_pairs_valid']
    consider_percentile_train = yaml_config['consider_percentile_train']
    consider_percentile_valid = yaml_config['consider_percentile_valid']
    data_loader_num_workers = yaml_config['data_loader_num_workers']
    phylogenetic_tree_distances_path = yaml_config['phylogenetic_tree_distances_path']
    output_folder_root = Path(yaml_config['output_folder_root'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_model_name = f'siamese_{resnet_model}_{image_size}_{final_embedding_size}_{train_batch_size * n_grad_accumulation_steps}_mpt'

    print("Config loaded.")

    train_dataset_brazil = SiameseSamplerInternalDataset(
        fold='train',
        root_path=source_data_root / 'brazil_zf2',
        date_pattern=r'^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})'
    )
    train_dataset_equator = SiameseSamplerInternalDataset(
        fold='train',
        root_path=source_data_root / 'equator',
        date_pattern=r'^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})'
    )
    train_dataset_panama = SiameseSamplerInternalDataset(
        fold='train',
        root_path=source_data_root / 'panama',
        date_pattern=r'^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})|bci_50ha_(?P<year2>\d{4})_(?P<month2>\d{2})_(?P<day2>\d{2})_'
    )
    train_dataset_quebec = SiameseSamplerInternalDataset(
        fold='train',
        root_path=source_data_root / 'quebec_trees',
        date_pattern=r'^(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_'
    )

    valid_dataset_brazil = SiameseSamplerInternalDataset(
        fold='valid',
        root_path=source_data_root / 'brazil_zf2',
        date_pattern=r'^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})'
    )
    valid_dataset_equator = SiameseSamplerInternalDataset(
        fold='valid',
        root_path=source_data_root / 'equator',
        date_pattern=r'^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})'
    )
    valid_dataset_panama = SiameseSamplerInternalDataset(
        fold='valid',
        root_path=source_data_root / 'panama',
        date_pattern=r'^(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})|bci_50ha_(?P<year2>\d{4})_(?P<month2>\d{2})_(?P<day2>\d{2})_'
    )
    valid_dataset_quebec = SiameseSamplerInternalDataset(
        fold='valid',
        root_path=source_data_root / 'quebec_trees',
        date_pattern=r'^(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_'
    )

    train_dataset_config = {
        'datasets': {
            'brazil': train_dataset_brazil,
            'equator': train_dataset_equator,
            'panama': train_dataset_panama,
            'quebec': train_dataset_quebec
        },
    }
    valid_dataset_config = {
        'datasets': {
            'brazil': train_dataset_brazil,
            'equator': train_dataset_equator,
            'panama': valid_dataset_panama,
            'quebec': valid_dataset_quebec
        },
    }

    if 'brazil' not in use_datasets:
        del train_dataset_config['datasets']['brazil']
        del valid_dataset_config['datasets']['brazil']
    if 'equator' not in use_datasets:
        del train_dataset_config['datasets']['equator']
        del valid_dataset_config['datasets']['equator']
    if 'panama' not in use_datasets:
        del train_dataset_config['datasets']['panama']
        del valid_dataset_config['datasets']['panama']
    if 'quebec' not in use_datasets:
        del train_dataset_config['datasets']['quebec']
        del valid_dataset_config['datasets']['quebec']

    siamese_sampler_dataset_train = SiameseSamplerDataset(
        dataset_config=train_dataset_config,
        image_size=image_size,
        transform=A.Compose(embedder_transforms),
        n_positive_pairs=n_positive_pairs_train,
        n_negative_pairs=n_negative_pairs_train,
        consider_percentile=consider_percentile_train,
        normalize=True,
        mean=FOREST_QPEB_MEAN,
        std=FOREST_QPEB_STD,
        taxa_distances_df=pd.read_csv(phylogenetic_tree_distances_path)
    )

    siamese_sampler_dataset_valid = SiameseSamplerDataset(
        dataset_config=valid_dataset_config,
        image_size=image_size,
        transform=None,
        n_positive_pairs=n_positive_pairs_valid,
        n_negative_pairs=n_negative_pairs_valid,
        consider_percentile=consider_percentile_valid,
        normalize=True,
        mean=FOREST_QPEB_MEAN,
        std=FOREST_QPEB_STD,
        taxa_distances_df=pd.read_csv(phylogenetic_tree_distances_path)
    )

    valid_train_dataset_quebec_for_classification = SiameseValidationDataset(
        fold='train',
        root_path=source_data_root / 'quebec_trees/from_annotations',
        date_pattern=r'^(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_',
        image_size=image_size,
        normalize=True,
        mean=FOREST_QPEB_MEAN,
        std=FOREST_QPEB_STD
    )
    valid_valid_dataset_quebec_for_classification = SiameseValidationDataset(
        fold='valid',
        root_path=source_data_root / 'quebec_trees/from_annotations',
        date_pattern=r'^(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_',
        image_size=image_size,
        normalize=True,
        mean=FOREST_QPEB_MEAN,
        std=FOREST_QPEB_STD
    )

    model = SiameseNetwork2(
        resnet_model=resnet_model,
        final_embedding_size=final_embedding_size,
        dropout=dropout
    ).to(device)

    if start_from_checkpoint:
        model.load_state_dict(torch.load(start_from_checkpoint))
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_every_n_updates, gamma=scheduler_gamma)

    output_dir = output_folder_root / f'{output_model_name}_{int(time.time())}'
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(str(output_dir))

    with open(output_dir / 'siamese_train_config.yaml', 'w') as config_file:
        yaml.safe_dump(yaml_config, config_file)

    train(
        model=model,
        train_dataset=siamese_sampler_dataset_train,
        valid_dataset=siamese_sampler_dataset_valid,
        valid_train_dataset_for_classification=valid_train_dataset_quebec_for_classification,
        valid_valid_dataset_for_classification=valid_valid_dataset_quebec_for_classification,
        use_multi_gpu=use_multi_gpu,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        n_grad_accumulation_steps=n_grad_accumulation_steps,
        writer=writer,
        output_dir=output_dir,
        save_every_n_updates=save_every_n_updates,
        num_epochs=num_epochs,
        data_loader_num_workers=data_loader_num_workers
    )


