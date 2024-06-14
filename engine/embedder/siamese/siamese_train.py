import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

from engine.embedder.siamese.siamese_dataset import SiameseSamplerDataset, SiameseSamplerInternalDataset, \
    SiameseValidationDataset

from engine.embedder.siamese.siamese_model import SiameseNetwork2, ContrastiveLoss
from engine.embedder.siamese.siamese_utils import train_collate_fn, valid_collate_fn
from engine.embedder.siamese.transforms import embedder_transforms
from engine.constants import data_paths


def infer_model(model, dataloader, device):
    all_labels = []
    all_embeddings = []

    for batch_images, batch_labels in dataloader:
        data = torch.Tensor(batch_images).to(device)
        if len(data.shape) == 3:
            data = data.unsqueeze(0)

        output = model.infer(data)
        output = output.detach().cpu().numpy()

        all_labels.append(batch_labels.cpu().numpy())
        all_embeddings.append(output)

    final_labels = np.concatenate(all_labels, axis=0)
    final_embeddings = np.concatenate(all_embeddings, axis=0)

    return final_labels, final_embeddings


def train(model, data_loader, valid_train_loader, valid_valid_loader, criterion, optimizer, writer, output_dir, save_every_n_steps, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        loss_since_last_log = 0
        step_since_last_log = 0
        overall_step = epoch * len(data_loader)
        for data in tqdm(data_loader, desc=f'Epoch {epoch}...'):
            imgs1, imgs2, labels, margins = data
            imgs1, imgs2, labels = imgs1.to(device), imgs2.to(device), labels.to(device)

            optimizer.zero_grad()
            output1, output2 = model(imgs1, imgs2)

            loss = criterion(output1, output2, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            overall_step += 1
            loss_since_last_log += loss.item()
            step_since_last_log += 1

            if overall_step != 0 and overall_step % 100 == 0:
                writer.add_scalar('Loss', loss_since_last_log / step_since_last_log, overall_step)
                loss_since_last_log = 0
                step_since_last_log = 0

            if overall_step != 0 and overall_step % save_every_n_steps == 0:
                print(f'Validating for step {overall_step}...')
                model.eval()
                with torch.no_grad():
                    train_labels, train_embeddings = infer_model(model, valid_train_loader, device)
                    valid_labels, valid_embeddings = infer_model(model, valid_valid_loader, device)

                    train_to_delete = train_labels == -1
                    valid_to_delete = valid_labels == -1

                    train_embeddings = train_embeddings[~train_to_delete]
                    train_labels = train_labels[~train_to_delete]
                    valid_embeddings = valid_embeddings[~valid_to_delete]
                    valid_labels = valid_labels[~valid_to_delete]

                    # Standardize the embeddings
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(train_embeddings)
                    X_test = scaler.transform(valid_embeddings)

                    # Train the SVC
                    svc = SVC(kernel='linear')

                    svc.fit(X_train, train_labels)

                    # Evaluate the SVC
                    y_pred = svc.predict(X_test)

                    accuracy = accuracy_score(valid_labels, y_pred)
                    metrics = precision_recall_fscore_support(valid_labels, y_pred, average='macro')
                    metrics_weighted = precision_recall_fscore_support(valid_labels, y_pred, average='weighted')
                    writer.add_scalar('Valid/Accuracy', accuracy, overall_step)
                    writer.add_scalar('Valid/Macro_Precision', metrics[0], overall_step)
                    writer.add_scalar('Valid/Macro_Recall', metrics[1], overall_step)
                    writer.add_scalar('Valid/Macro_F1', metrics[2], overall_step)
                    writer.add_scalar('Valid/Weighted_Precision', metrics_weighted[0], overall_step)
                    writer.add_scalar('Valid/Weighted_Recall', metrics_weighted[1], overall_step)
                    writer.add_scalar('Valid/Weighted_F1', metrics_weighted[2], overall_step)

                checkpoint_output_file = os.path.join(output_dir, f'checkpoint_{epoch}_{overall_step}.pth')
                torch.save(model.state_dict(), checkpoint_output_file)
                model.train()

        print(f'Epoch {epoch}, Loss: {total_loss / len(data_loader)}')

        checkpoint_output_file = os.path.join(output_dir, f'checkpoint_{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_output_file)


if __name__ == '__main__':
    resnet_model = 'resnet50'
    final_embedding_size = 1024

    train_dataset_quebec = SiameseSamplerInternalDataset(
        fold='train',
        root_path=Path('C:/Users/Hugo/Documents/Data/pre_processed/final_dataset_polygons/quebec_trees')
    )

    dataset_config = {
        'datasets': {
            'quebec': train_dataset_quebec,
            # 'equator': dataset_equator,
            # 'neon': dataset_neon,
        },
        'sampling_strategies': [
            {
                'dataset_1': 'quebec',
                'positive_pairs': {
                    'strategies': ['class'],
                    'total_n_samples': 1000000,      # TODO add more samples
                    'lower_than_percentile': 75     # TODO tweak this, probably use something closer to 50 to only keep hard positive pairs
                },
                'negative_pairs': {
                    'strategies': ['class'],
                    'total_n_samples': 1000000,      # TODO add more samples
                    'higher_than_percentile': 25,   # TODO tweak this, probably use 0 as its ok to use all negative pairs
                    'margin': 1
                }
            },
            # {
            #     'dataset_1': 'equator',
            #     'positive_pairs_strategies': ['class'],
            #     'negative_pairs_strategies': ['class'],
            #     'positive_pairs_percentage': 0.2,
            #     'negative_pairs_percentage': 0.2,
            #     'negative_pairs_margin': 1
            # },
            # {
            #     'dataset_1_id': 'quebec',
            #     'dataset_2_id': 'equator',
            #     'positive_pairs_strategies': ['none'],
            #     'negative_pairs_strategies': ['any'],
            #     'positive_pairs_percentage': 0.2,
            #     'negative_pairs_percentage': 0.2,
            #     'negative_pairs_margin': 3
            # }
        ]
    }

    transform = A.Compose(embedder_transforms)

    siamese_sampler = SiameseSamplerDataset(dataset_config=dataset_config, transform=transform)

    valid_train_dataset = SiameseValidationDataset(
        fold='train',
        root_path=Path('C:/Users/Hugo/Documents/Data/pre_processed/final_dataset_polygons/quebec_trees')
    )
    valid_valid_dataset = SiameseValidationDataset(
        fold='valid',
        root_path=Path('C:/Users/Hugo/Documents/Data/pre_processed/final_dataset_polygons/quebec_trees')
    )

    valid_train_loader = torch.utils.data.DataLoader(valid_train_dataset, batch_size=128, shuffle=False, num_workers=3,
                                                     collate_fn=valid_collate_fn)
    valid_valid_loader = torch.utils.data.DataLoader(valid_valid_dataset, batch_size=128, shuffle=False, num_workers=0,
                                                     collate_fn=valid_collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNetwork2(resnet_model=resnet_model, final_embedding_size=final_embedding_size).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)      # TODO adjust this

    train_loader = DataLoader(siamese_sampler, batch_size=32, shuffle=True, collate_fn=train_collate_fn, num_workers=3)      # TODO try accumulate gradient to increase effective batch size beyond GPU memory limits

    save_every_n_steps = 2000
    output_dir = f'./output_test/siamese_2_quebec_NEW_SAMPLING_{time.time()}'  # Replace with your actual directory
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(output_dir)

    train(model, train_loader, valid_train_loader, valid_valid_loader, criterion, optimizer, writer, output_dir, save_every_n_steps, num_epochs=10)


