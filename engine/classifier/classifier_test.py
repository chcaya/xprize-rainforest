import os

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Check if CUDA is available and set device to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes, label_map):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
        self.label_map = {v: k for k, v in label_map.items()}  # Store the reverse mapping

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
            # Map predictions back to original labels
            mapped_predictions = [self.label_map[pred.item()] for pred in predictions]
            return mapped_predictions


def map_labels(labels):
    unique_labels = np.unique(labels)
    label_map = {original: new for new, original in enumerate(unique_labels)}
    mapped_labels = np.vectorize(label_map.get)(labels)
    return mapped_labels, label_map


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs, output_dir):
    best_f1_macro = 0
    writer = SummaryWriter(output_dir)

    step = 0
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()

        total_loss = 0
        for data, target in tqdm(train_loader, desc="Training", leave=False):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            total_loss += loss_value
            if step % 100 == 0:
                writer.add_scalar('Loss/train', loss_value, step)
            step += 1

        scheduler.step()  # Adjust the learning rate based on the scheduler

        print(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader)}")

        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for data, target in tqdm(valid_loader, desc="Validation", leave=False):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                preds = torch.argmax(output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_val_loss = val_loss / len(valid_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
        accuracy = accuracy_score(all_targets, all_preds)
        writer.add_scalar('Accuracy/validation', accuracy, epoch)
        writer.add_scalar('Precision/validation/macro', macro_precision, epoch)
        writer.add_scalar('Recall/validation/macro', macro_recall, epoch)
        writer.add_scalar('F1-Score/validation/macro', macro_f1, epoch)
        writer.add_scalar('Precision/validation/weighted', weighted_precision, epoch)
        writer.add_scalar('Recall/validation/weighted', weighted_recall, epoch)
        writer.add_scalar('F1-Score/validation/weighted', weighted_f1, epoch)

        torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{epoch}.pth'))
        if macro_f1 > best_f1_macro:
            best_f1_macro = macro_f1
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_checkpoint.pth'))
            print("Saved best model checkpoint (based on macro_f1 metric).")

        writer.flush()
    writer.close()


def load_data(filename):
    # Load a dataframe and convert embeddings from string to numpy arrays
    df = pd.read_csv(filename)
    df['embeddings'] = df['embeddings'].apply(eval)  # Convert string representations of lists to actual lists
    return df


if __name__ == "__main__":
    # Load datasets
    print("Loading data...")
    train_df_z1 = load_data(
        'C:/Users/Hugo/Documents/Data/embeddings/embeddings_dinov2_large/2021_09_02_sbl_z1_rgb_cog_embeddings_train.csv')
    train_df_z2 = load_data(
        'C:/Users/Hugo/Documents/Data/embeddings/embeddings_dinov2_large/2021_09_02_sbl_z2_rgb_cog_embeddings_train.csv')
    valid_df_z1 = load_data(
        'C:/Users/Hugo/Documents/Data/embeddings/embeddings_dinov2_large/2021_09_02_sbl_z1_rgb_cog_embeddings_valid.csv')
    test_df_z1 = load_data(
        'C:/Users/Hugo/Documents/Data/embeddings/embeddings_dinov2_large/2021_09_02_sbl_z1_rgb_cog_embeddings_test.csv')
    test_df_z3 = load_data(
        'C:/Users/Hugo/Documents/Data/embeddings/embeddings_dinov2_large/2021_09_02_sbl_z3_rgb_cog_embeddings_test.csv')

    # Concatenate the train dataframes
    print("Concatenating data...")
    train_df = pd.concat([train_df_z1, train_df_z2], ignore_index=True)
    valid_df = valid_df_z1
    test_df = pd.concat([test_df_z1, test_df_z3], ignore_index=True)

    # Drop -1 labels
    print("Dropping '-1' labels...")
    train_df = train_df[train_df['labels'] != -1]
    valid_df = valid_df[valid_df['labels'] != -1]
    test_df = test_df[test_df['labels'] != -1]

    # Prepare data
    print("Preparing data...")
    X_train = np.vstack(train_df['embeddings'])
    y_train = train_df['labels'].values
    X_valid = np.vstack(valid_df['embeddings'])
    y_valid = valid_df['labels'].values
    X_test = np.vstack(test_df['embeddings'])
    y_test = test_df['labels'].values

    print("Mapping labels...")
    y_train, train_label_map = map_labels(y_train)
    y_valid, valid_label_map = map_labels(y_valid)
    y_test, test_label_map = map_labels(y_test)

    print("Training data shape:", X_train.shape, y_train.shape)
    print("Validation data shape:", X_valid.shape, y_valid.shape)
    print("Test data shape:", X_test.shape, y_test.shape)

    # Convert numpy arrays to torch tensors
    print("Converting to torch tensors...")
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.long)
    X_valid_torch = torch.tensor(X_valid, dtype=torch.float32)
    y_valid_torch = torch.tensor(y_valid, dtype=torch.long)

    # Create datasets and loaders
    print("Creating torch datasets and loaders...")
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    valid_dataset = TensorDataset(X_valid_torch, y_valid_torch)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

    # Model initialization
    print("Initializing model...")
    model = Classifier(input_size=1024, num_classes=len(np.unique(y_train)), label_map=train_label_map).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Decays the learning rate by a factor of 0.1 every 10 epochs

    # Directory for saving model checkpoints
    print("Creating output directory...")
    output_dir = '../../output/classifier_3'  # Replace with your actual directory
    os.makedirs(output_dir, exist_ok=True)

    # Start training
    print("Training model...")
    train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=100, output_dir=output_dir)
