import glob
import os

import pandas as pd
from PIL import Image
from pathlib import Path
import open_clip

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any


from typing import Tuple, Any, Dict
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.optim as optim


class BioCLIPModel:
    def __init__(self, model_name, pretrained_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.preprocess_train, self.preprocess_val = self._load_model(model_name, pretrained_path)

    def _load_model(self, model_name, pretrained_path):
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained_path
        )
        return model.to(self.device), preprocess_train, preprocess_val

    def generate_embeddings(self, image_tensors):
        """Generate embeddings for the preprocessed images."""
        with torch.no_grad(), torch.cuda.amp.autocast():
            embeddings = [self.model.encode_image(image_tensor.to(self.device)) for image_tensor in image_tensors]
            return np.stack(embeddings, axis=0).squeeze()




class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_and_preprocess_images(image_paths, preprocess):
    """Load and preprocess images."""
    return [preprocess(Image.open(image).convert("RGB")).unsqueeze(0) \
            for image in image_paths]


def generate_embeddings(model, image_tensors):
    """Generate embeddings for the preprocessed images."""
    with torch.no_grad(), torch.cuda.amp.autocast():
        embeddings = [model.encode_image(image_tensor) for image_tensor in image_tensors]
        return np.stack(embeddings, axis=0).squeeze()


def plot_embeddings(embeddings, labels, markers):
    """Plot embeddings using t-SNE, with markers indicating categories and colors for different groups."""
    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': labels,
        'marker': markers
    })

    # Define mappings for markers to folders and folder color palettes
    marker_to_folder = {'o': 'folder1', '^': 'folder2'}
    folder_color_palette = {'folder1': 'tab20c', 'folder2': 'Reds'}

    plt.figure(figsize=(12, 8))
    df['folder'] = df['marker'].map(marker_to_folder)

    # Plot each folder's embeddings with its unique color palette
    for folder, group_df in df.groupby('folder'):
        color_palette = plt.get_cmap(folder_color_palette[folder])
        unique_labels = group_df['label'].unique()
        num_colors = len(unique_labels)
        colors = color_palette(np.linspace(0.4, 1, num_colors))

        label_color_map = dict(zip(unique_labels, colors))

        for label, color in label_color_map.items():
            subset = group_df[group_df['label'] == label]
            for marker in subset['marker'].unique():
                subset_by_marker = subset[subset['marker'] == marker]
                plt.scatter(subset_by_marker['x'], subset_by_marker['y'], color=color,
                            marker=marker, label=f"{label} ({marker})", s=70)

    # Finalize plot settings
    plt.title('t-SNE visualization of embeddings by Family and Genus')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title='Category (Marker)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: np.ndarray) -> None:
    """
    Plot the confusion matrix using seaborn heatmap.

    Parameters:
    cm (np.ndarray): Confusion matrix.
    class_names (np.ndarray): Array of class names.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()




def get_folders(directory):
    """Get a list of directories within a specified directory."""
    return [f.path for f in os.scandir(directory) if f.is_dir()]


def get_taxon_key_from_df(file_name, df, key='genusKey'):
    if file_name.__contains__('_crop'):
        # reconstruct the original file name
        file_name = file_name.split('_crop')[0] + '.JPG'
    if file_name.__contains__('tile'):
        # reconstruct the original file name
        file_name = file_name.split('_tile')[0] + '.JPG'
    search_row = df[df['fileName'] == file_name]
    search_row = search_row.fillna(-1)
    return int(search_row.iloc[0][key])


def preprocess_data(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, encoded_labels,
        test_size=test_size,
        stratify=encoded_labels,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test, label_encoder

def convert_to_tensor(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def train_downstream_model(X_train: np.ndarray,
                           X_test: np.ndarray,
                           y_train: np.ndarray,
                           y_test: np.ndarray,
                           classifier: Any,
                           label_encoder: LabelEncoder,
    ) -> Tuple[Any, str]:
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Calculate accuracy and report per class
    print("Overall accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report per class:")
    print(classification_report(y_test, y_pred))

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    class_names = np.unique(y_test)
    class_names = label_encoder.inverse_transform(class_names)
    plot_confusion_matrix(cm, class_names)

    return classifier, classification_report(y_test, y_pred)

def train_downstream_model_nn(
        X_train: np.ndarray,
        X_test: np.ndarray,
        Y_train: np.ndarray,
        Y_test: np.ndarray,
        config: Dict[str, Any],
        label_encoder: LabelEncoder,
) -> Tuple[Any, float, str]:
    """
    Train a simple neural network on the provided embeddings and labels.

    Parameters:
    X_train (torch.Tensor): The training embeddings.
    X_test (torch.Tensor): The test embeddings.
    y_train (torch.Tensor): The training labels.
    y_test (torch.Tensor): The test labels.
    config (dict): Configuration dictionary with keys 'test_size', 'random_state',
                   'hidden_dim', 'num_epochs', and 'learning_rate'.

    Returns:
    Tuple[TwoLayerNN, float, str]: Trained model, overall accuracy, and classification report.
    """
    X_train, X_test, Y_train, Y_test = convert_to_tensor(X_train, X_test, Y_train, Y_test)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model, loss function, and optimizer
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(Y_train.cpu().numpy()))
    model = TwoLayerNN(input_dim, config['hidden_dim'], output_dim).to(device)
    criterion = config['loss']
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Move data to device
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)

    best_model = model
    best_acc = 0
    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            # Evaluate the model
            model.eval()
            with torch.no_grad():
                y_pred_nn = model(X_test)
                y_pred_nn = torch.argmax(y_pred_nn, axis=1).cpu().numpy()
                overall_accuracy = accuracy_score(Y_test.cpu().numpy(), y_pred_nn)
                print (f'epoch {epoch}: {overall_accuracy} accuracy')
                if overall_accuracy >= best_acc:
                    best_model = model
                    best_acc = overall_accuracy

    best_model.eval()
    with torch.no_grad():
        y_pred_nn = best_model(X_test)
        y_pred_nn = torch.argmax(y_pred_nn, axis=1).cpu().numpy()
    # Confusion matrix
    cm_nn = confusion_matrix(Y_test.cpu().numpy(), y_pred_nn)
    class_names = np.unique(Y_test.cpu().numpy())
    class_names = label_encoder.inverse_transform(class_names)

    # Plotting the confusion matrix
    plot_confusion_matrix(cm_nn, class_names)

    # Calculate accuracy and report per class
    overall_accuracy = accuracy_score(Y_test.cpu().numpy(), y_pred_nn)
    class_report = classification_report(Y_test.cpu().numpy(), y_pred_nn)

    print("Overall accuracy:", overall_accuracy)
    print("Classification report per class:")
    print(class_report)

    return model, overall_accuracy, class_report




if __name__ == "__main__":

    # visualize clusters for more interpretability
    make_plot = False

    # neural network train config
    config = {
        'test_size': 0.2,
        'random_state': 42,
        'hidden_dim': 512,
        'num_epochs': 400,
        'learning_rate': 0.001,
        'loss': nn.CrossEntropyLoss()

    }

    # Define paths
    dir_path = Path('/Users/daoud/PycharmAssets/xprize/')
    brazil_photos_taxonomy_path = dir_path / 'photos_exif_taxo.csv'

    # File search pattern
    folder_glob_search_str = dir_path / "dji/zoomed_out/cropped/*"
    folders = glob.glob(str(folder_glob_search_str))

    # Load data
    brazil_photos_taxonomy = pd.read_csv(brazil_photos_taxonomy_path)
    filename_and_key = brazil_photos_taxonomy[['fileName', 'familyKey', 'genusKey', 'speciesKey']]

    # Load the model
    model_path = dir_path / "open_clip_pytorch_model.bin"
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name="hf-hub:imageomics/bioclip",
        pretrained=str(model_path)
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Initialization
    all_embeddings, labels, markers = [], [], []

    # Process images to create BioCLIP embeddings, labels
    for idx, folder in enumerate(folders):
        image_paths = glob.glob(str(dir_path / folder / '*'))
        filenames = [os.path.basename(image_path) for image_path in image_paths]
        taxon_keys = {
            key: [get_taxon_key_from_df(filename, filename_and_key, key=key)
                  for filename in filenames]
            for key in ['speciesKey', 'genusKey', 'familyKey']
        }

        preprocessed_images = load_and_preprocess_images(image_paths, preprocess_val)
        embeddings = generate_embeddings(model, preprocessed_images)
        all_embeddings.append(embeddings)
        labels += [f"{family}_{genus}" for family, genus in zip(taxon_keys['familyKey'], taxon_keys['genusKey'])]
        markers += ['o'] * len(taxon_keys['familyKey'])

    # Concatenate all embeddings for t-SNE
    all_embeddings = np.vstack(all_embeddings)
    # todo: save embeddings for lazy loading

    # visualize T-SNE of BioCLIP embeddings to see presence of clusters
    if make_plot:
        plot_embeddings(all_embeddings, labels, markers)

    labels = np.vstack(labels)

    # downstream prediction
    knn = KNeighborsClassifier(n_neighbors=10)
    svc = SVC(kernel='linear')  # You can change the kernel type based on your needs

    X_train, X_test, Y_train, Y_test, label_encoder = preprocess_data(all_embeddings, labels, test_size=config['test_size'], random_state=42)
    model_svc, classification_report_svc = train_downstream_model(X_train, X_test, Y_train, Y_test, classifier=svc, label_encoder=label_encoder)
    model_nn, acc_nn, classification_report_nn = train_downstream_model_nn(X_train, X_test, Y_train, Y_test, config=config, label_encoder=label_encoder)

    #todo: package embedder and trained classifier into one object