import glob
import os

import pandas as pd
import torch
from PIL import Image
from pathlib import Path
import open_clip
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns  # For a nicer confusion matrix visualization
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim


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

def plot_confusion_matrix(confusion_matrix, class_names):
    # Plotting the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
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


def train_downstream_model(embeddings, labels, classifier):
    # knn model
    label_encoder = LabelEncoder()

    # Fit label encoder and return encoded labels
    encoded_labels = label_encoder.fit_transform(labels)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(embeddings, encoded_labels, test_size=0.4, random_state=42)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Calculate accuracy and report per class
    print("Overall accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report per class:")
    print(classification_report(y_test, y_pred))

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    class_names = label_encoder.inverse_transform(np.unique(encoded_labels))
    plot_confusion_matrix(cm, class_names)




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


if __name__ == "__main__":

    # constants
    make_plot = True

    # Define paths
    dir_path = Path('/Users/daoud/PycharmAssets/xprize/')
    brazil_photos_taxonomy_path = dir_path / 'photos_exif_taxo.csv'

    # Load data
    brazil_photos_taxonomy = pd.read_csv(brazil_photos_taxonomy_path)
    filename_and_key = brazil_photos_taxonomy[['fileName', 'familyKey', 'genusKey', 'speciesKey']]

    # File search pattern
    folder_glob_search_str = dir_path / "dji/zoomed_out/cropped/*"
    folders = glob.glob(str(folder_glob_search_str))

    # Load the model
    model_path = dir_path / "open_clip_pytorch_model.bin"
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name="hf-hub:imageomics/bioclip",
        pretrained=str(model_path)
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Initialization
    all_embeddings, labels, markers = [], [], []

    # Process images
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

    if make_plot:
        plot_embeddings(all_embeddings, labels, markers)

    # downstream prediction
    knn = KNeighborsClassifier(n_neighbors=10)
    svc = SVC(kernel='linear')  # You can change the kernel type based on your needs

    train_downstream_model(all_embeddings, labels, classifier=svc)

# # Setup the network
#     input_dim = X_train.shape[1]
#     output_dim = len(np.unique(y_train))
#     model = SimpleNN(input_dim, output_dim).to(device)
#     hidden_dim = 100
#     model = TwoLayerNN(input_dim, hidden_dim, output_dim)
#     # Loss and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#
#     # Convert data to tensors
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float).to(device)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float).to(device)
#
#     # Training loop
#     for epoch in range(100):  # number of epochs can be adjusted
#         print (epoch)
#         optimizer.zero_grad()
#         outputs = model(X_train_tensor)
#         loss = criterion(outputs, y_train_tensor)
#         loss.backward()
#         optimizer.step()
#
#     # Evaluate the model
#     with torch.no_grad():
#         y_pred_nn = model(X_test_tensor)
#         y_pred_nn = torch.argmax(y_pred_nn, axis=1).cpu().numpy()
#
#     # Confusion matrix
#     cm_nn = confusion_matrix(y_test, y_pred_nn)
#     class_names = label_encoder.inverse_transform(np.unique(encoded_labels))
#
#     # Plotting the confusion matrix
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(cm_nn, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
#     plt.title('Confusion Matrix')
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.show()
#
#     # Calculate accuracy and report per class
#     print("Overall accuracy:", accuracy_score(y_test, y_pred_nn))
#     print("Classification report per class:")
#     print(classification_report(y_test, y_pred_nn))
