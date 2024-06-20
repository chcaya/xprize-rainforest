import glob
import os
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


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in np.ndenumerate(cm):
        plt.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]:d}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



def load_and_preprocess_images(image_paths, preprocess):
    """Load and preprocess images."""
    return [preprocess(Image.open(image).convert("RGB")).unsqueeze(0) for image in image_paths]

def generate_embeddings(model, image_tensors):
    """Generate embeddings for the preprocessed images."""
    with torch.no_grad(), torch.cuda.amp.autocast():
        embeddings = [model.encode_image(image_tensor) for image_tensor in image_tensors]
        return np.stack(embeddings, axis=0).squeeze()

def train_knn_classifier(X_train, y_train):
    """Train a KNN classifier."""
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    return knn

def plot_embeddings(embeddings, labels):
    """Plot embeddings using t-SNE with different colors for each set."""
    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)  # Adjust perplexity appropriately
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 8))
    unique_labels = set(labels)
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=label)
    plt.title('t-SNE visualization of embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

def get_folders(directory):
    """Get a list of directories within a specified directory."""
    return [f.path for f in os.scandir(directory) if f.is_dir()]


if __name__ == "__main__":
    dir_path = Path('/Users/daoud/PycharmAssets/xprize/')
    folders = ["images/8802_2884049_7485884",
               "images/3231623_7288308_4030655",
               "images/6646_3189556_3713863",
               "images/7681_2738048_2738136",
               "images/7681_2738048_2738133",
               "images/DJI_8802",
               "images/DJI_6646",
               "images/DJI_3231623",
               ]  # Add more folders as needed

    # Load the model
    model_path = dir_path / "open_clip_pytorch_model.bin"
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name="hf-hub:imageomics/bioclip",
        pretrained=str(model_path)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    all_embeddings = []
    labels = []

    make_plot = False
    if make_plot:
        for folder in folders:
            image_paths = glob.glob(str(dir_path / folder) + '/*')
            preprocessed_images = load_and_preprocess_images(image_paths, preprocess_val)
            embeddings = generate_embeddings(model, preprocessed_images)
            all_embeddings.append(embeddings)
            labels += [folder.split('/')[-1]] * len(embeddings)  # Use the folder name as the label

        # Concatenate all embeddings for t-SNE
        all_embeddings = np.vstack(all_embeddings)
        plot_embeddings(all_embeddings, labels)

    training_folders = ["images/8802_2884049_7485884",
                        "images/3231623_7288308_4030655",
                        "images/7681_2738048_2738136",
                        "images/7681_2738048_2738133",
                        "images/6646_3189556_3713863"]
    test_folders = ["images/DJI_8802", "images/DJI_3231623", "images/DJI_6646"]

    # Training data
    X_train, y_train = [], []
    label_dict = {folder.split('/')[-1]: idx for idx, folder in enumerate(training_folders)}

    for folder in training_folders:
        image_paths = glob.glob(str(dir_path / folder) + '/*')
        preprocessed_images = load_and_preprocess_images(image_paths, preprocess_val)
        embeddings = generate_embeddings(model, preprocessed_images)
        X_train.extend(embeddings)
        y_train.extend([label_dict[folder.split('/')[-1]] for _ in range(len(embeddings))])

    knn = train_knn_classifier(np.array(X_train), np.array(y_train))

    # Testing data
    for folder in test_folders:
        image_paths = glob.glob(str(dir_path / folder) + '/*')
        preprocessed_images = load_and_preprocess_images(image_paths, preprocess_val)
        embeddings = generate_embeddings(model, preprocessed_images)
        predictions = knn.predict(embeddings)
        predicted_labels = [list(label_dict.keys())[list(label_dict.values()).index(pred)] for pred in predictions]
        print(f"Predictions for {folder}: {predicted_labels}")
