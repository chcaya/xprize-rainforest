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
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Constants
MODEL_NAME = "hf-hub:imageomics/bioclip"
MODEL_PATH = "/Users/daoud/PycharmAssets/xprize/open_clip_pytorch_model.bin"
DATA_PATH = "/Users/daoud/PycharmAssets/xprize/images/"


def load_and_preprocess_images(image_paths, preprocess):
    return [preprocess(Image.open(img).convert("RGB")).unsqueeze(0) for img in image_paths]


def generate_embeddings(model, image_tensors):
    with torch.no_grad(), torch.cuda.amp.autocast():
        return np.stack([model.encode_image(tensor) for tensor in image_tensors]).squeeze()


def train_knn_classifier(X_train, y_train, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn


def perform_knn_operations(X_train, y_train, test_data, preprocess, label_dict, model):
    knn = train_knn_classifier(np.array(X_train), np.array(y_train))
    predictions = []
    for folder, preprocessed_images in test_data:
        embeddings = generate_embeddings(model, preprocessed_images)
        pred = knn.predict(embeddings)
        predicted_labels = [list(label_dict.keys())[list(label_dict.values()).index(p)] for p in pred]
        predictions.append((folder, predicted_labels))
    return predictions


def plot_embeddings(embeddings, labels, perplexity=5):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    plt.figure(figsize=(12, 8))
    for label in set(labels):
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=label)
    plt.title('t-SNE visualization of embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in np.ndenumerate(cm):
        plt.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]:d}",
                 horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def get_folders(directory):
    return [f.path for f in os.scandir(directory) if f.is_dir() and any(os.scandir(f.path))]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=MODEL_PATH)
    model = model.to(device)

    training_folders = get_folders(DATA_PATH + "train")
    test_folders = get_folders(DATA_PATH + "test")
    label_dict = {os.path.basename(folder): idx for idx, folder in enumerate(training_folders)}

    X_train, y_train = [], []
    for folder in training_folders:
        image_paths = glob.glob(os.path.join(folder, '*'))
        preprocessed_images = load_and_preprocess_images(image_paths, preprocess_train)
        embeddings = generate_embeddings(model, preprocessed_images)
        X_train.extend(embeddings)
        y_train.extend([label_dict[os.path.basename(folder)] for _ in range(len(embeddings))])

    test_data = []
    for folder in test_folders:
        image_paths = glob.glob(os.path.join(folder, '*'))
        preprocessed_images = load_and_preprocess_images(image_paths, preprocess_val)
        test_data.append((folder, preprocessed_images))

    predictions = perform_knn_operations(X_train, y_train, test_data, preprocess_val, label_dict, model)
    for folder, predicted_labels in predictions:
        print(f"Predictions for {folder}: {predicted_labels}")


if __name__ == "__main__":
    main()
