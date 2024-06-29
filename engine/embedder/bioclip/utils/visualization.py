import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def plot_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': labels
    })

    plt.figure(figsize=(12, 8))
    unique_labels = df['label'].unique()
    num_colors = len(unique_labels)
    colors = plt.cm.get_cmap('tab20c', num_colors)

    label_color_map = dict(zip(unique_labels, colors.colors))

    for label, color in label_color_map.items():
        subset = df[df['label'] == label]
        plt.scatter(subset['x'], subset['y'], color=color, marker='o', label=label, s=70)

    plt.title('t-SNE visualization of embeddings by Family and Genus')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()