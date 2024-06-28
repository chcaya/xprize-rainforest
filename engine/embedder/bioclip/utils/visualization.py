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

def plot_embeddings(embeddings, labels, markers):
    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': labels,
        'marker': markers
    })
    marker_to_folder = {'o': 'folder1', '^': 'folder2'}
    folder_color_palette = {'folder1': 'tab20c', 'folder2': 'Reds'}
    plt.figure(figsize=(12, 8))
    df['folder'] = df['marker'].map(marker_to_folder)
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
    plt.title('t-SNE visualization of embeddings by Family and Genus')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(title='Category (Marker)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()
