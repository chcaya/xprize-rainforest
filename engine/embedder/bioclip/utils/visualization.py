import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.cm as cm


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

def visualize_clusters(labeled_features, unlabeled_features, labeled_labels, new_labels, iteration):
    # Reduce dimensionality for visualization
    all_features = np.concatenate([labeled_features, unlabeled_features], axis=0)
    all_labels = np.concatenate([labeled_labels, new_labels], axis=0)
    reduced_features =  TSNE(n_components=2).fit_transform(all_features)

    # Ensure labels are numpy arrays
    labeled_labels = np.array(labeled_labels)
    new_labels = np.array(new_labels)

    # Separate labeled and unlabeled data
    num_labeled = len(labeled_features)
    labeled_reduced = reduced_features[:num_labeled]
    unlabeled_reduced = reduced_features[num_labeled:]

    # Create a color map
    unique_labels = np.unique(np.concatenate((labeled_labels, new_labels)))
    colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    color_map = {label: color for label, color in zip(unique_labels, colors)}

    # Plot the clusters
    plt.figure(figsize=(10, 7))
    for label in np.unique(all_labels):
        labeled_points = labeled_reduced[labeled_labels == label]
        unlabeled_points = unlabeled_reduced[new_labels == label]
        plt.scatter(labeled_points[:, 0], labeled_points[:, 1], s=50, color=color_map[label],
                    label=f'GT Labelled Class {label}: {len(labeled_points)}')
        plt.scatter(unlabeled_points[:, 0], unlabeled_points[:, 1], s=100, facecolors='none',
                    edgecolors=color_map[label],
                    label=f'Unseen point predicted Class {label}: {len(unlabeled_points)}')

    plt.title(f'Clusters after iteration {iteration}, Labelled Points: {len(labeled_labels)}/{len(all_labels)}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def visualize_dbscan_output(features, all_labels):
        # Reduce dimensionality for visualization
        reduced_features = TSNE(n_components=2).fit_transform(features)

        # Plot the clusters
        plt.figure(figsize=(10, 7))
        unique_labels = np.unique(all_labels)
        for label in unique_labels:
            if label == -1:
                # Noise
                plt.scatter(reduced_features[all_labels == label, 0], reduced_features[all_labels == label, 1], s=50,
                            c='k', marker='x', label='Noise')
            else:
                plt.scatter(reduced_features[all_labels == label, 0], reduced_features[all_labels == label, 1], s=50,
                            label=f'Cluster {label}')

        plt.title('DBSCAN Clusters')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.show()
