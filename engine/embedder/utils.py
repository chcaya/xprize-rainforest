import numpy as np
from sklearn.decomposition import PCA


def apply_pca_to_images(embeddings: np.ndarray, n_patches: int, n_features: int):
    n, h, w, d = embeddings.shape
    embeddings_array_flat = embeddings.reshape(-1, embeddings.shape[-1])
    if n_patches > embeddings_array_flat.shape[0]:
        print(f"Warning: n_patches={n_patches} is greater than the number of patches in the dataset."
              f" Reducing to {embeddings_array_flat.shape[0]}.")
        n_patches = embeddings_array_flat.shape[0]
    random_indices = np.random.choice(embeddings_array_flat.shape[0], size=n_patches, replace=False)
    selected_patches = embeddings_array_flat[random_indices]

    pca = PCA(n_components=n_features)
    print(f"Computing PCA with n_features={n_features} and n_patches={n_patches}...")
    pca.fit(selected_patches)
    print(f"Done. Reducing dimensionality from {d} to {n_features} for all images...")
    pca_result = pca.transform(embeddings_array_flat)
    pca_result = pca_result.reshape(n, h, w, n_features)
    print('Done.')
    return pca_result
