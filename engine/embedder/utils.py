from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# QPEB = Quebec Panama Equator Brazil
FOREST_QPEB_MEAN = np.array([0.463, 0.537, 0.363])
FOREST_QPEB_STD = np.array([0.207, 0.206, 0.162])


def apply_pca_to_images(embeddings: np.ndarray, pca_model_path: str, n_patches: int, n_features: int):
    n, h, w, d = embeddings.shape
    embeddings_array_flat = embeddings.reshape(-1, embeddings.shape[-1])
    if n_patches > embeddings_array_flat.shape[0]:
        print(f"Warning: n_patches={n_patches} is greater than the number of patches in the dataset."
              f" Reducing to {embeddings_array_flat.shape[0]}.")
        n_patches = embeddings_array_flat.shape[0]
    random_indices = np.random.choice(embeddings_array_flat.shape[0], size=n_patches, replace=False)
    selected_patches = embeddings_array_flat[random_indices]

    print(f"Computing PCA with n_features={n_features} and n_patches={n_patches}...")
    if Path(pca_model_path).exists():
        print(f"Loading PCA model from {pca_model_path}.")
        pca = joblib.load(pca_model_path)
        assert pca.n_components == n_features, \
            f"PCA model has {pca.n_components} components, but config specified {n_features}."
        assert embeddings.shape[-1] == pca.n_features_in_, \
            f"PCA model was trained with {pca.n_features_in_} input features, but input has {embeddings.shape[-1]}."
    else:
        print("Fitting new PCA model...")
        pca = PCA(n_components=n_features)
        pca.fit(selected_patches)
        Path(pca_model_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving new PCA model to {pca_model_path}.")
        joblib.dump(pca, pca_model_path)
    print(f"Done. Reducing dimensionality from {d} to {n_features} for all images...")
    pca_result = pca.transform(embeddings_array_flat)
    pca_result = pca_result.reshape(n, h, w, n_features)
    print('Done.')
    return pca_result
