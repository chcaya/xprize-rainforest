import glob
from bioclip import TreeOfLifeClassifier, Rank

import torch
from PIL import Image
from pathlib import Path
import open_clip
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE



def plot_embeddings(embeddings1, embeddings2):
    """Plot two sets of embeddings using t-SNE."""
    # Combine both sets of embeddings
    combined_embeddings = np.vstack((embeddings1, embeddings2))

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    reduced_embeddings = tsne.fit_transform(combined_embeddings)

    # Split the reduced embeddings back into two sets
    reduced_embeddings1 = reduced_embeddings[:len(embeddings1)]
    reduced_embeddings2 = reduced_embeddings[len(embeddings1):]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_embeddings1[:, 0], reduced_embeddings1[:, 1], color='blue', label='Set 1')
    plt.scatter(reduced_embeddings2[:, 0], reduced_embeddings2[:, 1], color='red', label='Set 2')
    plt.title('t-SNE visualization of embeddings')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":

    dir_path = Path('/Users/daoud/PycharmAssets/xprize/')
    image_set_1_folder = "images/8802_2884049_7485884"
    image_set_2_folder = "images/3231623_7288308_4030655"

    image_set_1 = glob.glob(str(dir_path/image_set_1_folder) + '/*')
    image_set_2 = glob.glob(str(dir_path/image_set_2_folder) + '/*')
    print ('loading model')
    model_path = dir_path / "open_clip_pytorch_model.bin"

    # Load the model
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name="hf-hub:imageomics/bioclip",  # This would need to align with what open_clip expects
        pretrained= str(model_path)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)


    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')

    preprocessed_set_1 = [preprocess_val(Image.open(image)).unsqueeze(0)\
                          for image in image_set_1]
    preprocessed_set_2 = [preprocess_val(Image.open(image)).unsqueeze(0)\
                          for image in image_set_2]

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features_1 = [model.encode_image(image) \
                            for image in preprocessed_set_1]
        image_features_2 = [model.encode_image(image) \
                            for image in preprocessed_set_2]

        image_features_1 = np.stack(image_features_1, axis=0).squeeze()
        image_features_2 = np.stack(image_features_2, axis=0).squeeze()

        print("Embeddings 1 shape:", image_features_1.shape)
        print("Embeddings 2 shape:", image_features_2.shape)

        plot_embeddings(image_features_1, image_features_2)