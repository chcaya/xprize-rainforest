import glob
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np

from bioclip_model import BioCLIPModel
from downstream_trainer import DownstreamModelTrainer
from utils.data_utils import get_taxon_key_from_df
from utils.visualization import plot_confusion_matrix, plot_embeddings

import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    config_path = 'config.yaml'
    config = load_config(config_path)

    bioclip_model = BioCLIPModel(config['model_name'], config['pretrained_path'])
    trainer = DownstreamModelTrainer(config)

    dir_path = Path('/Users/daoud/PycharmAssets/xprize/')
    brazil_photos_taxonomy_path = dir_path / 'photos_exif_taxo.csv'
    folder_glob_search_str = dir_path / "dji/zoomed_out/cropped/*"
    folders = glob.glob(str(folder_glob_search_str))
    brazil_photos_taxonomy = pd.read_csv(brazil_photos_taxonomy_path)
    filename_and_key = brazil_photos_taxonomy[['fileName', 'familyKey', 'genusKey', 'speciesKey']]
    all_embeddings, labels, markers = [], [], []

    for idx, folder in enumerate(folders[:20]):
        image_paths = glob.glob(str(Path(folder) / '*'))
        filenames = [os.path.basename(image_path) for image_path in image_paths]
        taxon_keys = {
            key: [get_taxon_key_from_df(filename, filename_and_key, key=key)
                  for filename in filenames]
            for key in ['speciesKey', 'genusKey', 'familyKey']
        }
        preprocessed_images_train = [bioclip_model.preprocess_train(Image.open(image).convert("RGB")).unsqueeze(0) for image in image_paths]
        embeddings_train = bioclip_model.generate_embeddings(preprocessed_images_train)
        all_embeddings.append(embeddings_train)
        labels += [f"{family}_{genus}" for family, genus in zip(taxon_keys['familyKey'], taxon_keys['genusKey'])]
        markers += ['o'] * len(taxon_keys['familyKey'])

    all_embeddings = np.vstack(all_embeddings)

    # Visualize embeddings if flag is set
    visualize_embeddings = True  # Set this flag as needed
    if visualize_embeddings:
        plot_embeddings(all_embeddings, labels, markers)


    labels = np.array(labels)



    X_train, X_test, y_train, y_test, label_encoder = trainer.preprocess_data(all_embeddings, labels, test_size=config['test_size'], random_state=config['random_state'])

    #
    # preprocessed_images_test = [bioclip_model.preprocess_val(Image.open(image).convert("RGB")).unsqueeze(0) for image in X_test]
    # embeddings_test = bioclip_model.generate_embeddings(preprocessed_images_test)

    model_svc, classifier_preds, svc_accuracy, svc_report, cm_svc = trainer.train_svc(X_train, X_test, y_train, y_test)
    model_nn, nn_preds, nn_accuracy, nn_report, cm_nn = trainer.train_nn(X_train, X_test, y_train, y_test)

    class_names = label_encoder.inverse_transform(np.unique(y_test))
    trainer.plot_confusion_matrix(cm_svc, class_names)
    trainer.plot_confusion_matrix(cm_nn, class_names)

