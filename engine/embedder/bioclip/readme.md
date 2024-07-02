# BioCLIP Model Training and Inference


## Model Weights

Please download model weights from the drive link below:

`link/to/drive`


## Usage

### Configuration

Modify the `config.yaml` file to set the required parameters for model training and inference.

### Running the Pipeline

You can run the entire pipeline using:

```bash
python bioclip_trainer.py
```

### Visualizing Embeddings

To visualize the embeddings using t-SNE, ensure the `visualize_embeddings` flag is set to `True` in `run_bioclip_trainer.py`.

## Directory Structure

- `config.yaml`: Configuration files
- `bioclip_model.py`: Defines the BioCLIP model.
- `downstream_model_trainer.py`: Handles the training of downstream models (KNN, SVC, and NN) and provides methods for preprocessing data, training models, and plotting confusion matrices.
- `two_layer_nn.py`: Defines a simple two-layer neural network.
- `utils/`: Utility functions
  - `data_utils.py`: Helper functions for data processing.
  - `visualization.py`: Functions for plotting embeddings(to help visualize clusters) and confusion matrices.
- `run_bioclip_trainer.py`: Main script to run the pipeline

