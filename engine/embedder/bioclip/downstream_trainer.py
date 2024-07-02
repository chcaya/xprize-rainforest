import logging
from typing import Any, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

from two_layer_nn import TwoLayerNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DownstreamModelTrainer:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the DownstreamModelTrainer with configuration parameters."""
        self.config = config

    def preprocess_data(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
        """Preprocess data by encoding labels and splitting into training and test sets."""
        try:
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            X_train, X_test, y_train, y_test = train_test_split(
                embeddings, encoded_labels,
                test_size=test_size,
                stratify=encoded_labels,
                random_state=random_state
            )
            self._check_contamination(X_train, X_test)

            return X_train, X_test, y_train, y_test, label_encoder
        except Exception as e:
            logger.error(f"Error in preprocess_data: {e}")
            raise

    @staticmethod
    def _check_contamination(X_train: np.ndarray, X_test: np.ndarray):
        """Check for contamination between training and test sets."""
        intersection = np.intersect1d(X_train.view([('', X_train.dtype)] * X_train.shape[1]),
                                      X_test.view([('', X_test.dtype)] * X_test.shape[1]))
        if len(intersection) > 0:
            logger.warning(
                f"Data contamination detected: {len(intersection)} overlapping samples found between training and test sets.")

    def train_knn(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Tuple:
        """Train a K-Nearest Neighbors classifier."""
        knn = KNeighborsClassifier(n_neighbors=self.config.get('n_neighbors', 10))
        return self._train_model(knn, X_train, X_test, y_train, y_test)

    def train_svc(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Tuple:
        """Train a Support Vector Classifier."""
        svc = SVC(kernel=self.config.get('kernel', 'linear'))
        return self._train_model(svc, X_train, X_test, y_train, y_test)

    def _train_model(self, model, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Tuple:
        """Train a given model and evaluate it on the test set."""
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"Overall accuracy: {accuracy}")
            logger.info("Classification report per class:")
            logger.info(class_report)
            return model, y_pred, accuracy, class_report, cm
        except Exception as e:
            logger.error(f"Error in _train_model: {e}")
            raise

    def train_nn(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple:
        """Train a simple neural network and return the best model based on validation accuracy."""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_dim = X_train.shape[1]
            output_dim = len(np.unique(y_train))
            model = TwoLayerNN(input_dim, self.config['hidden_dim'], output_dim).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])

            X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = self.convert_to_tensor(X_train, X_test, y_train, y_test)
            X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(device)
            X_test_tensor = X_test_tensor.to(device)

            best_accuracy = 0
            best_model = None

            for epoch in range(self.config['num_epochs']):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        y_pred_nn = model(X_test_tensor)
                        y_pred_nn = torch.argmax(y_pred_nn, axis=1).cpu().numpy()
                        test_loss = criterion(model(X_test_tensor), y_test_tensor).item()
                        accuracy = accuracy_score(y_test, y_pred_nn)
                        logger.info(f"Epoch {epoch + 1}/{self.config['num_epochs']}, Test Loss: {test_loss}, Test Accuracy: {accuracy}")

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = model.state_dict()

            model.load_state_dict(best_model)
            y_pred_nn = model(X_test_tensor)
            y_pred_nn = torch.argmax(y_pred_nn, axis=1).cpu().numpy()
            overall_accuracy = accuracy_score(y_test, y_pred_nn)
            class_report = classification_report(y_test, y_pred_nn)
            cm_nn = confusion_matrix(y_test, y_pred_nn)

            logger.info(f"Best Overall accuracy: {overall_accuracy}")
            logger.info("Best Classification report per class:")
            logger.info(class_report)

            return model, y_pred_nn, overall_accuracy, class_report, cm_nn
        except Exception as e:
            logger.error(f"Error in train_nn: {e}")
            raise

    @staticmethod
    def convert_to_tensor(
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert numpy arrays to PyTorch tensors."""
        try:
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long)
            return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
        except Exception as e:
            logger.error(f"Error in convert_to_tensor: {e}")
            raise

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]):
        """Plot the confusion matrix using seaborn."""
        try:
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
        except Exception as e:
            logger.error(f"Error in plot_confusion_matrix: {e}")
            raise
