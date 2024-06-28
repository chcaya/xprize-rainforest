import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DownstreamModelTrainer:
    def __init__(self, config: dict):
        self.config = config

    def preprocess_data(self, embeddings, labels, test_size=0.2, random_state=42):
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, encoded_labels,
            test_size=test_size,
            stratify=encoded_labels,
            random_state=random_state
        )
        return X_train, X_test, y_train, y_test, label_encoder

    def train_knn(self, X_train, X_test, y_train, y_test):
        knn = KNeighborsClassifier(n_neighbors=self.config.get('n_neighbors', 10))
        return self._train_model(knn, X_train, X_test, y_train, y_test)

    def train_svc(self, X_train, X_test, y_train, y_test):
        svc = SVC(kernel=self.config.get('kernel', 'linear'))
        return self._train_model(svc, X_train, X_test, y_train, y_test)

    def _train_model(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        print(f"Overall accuracy: {accuracy}")
        print("Classification report per class:")
        print(class_report)
        cm = confusion_matrix(y_test, y_pred)
        return model, y_pred, accuracy, class_report, cm

    def train_nn(self, X_train, X_test, y_train, y_test):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = X_train.shape[1]
        output_dim = len(np.unique(y_train))
        model = TwoLayerNN(input_dim, self.config['hidden_dim'], output_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = self.convert_to_tensor(X_train, X_test, y_train, y_test)
        X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(device)
        X_test_tensor = X_test_tensor.to(device)

        for epoch in range(self.config['num_epochs']):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred_nn = model(X_test_tensor)
            y_pred_nn = torch.argmax(y_pred_nn, axis=1).cpu().numpy()

        overall_accuracy = accuracy_score(y_test, y_pred_nn)
        class_report = classification_report(y_test, y_pred_nn)
        print("Overall accuracy:", overall_accuracy)
        print("Classification report per class:")
        print(class_report)
        cm_nn = confusion_matrix(y_test, y_pred_nn)
        return model, y_pred_nn, overall_accuracy, class_report, cm_nn

    @staticmethod
    def convert_to_tensor(X_train, X_test, y_train, y_test):
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

    def plot_confusion_matrix(self, cm, class_names):
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
