import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from tqdm import tqdm


class SiameseResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(2048 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)
        combined = torch.cat((f1, f2), dim=1)
        return self.fc(combined)


class SiameseBasePipeline:
    def __init__(self, batch_size, architecture, checkpoint_state_dict_path, pretrained=True):
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SiameseResNet(pretrained=pretrained).to(self.device)

        if checkpoint_state_dict_path:
            self.model.load_state_dict(torch.load(checkpoint_state_dict_path))

        self.model.to(self.device)


class SiameseTrainPipeline(SiameseBasePipeline):
    def __init__(self, batch_size, architecture, checkpoint_state_dict_path, learning_rate, num_epochs,
                 pretrained=True):
        super().__init__(batch_size, architecture, checkpoint_state_dict_path, pretrained)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()

    def _train_one_epoch(self, data_loader, epoch):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (img1, img2, labels) in enumerate(data_loader):
            img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(img1, img2)
            loss = self.criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(data_loader)}')

    def train(self, train_loader, valid_loader=None):
        for epoch in range(self.num_epochs):
            self._train_one_epoch(train_loader, epoch)
            if valid_loader:
                self._evaluate(valid_loader, epoch)  # Implement this method if needed for validation


class SiameseScorePipeline(SiameseBasePipeline):
    def __init__(self, batch_size, architecture, checkpoint_state_dict_path, pretrained=True):
        super().__init__(batch_size, architecture, checkpoint_state_dict_path, pretrained)

    def _evaluate(self, data_loader, epoch=None):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (img1, img2, labels) in enumerate(tqdm(data_loader, desc="Evaluating", leave=True)):
                img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                outputs = self.model(img1, img2).squeeze()
                predicted = torch.round(outputs)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        print(f'Accuracy: {accuracy}%')
        return accuracy

    def score(self, test_loader):
        accuracy = self._evaluate(test_loader)
        print(f"Test Accuracy: {accuracy}%")


class SiameseInferencePipeline(SiameseBasePipeline):
    def __init__(self, batch_size, architecture, checkpoint_state_dict_path, pretrained=True):
        super().__init__(batch_size, architecture, checkpoint_state_dict_path, pretrained)

    def _infer(self, data_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for images in tqdm(data_loader, desc="Inferring", leave=True):
                img1, img2 = images[0].to(self.device), images[1].to(self.device)
                output = self.model(img1, img2).squeeze()
                predictions.append(torch.round(output).cpu().numpy())
        return predictions

    def infer(self, infer_loader):
        results = self._infer(infer_loader)
        return results
