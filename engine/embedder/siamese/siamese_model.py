import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class SiameseNetwork2(nn.Module):
    def __init__(self):
        super(SiameseNetwork2, self).__init__()
        # Load a ResNet50 model pre-trained on ImageNet
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)            # TODO try bigger models as backbone!!!
        # Remove the final fully connected layer
        self.backbone.fc = nn.Identity()

        # Adding a new fully connected layer to create embeddings
        self.fc = nn.Sequential(
            nn.Linear(2048, 1536),
            nn.ReLU(),
            nn.Linear(1536, 1024),
        )

    def forward(self, x1, x2):
        # Forward pass for both inputs through the backbone
        output1 = self.backbone(x1)
        output2 = self.backbone(x2)

        # Pass through the new fully connected layer
        output1 = self.fc(output1)
        output2 = self.fc(output2)

        return output1, output2

    def infer(self, x):
        output = self.backbone(x)
        output = self.fc(output)
        return output


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin        # TODO this is currently fixed and not using the value coming from the dataset

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)

        positive_loss = label * torch.pow(euclidean_distance, 2)
        negative_loss = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

        loss_contrastive = torch.mean(positive_loss + negative_loss)
        return loss_contrastive
