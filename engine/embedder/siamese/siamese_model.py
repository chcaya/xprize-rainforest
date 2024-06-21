import math

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    def __init__(self, resnet_model: str, final_embedding_size: int):
        super(SiameseNetwork, self).__init__()
        # Load a ResNet50 model pre-trained on ImageNet
        if resnet_model == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif resnet_model == 'resnet101':
            self.backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif resnet_model == 'resnet152':
            self.backbone = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            raise ValueError(f'Invalid resnet model: {resnet_model}')

        self.final_embedding_size = final_embedding_size

        # Remove the final fully connected layer
        self.backbone.fc = nn.Identity()

        # Adding a new fully connected layer to create embeddings
        self.fc = nn.Sequential(
            nn.Linear(2048, 1536),
            nn.ReLU(),
            nn.Linear(1536, self.final_embedding_size),
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


class SiameseNetwork2(nn.Module):
    def __init__(self,
                 resnet_model: str,
                 final_embedding_size: int,
                 dropout: float):
        super(SiameseNetwork2, self).__init__()
        # Load a ResNet50 model pre-trained on ImageNet
        if resnet_model == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif resnet_model == 'resnet101':
            self.backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif resnet_model == 'resnet152':
            self.backbone = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            raise ValueError(f'Invalid resnet model: {resnet_model}')

        self.final_embedding_size = final_embedding_size

        # Remove the final fully connected layer
        self.backbone.fc = nn.Identity()

        # Adding a new fully connected layer to create embeddings
        self.fc = nn.Sequential(
            nn.Linear(2048 + 4, 1536),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1536, self.final_embedding_size),
        )

    @staticmethod
    def _get_date_encoding(month, day):
        # Sinusoidal encoding for month
        month_sin = torch.sin(2 * math.pi * month / 12).unsqueeze(1)
        month_cos = torch.cos(2 * math.pi * month / 12).unsqueeze(1)

        # Sinusoidal encoding for day
        day_sin = torch.sin(2 * math.pi * day / 31).unsqueeze(1)
        day_cos = torch.cos(2 * math.pi * day / 31).unsqueeze(1)

        # Concatenate in a single tensor of size batch_size x 4
        date_encodings = torch.cat((month_sin, month_cos, day_sin, day_cos), dim=1)

        return date_encodings

    def forward(self, x1, x2, month1, month2, day1, day2):
        # Forward pass for both inputs through the backbone
        output1 = self.backbone(x1)
        output2 = self.backbone(x2)

        # Get the date encoding
        date1_encoding = self._get_date_encoding(month1, day1)
        date2_encoding = self._get_date_encoding(month2, day2)

        # Concatenate the date encoding to the output
        output1 = torch.cat((output1, date1_encoding), dim=1)
        output2 = torch.cat((output2, date2_encoding), dim=1)

        # Pass through the new fully connected layer
        output1 = self.fc(output1)
        output2 = self.fc(output2)

        return output1, output2

    def infer(self, x, month, day):
        output = self.backbone(x)
        date_encoding = self._get_date_encoding(month, day)
        output = torch.cat((output, date_encoding), dim=1)
        output = self.fc(output)
        return output


class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, outputs1, outputs2, labels, margins):
        euclidean_distance = F.pairwise_distance(outputs1, outputs2)

        positive_loss = labels * torch.pow(euclidean_distance, 2)
        negative_loss = (1 - labels) * torch.pow(torch.clamp(margins - euclidean_distance, min=0.0), 2)

        loss_contrastive = torch.mean(positive_loss + negative_loss)
        return loss_contrastive
