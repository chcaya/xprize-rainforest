import math

import torch
from torch import nn
from torchvision import models


class XPrizeTreeEmbedder(nn.Module):
    def __init__(self,
                 resnet_model: str,
                 final_embedding_size: int,
                 dropout: float):
        super(XPrizeTreeEmbedder, self).__init__()
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

    def forward(self, x, month, day):
        output = self.backbone(x)
        date_encoding = self._get_date_encoding(month, day)
        output = torch.cat((output, date_encoding), dim=1)
        output = self.fc(output)
        return output
