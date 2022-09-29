import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as T

transforms = T.Compose([
    T.ConvertImageDtype(torch.float)
])

class ResNet18Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = transforms
        self.encoder = resnet18(weights=None)

    def forward(self, x):
        self.features = []

        x = self.transforms(x)

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        self.features.append(x)

        x = self.encoder.layer1(x)
        self.features.append(x)

        x = self.encoder.layer2(x)
        self.features.append(x)

        x = self.encoder.layer3(x)
        self.features.append(x)

        x = self.encoder.layer4(x)
        self.features.append(x)

        x = self.encoder.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.encoder.fc(x)
        self.features.append(x)

        return self.features
