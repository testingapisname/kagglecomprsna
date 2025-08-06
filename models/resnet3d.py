import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18

class ResNet3D(nn.Module):
    def __init__(self, num_outputs=14, pretrained=False):
        super().__init__()
        self.base = r3d_18(pretrained=pretrained)

        # Replace the final classification layer
        in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, num_outputs)

    def forward(self, x):
        return self.base(x)
