import torch
from torch import nn

# Define the custom neural network
class Dino_vits16_100(nn.Module):
    def __init__(self):
        super(Dino_vits16_100, self).__init__()
        # Define layers of the neural network
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        self.classifier = nn.Linear(self.backbone.embed_dim, 100)

    def forward(self, x):
        # Define forward pass
        x = self.backbone(x)
        x = self.classifier(x)
        return x