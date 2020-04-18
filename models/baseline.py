import torch
import torch.nn as nn
import torch.nn.functional as F
from .ae_resnet import AEResNet


class Baseline(nn.Module):
    def __init__(self, num_classes=21, ngf=64, n_blocks=6):
        super(Baseline, self).__init__()
        self.num_classes = num_classes

        self.encoder = AEResNet(3, num_classes, ngf=ngf, n_blocks=n_blocks, last_layer='softmax')
        self.decoder = nn.Module()

    def forward(self, x):
        return self.seg(x), None

    def seg(self, x):
        output = self.encoder(x)
        return output
