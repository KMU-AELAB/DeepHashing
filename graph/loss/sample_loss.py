import torch
import numpy as np
import torch.nn as nn


class CodeLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()

    def forward(self, origin_code, trans_code, origin_feature, trans_feature, weight=0.001):
        code_similar = torch.mean(torch.sum((origin_code != trans_code).float(), dim=1))

        feature_similar = (self.loss(origin_feature, origin_code) + self.loss(trans_feature, trans_code)) / 2
        return code_similar + weight * feature_similar


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()

    def forward(self, recon, target):
        return self.loss(recon, target)
