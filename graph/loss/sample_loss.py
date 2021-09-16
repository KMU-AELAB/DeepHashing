import torch
import numpy as np
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()

    def forward(self, origin_code, trans_code, origin_feature, trans_feature, weight1=0.1, weight2=0.001):
        code_similar = torch.mean(torch.sum((trans_code != origin_code).float(), dim=1))

        feature_similar = (self.loss(origin_feature, origin_code.detach) +
                           self(trans_feature, trans_code.detach())) / 2
        return code_similar + weight2 * feature_similar
