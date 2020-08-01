import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import warnings
warnings.filterwarnings('ignore')


class PrimaryPointCapsLayer(nn.Module):
    def __init__(self, prim_vec_size=16, num_points=2048):
        super(PrimaryPointCapsLayer, self).__init__()
        self.capsules = nn.ModuleList([
            torch.nn.Sequential(OrderedDict([
                ('conv3', torch.nn.Conv1d(128, 1024, 1)),
                ('bn3', nn.BatchNorm1d(1024)),
                ('mp1', torch.nn.MaxPool1d(num_points)),
            ]))
            for _ in range(prim_vec_size)])

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.cat(u, dim=2)
        # return self.squash(u.squeeze())
        return u

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / \
                        ((1. + squared_norm) * torch.sqrt(squared_norm))
        if (output_tensor.dim() == 2):
            output_tensor = torch.unsqueeze(output_tensor, 0)
        return output_tensor