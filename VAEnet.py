import torch
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions
import numpy as np
# import matplotlib.animation as animation
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class VAEencoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.fc = nn.Linear(input_channels, 300)
        self.bn = nn.BatchNorm1d(momentum=0.99, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.3)

        self.mean_fc = nn.Linear(300, 128)
        self.mean_bn = nn.BatchNorm1d(momentum=0.99, eps=0.001)
        self.mean_relu = nn.LeakyReLU(negative_slope=0.3)

        self.logvar_fc = nn.Linear(300, 128)
        self.logvar_bn = nn.BatchNorm1d(momentum=0.99, eps=0.001)
        self.logvar_relu = nn.LeakyReLU(negative_slope=0.3)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.kaiming_normal_(self.mean_fc.weight)
        nn.init.kaiming_normal_(self.logvar_fc.weight)
        self.z_mean = torch.zeros(3, input_channels)
        self.z_log_var = torch.ones(3, input_channels)

    def forward(self, input_coordinates):
        # 每个batch 一个case
        # input N * 3
        x = self.fc(input_coordinates.transpose(0, 1))
        x = self.relu(self.bn(x))
        z_mean = self.mean_relu(self.mean_bn(self.mean_fc(x)))
        z_log_var = self.logvar_relu(self.logvar_bn(self.logvar_fc(x)))
        self.z_mean = z_mean
        self.z_log_var = z_log_var

        epsilon = torch.randn(z_mean.shape[0], 128)
        z = z_mean + torch.exp(z_log_var) * epsilon
        return z


class VAEdecoder(nn.Module):
    def __init__(self, output_channels=128):
        super().__init__()
        self.fc = nn.Linear(128, 300)
        self.bn = nn.BatchNorm1d(momentum=0.99, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.3)

        self.fc2 = nn.Linear(300, output_channels)
        self.bn2 = nn.BatchNorm1d(momentum=0.99, eps=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.3)

        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, latent_code):
        # input 3 * 128
        x = self.fc(latent_code)
        x = self.relu(self.bn(x))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = x.transpose(0,1)
        output = torch.tanh(x)  # N * 3

        return output


class VAEnn(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.encoder = VAEencoder(input_channels)
        self.decoder = VAEdecoder(input_channels)
        self.bceloss = nn.BCELoss(reduction='mean')

    def forward(self, input_coordinates):
        latent_code = self.encoder(input_coordinates)
        z_decoded = self.decoder(latent_code)
        if self.training:
            return self.vae_loss(input_coordinates, z_decoded), latent_code
        else:
            return z_decoded, latent_code

    def vae_loss(self, input_coordinates, z_decoded):
        # input N * 3
        x = torch.flatten(input_coordinates.transpose(0,1))
        z_decoded = torch.flatten(z_decoded.transpose(0,1))
        reconstr_loss = self.bceloss(x, z_decoded)
        # KL-loss
        z_log_var = self.encoder.z_log_var  # 3 * N
        z_mean = self.encoder.z_mean
        latent_loss = -0.5 * torch.mean(1+z_log_var - z_mean**2-torch.exp(z_log_var), dim=-1)  # TODO
        return torch.mean(reconstr_loss+latent_loss)
