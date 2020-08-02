import torch
import numpy as np
import os
import glob


def write_latent_code(array, save_path):
    with open(save_path, 'w') as f:
        np.savetxt(f, array, delimiter=' ', fmt='%f %f')


def read_latent_code(file_path):
    # 第一列是mean, 第二列是log_var
    with open(file_path) as f:
        latent = np.loadtxt(f)
        z_mean = latent[:, 0]
        z_log_var = latent[:, 1]
        return z_mean, z_log_var
