import numpy as np
import torch
import torch.nn as nn

# This module adapted from Schlemper et. all "A Deep Cascade..." 
# https://github.com/js3611/Deep-MRI-Reconstruction

def data_consistency(pred, k0, mask, noise_lvl=None):

    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * pred + mask * (pred + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * pred + mask * k0
    return out


class DataConsistencyInKspace(nn.Module):
    """ Create data consistency operator. All in kspace

    """

    def __init__(self, noise_lvl=None):
        super(DataConsistencyInKspace, self).__init__()
        self.noise_lvl = noise_lvl

    def forward(self, prediction, k0, mask):
        """
        prediction - input in kspace domain, of shape (batchsize, nspatial, dim)
        k0   - initially sampled elements in k-space (batchsize, dim, nx, ny)
        mask - corresponding nonzero location (batchsize, dim, nx, ny)
        """
        batchsize = np.shape(k0)[0]

        # reshape to size (batchsize, nspatial, dim), match prediction
        k0 = k0.view(batchsize,-1, 2)
        mask = mask.view(batchsize,-1, 2)

        out = data_consistency(prediction, k0, mask, self.noise_lvl)

        return out
