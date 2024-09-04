import torch

import numpy as np

class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, number_of_coordinates, num_input_channels],
     returns a tensor of size [batches, number_of_coordinates, num_fourier_features].

    One modification: we likely need far fewer features to capture contrast variation than 
    spatial information. So let's use a smaller number of distinct fourier features for that axis    
    """

    def __init__(self, num_input_channels, mapping_size_spatial=256, scale=10 ):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size_spatial
        self._spatial_dims = [0,1]
        self._B_spatial = torch.randn((num_input_channels, mapping_size_spatial)) * scale

    def forward(self, x):
        #print(f'size of input to feature transform: {np.shape(x)}')

        x = x @ self._B_spatial.to(x.device)
        #print(f'Size after transform: {np.shape(x)}')
        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)
    

class InverseFourierFeatureTransform(torch.nn.Module):
    """
    A simple inverse fourier transform for simple 2D cartesian recon. 

    Given an input of size [batches, number_of_coordinates, num_input_channels],
     returns a tensor of size [batches, Nx, Nx].
   
    """

    def __init__(self, num_input_channels, image_resolution = (64, 64)):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._image_resolution = image_resolution

    def forward(self, x):
        #print(f'size of input to feature transform: {np.shape(x)}')

        x = x @ self._B_spatial.to(x.device)
        #print(f'Size after transform: {np.shape(x)}')
        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)