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

    def __init__(self, num_input_channels, mapping_size_spatial=256, mapping_size_contrast = 64, scale=10, contrast_dim=2):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size_spatial
        self._spatial_dims = [0,1]
        self._contrast_dims = [contrast_dim]
        self._B_spatial = torch.randn((num_input_channels-len(self._contrast_dims), mapping_size_spatial)) * scale
        if mapping_size_contrast > 0:
            self._B_contrast = torch.randn((len(self._contrast_dims), mapping_size_contrast)) * scale

    def forward(self, x):
        print(f'size of input to feature transform: {np.shape(x)}')
        x_contrast = x[:,:,self._contrast_dims]
        x_spatial = x[:,:,self._spatial_dims]

        x_spatial = x_spatial @ self._B_spatial.to(x_spatial.device)
        
        if mapping_size_contrast > 0:
            x_contrast = x_contrast @ self._B_contrast.to(x_contrast.device)

            x = torch.cat([x_spatial, x_contrast],dim=2)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=2)