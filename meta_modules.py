'''Modules for hypernetwork experiments, Paper Sec. 4.4
'''

import torch
from torch import nn
from collections import OrderedDict
import modules
import numpy as np
import data_consistency

class HyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module):
        '''

        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        '''
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            hn = modules.FCBlock(in_features=hyper_in_features, out_features=int(torch.prod(torch.tensor(param.size()))),
                                 num_hidden_layers=hyper_hidden_layers, hidden_features=hyper_hidden_features,
                                 outermost_linear=True, nonlinearity='relu')
            self.nets.append(hn)

            if 'weight' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
            elif 'bias' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_bias_init(m))

    def forward(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)
        return params


class NeuralProcessImplicit2DHypernet(nn.Module):
    '''A canonical 2D representation hypernetwork mapping 2D coords to out_features.'''
    def __init__(self, in_features, out_features, image_resolution=None, encoder_nl='sine'):
        super().__init__()

        latent_dim = 256 # JBM was 256
        self.hypo_net = modules.SingleBVPNet(out_features=out_features, type='sine', sidelength=image_resolution,
                                             in_features=2)
        self.hyper_net = HyperNetwork(hyper_in_features=latent_dim, hyper_hidden_layers=1, hyper_hidden_features=256,
                                      hypo_module=self.hypo_net)
        self.set_encoder = modules.SetEncoder(in_features=in_features, out_features=latent_dim, num_hidden_layers=2,
                                              hidden_features=latent_dim, nonlinearity=encoder_nl)
        print(self)

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False

    def get_hypo_net_weights(self, model_input):
        pixels, coords = model_input['img_sub'], model_input['coords_sub']
        ctxt_mask = model_input.get('ctxt_mask', None)
        embedding = self.set_encoder(coords, pixels, ctxt_mask=ctxt_mask)
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def forward(self, model_input):
        if model_input.get('embedding', None) is None:
            pixels, coords = model_input['img_sub'], model_input['coords_sub']
            ctxt_mask = model_input.get('ctxt_mask', None)
            embedding = self.set_encoder(coords, pixels, ctxt_mask=ctxt_mask)
        else:
            embedding = model_input['embedding']
        hypo_params = self.hyper_net(embedding)

        model_output = self.hypo_net(model_input, params=hypo_params)
        return {'model_in':model_output['model_in'], 'model_out':model_output['model_out'], 'latent_vec':embedding,
                'hypo_params':hypo_params}
    
class NeuralProcessImplicit2DHypernetFourierFeatures(nn.Module):
    '''A canonical 2D representation hypernetwork mapping 2D coords to out_features.'''
    def __init__(self, in_features, out_features, image_resolution=None, encoder_nl='sine'):
        super().__init__()

        latent_dim = 256 # JBM was 256
        self.hypo_net = modules.SingleBVPNet(out_features=out_features, type='sine', sidelength=image_resolution,
                                             in_features=in_features)
        self.hyper_net = HyperNetwork(hyper_in_features=latent_dim, hyper_hidden_layers=1, hyper_hidden_features=256,
                                      hypo_module=self.hypo_net)
        self.set_encoder = modules.SetEncoder(in_features=in_features, out_features=latent_dim, num_hidden_layers=2,
                                              hidden_features=latent_dim, nonlinearity=encoder_nl)
        print(self)

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False

    def get_hypo_net_weights(self, model_input):
        pixels, coords = model_input['img_sub'], model_input['coords_sub']
        ctxt_mask = model_input.get('ctxt_mask', None)
        embedding = self.set_encoder(coords, pixels, ctxt_mask=ctxt_mask)
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def forward(self, model_input):
        if model_input.get('embedding', None) is None:
            pixels, coords = model_input['img_sub'], model_input['coords_sub']
            ctxt_mask = model_input.get('ctxt_mask', None)
            embedding = self.set_encoder(coords, pixels, ctxt_mask=ctxt_mask)
        else:
            embedding = model_input['embedding']
        hypo_params = self.hyper_net(embedding)

        model_output = self.hypo_net(model_input, params=hypo_params)
        return {'model_in':model_output['model_in'], 'model_out':model_output['model_out'], 'latent_vec':embedding,
                'hypo_params':hypo_params}


class ConvolutionalNeuralProcessImplicit2DHypernet(nn.Module):
    def __init__(self, in_features, out_features, image_resolution=None, partial_conv=False):
        super().__init__()
        latent_dim = 256

        if partial_conv:
            self.encoder = modules.PartialConvImgEncoder(channel=in_features, image_resolution=image_resolution)
        else:
            self.encoder = modules.ConvImgEncoder(channel=in_features, image_resolution=image_resolution)
        self.hypo_net = modules.SingleBVPNet(out_features=out_features, type='sine', sidelength=image_resolution,
                                             in_features=2)
        
        # JBM changed number of hyper hidden layers to 2. originally 1
        self.hyper_net = HyperNetwork(hyper_in_features=latent_dim, hyper_hidden_layers=1, hyper_hidden_features=256,
                                      hypo_module=self.hypo_net)
        print(self)

    def forward(self, model_input):
        if model_input.get('embedding', None) is None:
            embedding = self.encoder(model_input['img_sparse'])
        else:
            embedding = model_input['embedding']
        hypo_params = self.hyper_net(embedding)

        model_output = self.hypo_net(model_input, params=hypo_params)

        return {'model_in': model_output['model_in'], 'model_out': model_output['model_out'], 'latent_vec': embedding,
                'hypo_params': hypo_params}

    def get_hypo_net_weights(self, model_input):
        embedding = self.encoder(model_input['img_sparse'])
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False


class ConvolutionalNeuralProcessImplicit2DHypernetFourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, image_resolution=None, partial_conv=False, 
                 fourier_features_size=512, latent_dim=256, hidden_features=256, 
                 num_hidden_layers=5, hyper_hidden_features=512, hyper_hidden_layers=1, 
                 conv_kernel_size=3, num_conv_res_blocks=4, w0=30):
        super().__init__()

        self.dc = data_consistency.DataConsistencyInKspace(noise_lvl=None)

        if partial_conv:
            self.encoder = modules.PartialConvImgEncoder(channel=2, image_resolution=image_resolution)
        else:
            self.encoder = modules.ConvImgEncoder(channel=2, image_resolution=image_resolution, hidden_size=latent_dim, 
                                                  kernel_size=conv_kernel_size, num_conv_res_blocks=num_conv_res_blocks)
        self.hypo_net = modules.SingleBVPNet(out_features=out_features, type='sine', sidelength=image_resolution,
                                             in_features=fourier_features_size, hidden_features=hidden_features,num_hidden_layers=num_hidden_layers,
                                             w0=w0) # JBM USED TO BE 3 layer, 128 input. good perf with 5
        self.hyper_net = HyperNetwork(hyper_in_features=latent_dim, hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features, # JBM used to be 256 hyperhidden
                                      hypo_module=self.hypo_net)
            # JBM hyper was 1 layer, 256

        print(self)

    def forward(self, model_input):
        if model_input.get('embedding', None) is None:
            # image domain embedding: 
            #img_complex = model_input['img_sparse']
            #print(np.shape(img_complex))
            #img_complex = img_complex[:,0,:,:] + 1j* img_complex[:,1,:,:]
            #img_complex = torch.fft.ifft2(img_complex)
            #img_complex = torch.stack((torch.real(img_complex),torch.imag(img_complex)),dim=1)
            #print(np.shape(img_complex))
            #embedding = self.encoder(img_complex)
            embedding = self.encoder(model_input['img_sparse'])
        else:
            embedding = model_input['embedding']
        hypo_params = self.hyper_net(embedding)

        model_output = self.hypo_net(model_input, params=hypo_params)

        # TODO: What if have no img_sparse because doing latent space interpolation???
        if "img_sparse" in model_input:
            model_output['model_out'] = self.dc(model_output['model_out'], 
                                                model_input['img_sparse'], 
                                                model_input['dc_mask'])
        else:
            print('WARNING: not using DC')


        return {'model_in': model_output['model_in'], 'model_out': model_output['model_out'], 'latent_vec': embedding,
                'hypo_params': hypo_params}

    def get_hypo_net_weights(self, model_input):
        embedding = self.encoder(model_input['img_sparse'])
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False


class ConvolutionalNeuralProcessImplicit2DHypernetModelParallel(nn.Module):
    def __init__(self, in_features, out_features, dev0, dev1, dev2, image_resolution=None, partial_conv=False, 
                 fourier_features_size=512, latent_dim=256, hidden_features=256, 
                 num_hidden_layers=5, hyper_hidden_features=512, hyper_hidden_layers=1, 
                 conv_kernel_size=3, num_conv_res_blocks=4):
        super().__init__()

        self.dev0 = dev0
        self.dev1 = dev1
        self.dev2 = dev2

        self.dc = data_consistency.DataConsistencyInKspace(noise_lvl=None)

        if partial_conv:
            self.encoder = modules.PartialConvImgEncoder(channel=2, image_resolution=image_resolution)
        else:
            self.encoder = modules.ConvImgEncoder(channel=2, image_resolution=image_resolution, hidden_size=latent_dim, 
                                                  kernel_size=conv_kernel_size, num_conv_res_blocks=num_conv_res_blocks)
        self.hypo_net = modules.SingleBVPNet(out_features=out_features, type='sine', sidelength=image_resolution,
                                             in_features=fourier_features_size, hidden_features=hidden_features,num_hidden_layers=num_hidden_layers) # JBM USED TO BE 3 layer, 128 input. good perf with 5
        self.hyper_net = HyperNetwork(hyper_in_features=latent_dim, hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features, # JBM used to be 256 hyperhidden
                                      hypo_module=self.hypo_net)
        
        # model parallel setup
        self.encoder.to(dev0)
        self.hypo_net.to(dev1)
        self.hyper_net.to(dev2)

        print(self)

    def forward(self, model_input):
        if model_input.get('embedding', None) is None:
            embedding = self.encoder(model_input['img_sparse'])
        else:
            embedding = model_input['embedding']
        hypo_params = self.hyper_net(embedding)

        model_output = self.hypo_net(model_input, params=hypo_params)

        # TODO: What if have no img_sparse because doing latent space interpolation???
        if "img_sparse" in model_input:
            model_output['model_out'] = self.dc(model_output['model_out'], 
                                                model_input['img_sparse'], 
                                                model_input['dc_mask'])
        else:
            print('WARNING: not using DC')


        return {'model_in': model_output['model_in'], 'model_out': model_output['model_out'], 'latent_vec': embedding,
                'hypo_params': hypo_params}

    def get_hypo_net_weights(self, model_input):
        embedding = self.encoder(model_input['img_sparse'])
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

############################
# Initialization schemes
def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)


def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1/fan_in, 1/fan_in)
