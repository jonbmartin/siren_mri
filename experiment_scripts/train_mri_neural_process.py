# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions

import torch
from torch.utils.data import DataLoader
import configargparse
from functools import partial
from features import GaussianFourierFeatureTransform
import torch.nn as nn


p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=100)
p.add_argument('--lr', type=float, default=5e-5, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=401,
               help='Number of epochs to train for.')
p.add_argument('--kl_weight', type=float, default=1e-1,
               help='Weight for l2 loss term on code vectors z (lambda_latent in paper).')
p.add_argument('--fw_weight', type=float, default=1e2,
               help='Weight for the l2 loss term on the weights of the sine network')
p.add_argument('--train_sparsity_range', type=int, nargs='+', default=[2000, 4000],
               help='Two integers: lowest number of sparse pixels sampled followed by highest number of sparse'
                    'pixels sampled when training the conditional neural process')

p.add_argument('--epochs_til_ckpt', type=int, default=10,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--dataset', type=str, default='mri_image',
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model_type', type=str, default='sine',
               help='Nonlinearity for the hypo-network module')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

p.add_argument('--conv_encoder', action='store_true', default=False, help='Use convolutional encoder process')
opt = p.parse_args()

# JBM OVERRIDE TO A CONV_CNP
opt.conv_encoder = True
assert opt.dataset == 'mri_image'
if opt.conv_encoder: gmode = 'conv_cnp'
else: gmode = 'cnp'

# CONFIG. TODO: transition to config.yml
config = 'hand_tuned_manual'
if config=='default_manual':
    num_fourier_features = 30
    kl_weight = 0 # Not assuming anything about the weights of the latent 
    fw_weight = 2.7e-8
    lr = 1e-5 # reduced from default of 5e-5
    fourier_features_scale = 24
    latent_dim = 256
    hidden_features_hyper = 512
    hidden_layers_hyper = 1
    hidden_layers = 5
    hidden_features = 256
if config=='hand_tuned_manual':
    num_fourier_features = 60
    kl_weight = 1.07e-9 # Not assuming anything about the weights of the latent 
    fw_weight = 1.11e-7
    lr = 6e-5 # reduced from default of 5e-5
    fourier_features_scale = 19
    latent_dim = 256
    hidden_features_hyper = 512
    hidden_layers_hyper = 1
    hidden_layers = 3
    hidden_features = 256
    partial_conv = False
if config=='hyperoptII':
    num_fourier_features = 128
    kl_weight = 1.07e-9 # Not assuming anything about the weights of the latent 
    fw_weight = 1.11e-7
    lr = 6e-5 # reduced from default of 5e-5
    fourier_features_scale = 16
    latent_dim = 256
    hidden_features_hyper = 512
    hidden_layers_hyper = 2
    hidden_layers = 3
    hidden_features = 512
    partial_conv = False
elif config =='hyperopt':
    num_fourier_features = 128
    kl_weight = 0 
    fw_weight = 1.85e-7
    lr = 1.9e-4 
    fourier_features_scale = 19
    latent_dim = 256
    hidden_features_hyper = 512
    hidden_layers_hyper = 1
    hidden_layers = 2
    hidden_features = 64

image_resolution = (128, 128)
use_fourier_features = True
img_dataset = dataio.FastMRIBrainKspace(split='train', downsampled=True, image_resolution=image_resolution)
#img_dataset = dataio.FastMRIBrain(split='train', downsampled=True, image_resolution=image_resolution)
#img_dataset = dataio.MRIImageDomain(split='train',downsample=True)
coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=image_resolution, image=False)

# TODO: right now, test_sparsity= ... overwrites train sparsity for training. using this to get CS
device = torch.device('cuda:0')  # or whatever device/cpu you like
generalization_dataset = dataio.ImageGeneralizationWrapper(coord_dataset,
                                                           train_sparsity_range=opt.train_sparsity_range,
                                                           test_sparsity= 'CS_cartesian',
                                                           generalization_mode=gmode,
                                                           device=device)

dataloader = DataLoader(generalization_dataset, shuffle=True, batch_size=opt.batch_size,
                         pin_memory=False, num_workers=0,)

# VAL DATASET
img_dataset_val = dataio.FastMRIBrainKspace(split='val_small', downsampled=True, image_resolution=image_resolution)
coord_dataset_val = dataio.Implicit2DWrapper(img_dataset_val, sidelength=image_resolution, image=False)
generalization_dataset_val = dataio.ImageGeneralizationWrapper(coord_dataset_val,
                                                           train_sparsity_range=opt.train_sparsity_range,
                                                           test_sparsity= 'CS_cartesian',
                                                           generalization_mode=gmode,
                                                           device=device)
dataloader_val = DataLoader(generalization_dataset_val, shuffle=True, batch_size=opt.batch_size,
                         pin_memory=False, num_workers=0,)

if opt.conv_encoder:
    if use_fourier_features:
        model = meta_modules.ConvolutionalNeuralProcessImplicit2DHypernetFourierFeatures(in_features=2*num_fourier_features,
                                                                out_features=img_dataset.img_channels,
                                                                image_resolution=image_resolution,
                                                                fourier_features_size=2*num_fourier_features,
                                                                latent_dim=latent_dim,
                                                                hidden_features=hidden_features,
                                                                hyper_hidden_features=hidden_features_hyper,
                                                                hyper_hidden_layers=hidden_layers_hyper,
                                                                num_hidden_layers=hidden_layers,
                                                                device=device,
                                                                partial_conv=partial_conv)
    else:
        model = meta_modules.ConvolutionalNeuralProcessImplicit2DHypernet(in_features=img_dataset.img_channels,
                                                                        out_features=img_dataset.img_channels,
                                                                        image_resolution=image_resolution)
else:
    if use_fourier_features:
        # TODO: this does not work yet
        model = meta_modules.NeuralProcessImplicit2DHypernetFourierFeatures(in_features=2*num_fourier_features,
                                                    out_features=img_dataset.img_channels,
                                                    image_resolution=image_resolution)
    else:
        model = meta_modules.NeuralProcessImplicit2DHypernet(in_features=img_dataset.img_channels + 2,
                                                            out_features=img_dataset.img_channels,
                                                            image_resolution=image_resolution)

#model.cuda(device)
# trying data parallel
model = nn.DataParallel(model)
model.to(device)

# Define the loss
#loss_fn = partial(loss_functions.image_hypernetwork_log_loss, None, kl_weight, fw_weight)
loss_fn = partial(loss_functions.image_hypernetwork_loss, None, kl_weight, fw_weight)
#loss_fn = partial(loss_functions.image_hypernetwork_ift_loss, None, kl_weight, fw_weight)
summary_fn = partial(utils.write_image_summary_small, image_resolution, None)

root_path = os.path.join(opt.logging_root, opt.experiment_name)

fourier_transformer = GaussianFourierFeatureTransform(num_input_channels=2, mapping_size_spatial=num_fourier_features, 
                                                      scale=fourier_features_scale, device=device)

# Record the fourier feature transform matrix
fourier_transformer.save_B('current_B.pt')

training.train(model=model, train_dataloader=dataloader,val_dataloader=dataloader_val, epochs=opt.num_epochs,
            lr=lr, steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
            model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, clip_grad=True,
            fourier_feat_transformer=fourier_transformer, device=device, accumulation_steps=4)

# training.train_ddp(model=model, train_dataloader=dataloader,val_dataloader=dataloader_val, epochs=opt.num_epochs,
#              lr=lr, steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
#              model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, clip_grad=True,
#              fourier_feat_transformer=fourier_transformer, rank=3)
