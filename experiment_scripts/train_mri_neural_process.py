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

from torch.utils.data.dataloader import default_collate



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
p.add_argument('--steps_til_summary', type=int, default=1000,
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

image_resolution = (64, 64)
num_fourier_features = 64
use_fourier_features = True
img_dataset = dataio.FastMRIBrainKspace(split='train', downsampled=True, image_resolution=image_resolution)
#img_dataset = dataio.FastMRIBrain(split='train', downsampled=True, image_resolution=image_resolution)
#img_dataset = dataio.MRIImageDomain(split='train',downsample=True)
coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=image_resolution, image=False)

# TODO: right now, test_sparsity= ... overwrites train sparsity for training. using this to get CS
device = torch.device('cuda:4')  # or whatever device/cpu you like
generalization_dataset = dataio.ImageGeneralizationWrapper(coord_dataset,
                                                           train_sparsity_range=opt.train_sparsity_range,
                                                           test_sparsity= 'CS_cartesian',
                                                           generalization_mode=gmode,
                                                           device=device)

dataloader = DataLoader(generalization_dataset, shuffle=True, batch_size=opt.batch_size,
                         pin_memory=False, num_workers=0,)

if opt.conv_encoder:
    if use_fourier_features:
        model = meta_modules.ConvolutionalNeuralProcessImplicit2DHypernetFourierFeatures(in_features=2*num_fourier_features,
                                                                out_features=img_dataset.img_channels,
                                                                image_resolution=image_resolution,
                                                                fourier_features_size=2*num_fourier_features)
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

model.cuda(device)

# Define the loss
kl_weight = opt.kl_weight/4000
fw_weight = opt.fw_weight/4000

loss_fn = partial(loss_functions.image_hypernetwork_ift_loss, None, kl_weight, fw_weight)
#loss_fn = partial(loss_functions.image_hypernetwork_ift_loss, kl_weight, fw_weight)
summary_fn = partial(utils.write_image_summary_small, image_resolution, None)

root_path = os.path.join(opt.logging_root, opt.experiment_name)

fourier_transformer = GaussianFourierFeatureTransform(num_input_channels=2, mapping_size_spatial=num_fourier_features, 
                                                      scale=24, device=device)

# Record the fourier feature transform matrix
fourier_transformer.save_B('current_B.pt')


training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, clip_grad=True,
               fourier_feat_transformer=fourier_transformer)
