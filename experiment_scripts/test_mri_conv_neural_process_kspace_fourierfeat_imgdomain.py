# Enable import from parent package
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio, meta_modules
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import DataLoader
import configargparse

import imageio
from tqdm.autonotebook import tqdm
import utils
import skimage
from features import GaussianFourierFeatureTransform
from plotting import plot_weight_distribution

from PIL import Image

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
p.add_argument('--trained_with_ddp', type=bool, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--checkpoint_path', default=None, type=str, required=True,
               help='path to directory where checkpoints & tensorboard events will be saved.')
p.add_argument('--dataset', type=str, default='mri_image',
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model_type', type=str, default='sine',
               help='Nonlinearity in the neural implicit representation')
p.add_argument('--test_sparsity', type=float, default=3000,
               help='Amount of subsampled pixels input into the set encoder')
p.add_argument('--partial_conv', action='store_true', default=False, help='Use a partial convolution encoder')
opt = p.parse_args()

if opt.experiment_name is None:
    opt.experiment_name = opt.checkpoint_path.split('/')[-3] + '_TEST'
else:
    opt.experiment_name = opt.checkpoint_path.split('/')[-3] + '_' + opt.experiment_name

assert opt.dataset == 'mri_image'
image_resolution = (128, 128)

# CONFIG. TODO: transition to config.yml
config = 'from_early_expt'
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
    hidden_layers = 3 # modified from default (1)
    hidden_features = 512
    partial_conv = False
elif config =='hyperoptIII':
    num_fourier_features = 512
    kl_weight = 2.43e-7
    fw_weight = 9.68e-5 # JBM was e-5
    lr = 8e-5 
    fourier_features_scale = 20.4
    latent_dim = 512
    hidden_features_hyper = 512
    hidden_layers_hyper = 2
    hidden_layers = 6 # was 1
    hidden_features = 64
    partial_conv=False
elif config =='hyperoptIV':
    num_fourier_features = 150
    kl_weight = 2.78e-8
    fw_weight = 6.4e-6 # JBM was e-5
    lr = 5.57e-5 
    fourier_features_scale = 21
    latent_dim = 256
    hidden_features_hyper = 128
    hidden_layers_hyper = 3
    hidden_layers = 3 # was 1
    hidden_features = 256
    partial_conv=False
    conv_kernel_size = 7
    num_conv_res_blocks=5
elif config =='hyperoptIV_homebrew':
    num_fourier_features = 60
    kl_weight = 2.78e-8
    fw_weight = 6.4e-6 # JBM was e-5
    lr = 5.57e-6 # JBM was e-5 
    fourier_features_scale = 21
    latent_dim = 128
    hidden_features_hyper = 128
    hidden_layers_hyper = 2
    hidden_layers = 3 # was 1
    hidden_features = 256
    partial_conv=False
    conv_kernel_size = 7
    num_conv_res_blocks=5
elif config =='hyperopt_asinh':
    num_fourier_features = 103
    kl_weight = 2.08e-9 #1.3e-5
    fw_weight = 1.2e-5 # JBM was e-5
    lr = 1.04e-6 # JBM was e-5 
    fourier_features_scale = 20
    latent_dim = 64
    hidden_features_hyper = 512
    hidden_layers_hyper = 1
    hidden_layers = 3 # was 1
    hidden_features = 256
    partial_conv=False
    conv_kernel_size = 7
    num_conv_res_blocks= 6
elif config =='hyperopt_highfreq':
    num_fourier_features = 228
    kl_weight = 1.35e-8 #1.3e-5
    fw_weight = 2.6e-6 # JBM was e-5
    lr = 1.35e-6 # JBM was e-5 
    fourier_features_scale = 21
    latent_dim = 64
    hidden_features_hyper = 256
    hidden_layers_hyper = 2
    hidden_layers = 4 # was 1
    hidden_features = 512
    partial_conv=False
    conv_kernel_size = 7
    num_conv_res_blocks= 3
    w0=30
elif config =='from_early_expt':
    # Notes: Biggest improvements came from adding more hypernetwork layers. "Best = 0.0005 for batchsize 8"
    num_fourier_features = 128
    kl_weight = 2.78e-8 #optim # 0.1 in paper
    fw_weight = 1e-6#1e-2 best for mse #1e-6#optim # 100 in paper
    lr = 1e-5#5.e-5 best for mse
    fourier_features_scale = 1 # best = 10
    latent_dim = 256 # best = 256
    hidden_features_hyper = 256 #256 # best = 256
    hidden_layers_hyper = 5 # try just 1. 3 gave 0.0011 after 15 epochs. 5 gave 0.0004- was best!! 
    hidden_layers = 6
    hidden_features = 64
    partial_conv=False
    conv_kernel_size = 5
    num_conv_res_blocks= 3 # go back to orig paper, was 3
    w0=30

device = 'cuda:5'

#img_dataset_test = dataio.CelebA(split='test', downsampled=True)
img_dataset_test = dataio.FastMRIBrainImageKspaceEncode(split='val', downsampled=True, image_resolution=image_resolution)
coord_dataset_test = dataio.Implicit2DWrapper(img_dataset_test, sidelength=image_resolution, image=True)
generalization_dataset_test = dataio.ImageGeneralizationWrapper(coord_dataset_test, test_sparsity=3000,
                                                                generalization_mode='conv_cnp_test',
                                                                device=device)

#img_dataset_train = dataio.CelebA(split='train', downsampled=True)
img_dataset_train = dataio.FastMRIBrainImageKspaceEncode(split='train', downsampled=True, image_resolution=image_resolution)
coord_dataset_train = dataio.Implicit2DWrapper(img_dataset_train, sidelength=image_resolution, image=True)
generalization_dataset_train = dataio.ImageGeneralizationWrapper(coord_dataset_train, test_sparsity=3000,
                                                                generalization_mode='conv_cnp_test',
                                                                device=device)

# Define the model.
out_channels = 1
model = meta_modules.ConvolutionalNeuralProcessImplicit2DHypernetFourierFeatures(in_features=2*num_fourier_features,
                                                        out_features=out_channels,
                                                        image_resolution=image_resolution,
                                                        fourier_features_size=2*num_fourier_features,
                                                        latent_dim=latent_dim,
                                                        hidden_features=hidden_features,
                                                        hyper_hidden_features=hidden_features_hyper,
                                                        hyper_hidden_layers=hidden_layers_hyper,
                                                        num_hidden_layers=hidden_layers,
                                                        partial_conv=partial_conv,
                                                        conv_kernel_size=conv_kernel_size,
                                                        num_conv_res_blocks=num_conv_res_blocks,
                                                        w0=w0, use_dc=False)
model.cuda()
model.eval()

# define the fourier feature tranformer 
fourier_transformer = GaussianFourierFeatureTransform(num_input_channels=2,
                                                      mapping_size_spatial=num_fourier_features, scale=fourier_features_scale)

# Record the fourier feature transform matrix
#fourier_transformer.load_B('./logs/'+opt.experiment_name+'/current_B_DDP.pt')
# TODO this needs to be more automatic
#savepath = './logs/'+'DDP_RESET_large_dataset_20featscale'+'/current_B_DDP_placeholder.pt'
experiment_name = 'DDP_RESET_img_domain_128FF'
savepath = './logs/fourier_feat_mats/current_B_DDP_placeholder_'+experiment_name+'.pt'
fourier_transformer.load_B(savepath)
print(f"size of fourier B = {np.shape(fourier_transformer._B_spatial)}")

root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)

# Load checkpoint
model.load_state_dict(torch.load(opt.checkpoint_path))


# # Third experiment: interpolate between latent codes
# generalization_dataset_train.update_test_sparsity('full')
# idx1, idx2 = 57, 181
# model_input_1 = {'coords': dataio.get_mgrid(image_resolution)[None, :].cuda(),
#                  'img_sparse': generalization_dataset_train[idx1][0]['img_sparse'].unsqueeze(0).cuda()}
# model_input_2 = {'coords': dataio.get_mgrid(image_resolution)[None, :].cuda(),
#                  'img_sparse': generalization_dataset_train[idx2][0]['img_sparse'].unsqueeze(0).cuda()}

# embedding_1 = model.get_hypo_net_weights(model_input_1)[1]
# embedding_2 = model.get_hypo_net_weights(model_input_2)[1]
# for i in np.linspace(0,1,8):
#     embedding = i*embedding_1 + (1.-i)*embedding_2
#     model_input = {'coords': dataio.get_mgrid(image_resolution)[None, :].cuda(), 'embedding': embedding}
#     model_input['coords'] = fourier_transformer(model_input['coords'])

#     model_output = model(model_input)

#     out_img = dataio.lin2img(model_output['model_out'], image_resolution).squeeze().permute(1,2,0).detach().cpu().numpy()

#     if i == 0.:
#         out_img_cat = out_img
#     else:
#         out_img_cat = np.concatenate((out_img_cat, out_img), axis=1)

# sio.savemat(os.path.join(root_path, 'interpolated_image.mat'),{'out_img_cat':out_img_cat})



# Fourth experiment: Fit test images

def getTestMSE(dataloader, subdir, trial_num=0):
    MSEs = []
    total_steps = 0
    utils.cond_mkdir(os.path.join(root_path, subdir))
    utils.cond_mkdir(os.path.join(root_path, 'ground_truth'))

    with tqdm(total=len(dataloader)) as pbar:
        for step, (model_input, gt) in enumerate(dataloader):
            model_input['idx'] = torch.Tensor([model_input['idx']]).long()
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}
            model_input['coords'] = fourier_transformer(model_input['coords'])
            print('Size of INPUT to model')
            print(np.shape(model_input['dc_mask']))
            print(np.shape(model_input['img_sparse']))

            with torch.no_grad():
                model_output = model(model_input)

            out_img = dataio.lin2img(model_output['model_out'], image_resolution).squeeze().detach().cpu().numpy()
            gt_img = dataio.lin2img(gt['img'], image_resolution).squeeze().detach().cpu().numpy()


            sparse_img = model_input['img_sparse'].squeeze().detach().cpu().permute(1,2,0).numpy()
            mask = np.sum((sparse_img == 0), axis=2) == 3
            sparse_img[mask, ...] = 1.

            sio.savemat(os.path.join(root_path, f'ground_truth_img_{sparsity}_{trial_num}.mat'),{'gt_img':gt_img, 'pred_img':out_img, 'sparse_img':sparse_img})

            MSE = np.mean((out_img - gt_img) ** 2)
            MSEs.append(MSE)

            pbar.update(1)
            total_steps += 1
            
            # TODO: JBM JUST PULLING ONE EXAMPLE
            break

    return MSEs


#sparsities = [10, 100, 1000, 3000, 'full', 'half', 'CS_cartesian']
sparsities = ['CS_cartesian_from_img_domain']
#num_img_in_sparsity = 5
for sparsity in sparsities:
    generalization_dataset_test.update_test_sparsity(sparsity)
    dataloader = DataLoader(generalization_dataset_test, shuffle=False, batch_size=1, pin_memory=False, num_workers=0)
    MSE = getTestMSE(dataloader, 'test_' + str(sparsity) + '_pixels', trial_num=0)
    np.save(os.path.join(root_path, 'MSE_' + str(sparsity) + '_context.npy'), MSE)
    print(np.mean(MSE))

additional_trials = 4
for sparsity in sparsities:
    for ii in range(additional_trials):
        generalization_dataset_test.update_test_sparsity(sparsity)
        dataloader = DataLoader(generalization_dataset_test, shuffle=True, batch_size=1, pin_memory=False, num_workers=0)
        MSE = getTestMSE(dataloader, 'test_' + str(sparsity) + '_pixels', trial_num=ii)
        #np.save(os.path.join(root_path, 'MSE_' + str(sparsity) + '_context.npy'), MSE)
        print(np.mean(MSE))
