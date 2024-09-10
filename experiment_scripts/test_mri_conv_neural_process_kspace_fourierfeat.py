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

from PIL import Image

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
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
image_resolution = (64, 64)

#img_dataset_test = dataio.CelebA(split='test', downsampled=True)
img_dataset_test = dataio.FastMRIBrainKspace(split='val', downsampled=True, image_resolution=image_resolution)
coord_dataset_test = dataio.Implicit2DWrapper(img_dataset_test, sidelength=image_resolution, image=False)
generalization_dataset_test = dataio.ImageGeneralizationWrapper(coord_dataset_test, test_sparsity=3000,
                                                                generalization_mode='conv_cnp_test')

#img_dataset_train = dataio.CelebA(split='train', downsampled=True)
img_dataset_train = dataio.FastMRIBrainKspace(split='train', downsampled=True, image_resolution=image_resolution)
coord_dataset_train = dataio.Implicit2DWrapper(img_dataset_train, sidelength=image_resolution, image=False)
generalization_dataset_train = dataio.ImageGeneralizationWrapper(coord_dataset_train, test_sparsity=3000,
                                                                generalization_mode='conv_cnp_test')

# Define the model.
num_fourier_features = 64
out_channels = 2
model = meta_modules.ConvolutionalNeuralProcessImplicit2DHypernetFourierFeatures(in_features=2*num_fourier_features,
                                                        out_features=out_channels,
                                                        image_resolution=image_resolution,
                                                        fourier_features_size=2*num_fourier_features)
model.cuda()
model.eval()

# define the fourier feature tranformer 
fourier_transformer = GaussianFourierFeatureTransform(num_input_channels=2,
                                                      mapping_size_spatial=num_fourier_features, scale=15)

# Record the fourier feature transform matrix
fourier_transformer.load_B('current_B.pt')

root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)

# Load checkpoint
model.load_state_dict(torch.load(opt.checkpoint_path))

# First experiment: Upsample training image
model_input = {'coords':dataio.get_mgrid(image_resolution)[None,:].cuda(),
               'img_sparse':generalization_dataset_train[0][0]['img_sparse'].unsqueeze(0).cuda()}
model_input['coords'] = fourier_transformer(model_input['coords'])
model_output = model(model_input)

out_img = dataio.lin2img(model_output['model_out'], image_resolution).squeeze().permute(1,2,0).detach().cpu().numpy()


sio.savemat(os.path.join(root_path, 'upsampled_train.mat'),{'out_img':out_img})
#out_img = np.clip(out_img, 0., 1.)
#out_img = Image.fromarray(out_img)
#out_img = out_img.convert("L")

#imageio.imwrite(os.path.join(root_path, 'upsampled_train.png'), out_img)

# Second experiment: sample larger range
model_input = {'coords':dataio.get_mgrid(image_resolution)[None,:].cuda()*5,
               'img_sparse':generalization_dataset_train[0][0]['img_sparse'].unsqueeze(0).cuda()}
model_input['coords'] = fourier_transformer(model_input['coords'])
model_output = model(model_input)

out_img = dataio.lin2img(model_output['model_out'], image_resolution).squeeze().permute(1,2,0).detach().cpu().numpy()
out_img += 1
out_img /= 2.

sio.savemat(os.path.join(root_path, 'outside_range.mat'),{'out_img':out_img})


#fromarrayout_img = Image.fromarray(out_img)
#out_img = out_img.convert("L")

#imageio.imwrite(os.path.join(root_path, 'outside_range.png'), out_img)

# Third experiment: interpolate between latent codes
idx1, idx2 = 57, 181
model_input_1 = {'coords': dataio.get_mgrid(image_resolution)[None, :].cuda(),
                 'img_sparse': generalization_dataset_train[idx1][0]['img_sparse'].unsqueeze(0).cuda()}
model_input_2 = {'coords': dataio.get_mgrid(image_resolution)[None, :].cuda(),
                 'img_sparse': generalization_dataset_train[idx2][0]['img_sparse'].unsqueeze(0).cuda()}

embedding_1 = model.get_hypo_net_weights(model_input_1)[1]
embedding_2 = model.get_hypo_net_weights(model_input_2)[1]
for i in np.linspace(0,1,8):
    embedding = i*embedding_1 + (1.-i)*embedding_2
    model_input = {'coords': dataio.get_mgrid(image_resolution)[None, :].cuda(), 'embedding': embedding}
    model_input['coords'] = fourier_transformer(model_input['coords'])
    model_output = model(model_input)

    out_img = dataio.lin2img(model_output['model_out'], image_resolution).squeeze().permute(1,2,0).detach().cpu().numpy()

    if i == 0.:
        out_img_cat = out_img
    else:
        out_img_cat = np.concatenate((out_img_cat, out_img), axis=1)

sio.savemat(os.path.join(root_path, 'interpolated_image.mat'),{'out_img_cat':out_img_cat})

#out_img_cat = Image.fromarray(out_img_cat)
#out_img_cat = out_img_cat.convert("L")

#imageio.imwrite(os.path.join(root_path, 'interpolated_image.png'), out_img_cat)

# Fourth experiment: Fit test images

def getTestMSE(dataloader, subdir):
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

            with torch.no_grad():
                model_output = model(model_input)

            out_img = dataio.lin2img(model_output['model_out'], image_resolution).squeeze().permute(1,2,0).detach().cpu().numpy()
            gt_img = dataio.lin2img(gt['img'], image_resolution).squeeze().permute(1,2,0).detach().cpu().numpy()


            # TODO: Throwing a bug so I commented out
            sparse_img = model_input['img_sparse'].squeeze().detach().cpu().permute(1,2,0).numpy()
            mask = np.sum((sparse_img == 0), axis=2) == 3
            sparse_img[mask, ...] = 1.

            sio.savemat(os.path.join(root_path, f'ground_truth_img_{sparsity}.mat'),{'gt_img':gt_img, 'pred_img':out_img, 'sparse_img':sparse_img})

            MSE = np.mean((out_img - gt_img) ** 2)
            MSEs.append(MSE)

            pbar.update(1)
            total_steps += 1
            
            # TODO: JBM JUST PULLING ONE EXAMPLE
            break

    return MSEs


sparsities = [10, 100, 1000, 3000, 'full', 'half', 'CS_cartesian']
for sparsity in sparsities:
    generalization_dataset_test.update_test_sparsity(sparsity)
    dataloader = DataLoader(generalization_dataset_test, shuffle=False, batch_size=1, pin_memory=False, num_workers=0)
    MSE = getTestMSE(dataloader, 'test_' + str(sparsity) + '_pixels')
    np.save(os.path.join(root_path, 'MSE_' + str(sparsity) + '_context.npy'), MSE)
    print(np.mean(MSE))
