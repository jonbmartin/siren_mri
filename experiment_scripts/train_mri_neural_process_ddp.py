# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training_ddp, loss_functions

import torch
from torch.utils.data import DataLoader
import configargparse
from functools import partial
from features import GaussianFourierFeatureTransform
import torch.nn as nn
from training_ddp import ddp_setup


import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def main(rank, world_size, total_epochs, save_every, load_from_checkpoint_path, experiment_name, B):
    # Inputs: 
    #   rank: (int) the identifier of the gpu on which the particular process is being run
    #   world_size: (int) number of processes to run in parallel
    #   total_epochs: (int) number of epochs to train for
    #   save_every: (int) how often to save the model (every 'save_every' epochs)
    #   load_from_checkpoint_path: (None or string) path from which to load model checkpoint to resume training

    # fixed parameters
    batch_size = 16 # with accumulation steps =16, this is an effective batch size of 96 (16*6)
    accumulation_steps = 1
    image_resolution = (128, 128)
    train_sparsity_range = [2000, 4000] # this gets overwritten
    logging_root = './logs'
    num_epochs = total_epochs
    steps_til_summary = 100
    gmode = 'conv_cnp'


    # JBM OVERRIDE TO A CONV_CNP
    conv_encoder = True
    if conv_encoder: gmode = 'conv_cnp'
    else: gmode = 'cnp'

    # DDP setup
    ddp_setup(rank, world_size)

    # CONFIG. TODO: transition to config.yml
    config = 'hyperoptIV_homebrew'
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
        num_fourier_features = 8
        kl_weight = 2.78e-8
        fw_weight = 6.4e-6 # JBM was e-5
        lr = 5.57e-5 # JBM was e-5 # increased to -3
        fourier_features_scale = 21
        latent_dim = 128
        hidden_features_hyper = 128
        hidden_layers_hyper = 2
        hidden_layers = 3 # was 1
        hidden_features = 256
        partial_conv=False
        conv_kernel_size = 7
        num_conv_res_blocks=5
        w0 = 30
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
    elif config =='hyperoptIV_homebrew_small':
        num_fourier_features = 60
        kl_weight = 2.78e-8
        fw_weight = 6.4e-6 # JBM was e-5
        lr = 5.57e-5 # JBM was e-5 
        fourier_features_scale = 21
        latent_dim = 128
        hidden_features_hyper = 128
        hidden_layers_hyper = 2
        hidden_layers = 3 # was 1
        hidden_features = 256
        partial_conv=False
        conv_kernel_size = 3
        num_conv_res_blocks=3
        w0=30

    image_resolution = (128, 128)
    use_fourier_features = True
    img_dataset = dataio.FastMRIBrainKspace(split='train', downsampled=True, image_resolution=image_resolution)
    #img_dataset = dataio.FastMRIBrain(split='train', downsampled=True, image_resolution=image_resolution)
    #img_dataset = dataio.MRIImageDomain(split='train',downsample=True)
    coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=image_resolution, image=False)

    # TODO: right now, test_sparsity= ... overwrites train sparsity for training. using this to get CS
    generalization_dataset = dataio.ImageGeneralizationWrapper(coord_dataset,
                                                            train_sparsity_range=train_sparsity_range,
                                                            test_sparsity= 'CS_cartesian',
                                                            generalization_mode=gmode,
                                                            device=rank)

    dataloader = DataLoader(generalization_dataset, shuffle=False, batch_size=batch_size,
                            pin_memory=False, num_workers=0, sampler=DistributedSampler(generalization_dataset))

    # VAL DATASET
    img_dataset_val = dataio.FastMRIBrainKspace(split='val_small', downsampled=True, image_resolution=image_resolution)
    coord_dataset_val = dataio.Implicit2DWrapper(img_dataset_val, sidelength=image_resolution, image=False)
    generalization_dataset_val = dataio.ImageGeneralizationWrapper(coord_dataset_val,
                                                            train_sparsity_range=train_sparsity_range,
                                                            test_sparsity= 'CS_cartesian',
                                                            generalization_mode=gmode,
                                                            device=rank)
    dataloader_val = DataLoader(generalization_dataset_val, shuffle=True, batch_size=batch_size,
                            pin_memory=False, num_workers=0,)

    if conv_encoder:
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
                                                                    partial_conv=partial_conv,
                                                                    conv_kernel_size=conv_kernel_size,
                                                                    num_conv_res_blocks=num_conv_res_blocks,
                                                                    w0=w0)
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
    if load_from_checkpoint_path is not None:
        model.load_state_dict(torch.load(load_from_checkpoint_path))

    model.to(rank)
    model = DDP(model, device_ids =[rank],find_unused_parameters=True)

    # Define the loss
    #loss_fn = partial(loss_functions.image_hypernetwork_l1_loss, None, kl_weight, fw_weight)
    #loss_fn = partial(loss_functions.image_hypernetwork_loss, None, kl_weight, fw_weight)
    # NOTE THAT THIS IS THE SMAPE NOT THE MAPE
    loss_fn = partial(loss_functions.image_hypernetwork_loss, None, kl_weight, fw_weight)
    summary_fn = partial(utils.write_image_summary_small, image_resolution, None)

    root_path = os.path.join(logging_root, experiment_name)

    fourier_transformer = GaussianFourierFeatureTransform(num_input_channels=2, mapping_size_spatial=num_fourier_features, 
                                                        scale=fourier_features_scale, device=rank)
    fourier_transformer.set_B(B)
    # # load the transformation to be used by ALL DDP processes 
    print(f'Current working directory = {os.getcwd()}')
    savepath = './logs/'+experiment_name+'/current_B_DDP_mp'+str(rank)+'.pt'
    #fourier_transformer.load_B('./logs/'+experiment_name+'/current_B_DDP_mp'+str(rank)+'.pt')
    fourier_transformer.save_B(savepath)
    print(f'rank {rank} successfully loaded B')

    training_ddp.train_ddp(model=model, train_dataloader=dataloader,val_dataloader=dataloader_val, epochs=num_epochs,
                lr=lr, steps_til_summary=steps_til_summary, epochs_til_checkpoint=save_every,
                model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, clip_grad=True,
                fourier_feat_transformer=fourier_transformer, device=rank, accumulation_steps=accumulation_steps,
                ddp_run=True)

    destroy_process_group()


if __name__ == "__main__":
    import sys
    total_epochs = 200
    save_every = 5
    world_size = 4

    # TODO: manually setting this to be the same as that inside main()
    # create the fourier feature transform to be used by ALL DDP processes 
    num_fourier_features = 8
    fourier_features_scale = 21
    device = 1
    resume_from_save = False
    experiment_name = 'DDP_RESET_high_freq_training'

    if resume_from_save:
        load_from_checkpoint_path = './logs/DDP/checkpoints/model_epoch_0030.pth'
        print(f'Resuming training from checkpoint found at: {load_from_checkpoint_path}')
    else:
        load_from_checkpoint_path = None
        fourier_transformer = GaussianFourierFeatureTransform(num_input_channels=2, mapping_size_spatial=num_fourier_features, 
                                                        scale=fourier_features_scale, device=device)
        
        # Record the fourier feature transform matrix and place in current experiment folder.
        # Making multiple copies for the different processes
        #for ii in range(world_size):
            #savepath = './logs/'+experiment_name+'/current_B_DDP_mp'+str(ii)+'.pt'
            #print(f'Saving B transform mat at: {savepath}')
            #fourier_transformer.save_B(savepath)
        B = fourier_transformer.get_B()

    mp.spawn(main, args=(world_size, total_epochs,save_every,load_from_checkpoint_path, experiment_name, B), nprocs=world_size)
