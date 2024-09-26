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

import optuna



def objective(trial, device_id):

    # fixed parameters
    n_trials = 1
    batch_size = 4 # with accumulation steps =16, this is an effective batch size of 64
    device = torch.device(device_id)  # or whatever device/cpu you like
    image_resolution = (128, 128)
    train_sparsity_range = [2000, 4000] # this gets overwritten
    logging_root = './logs'
    experiment_name = 'hyperopt'
    num_epochs = 4
    steps_til_summary = 1000
    gmode = 'conv_cnp'

    # hyperopt parameters
    num_fourier_features = trial.suggest_categorical('num_fourier_features', [8, 16, 32, 64, 128, 256])
    latent_dim = trial.suggest_categorical('latent_dim', [32, 64, 128, 256, 512])
    kernel_size = trial.suggest_categorical('conv_kernel_size', [3, 5, 7])
    hidden_features = trial.suggest_categorical('hidden_features', [32, 64, 128, 256, 512])
    hidden_features_hyper = trial.suggest_categorical('hidden_features_hyper', [32, 64, 128, 256, 512])
    hidden_layers = trial.suggest_int('hidden_layers', 1,5)
    hidden_layers_hyper = trial.suggest_int('hidden_layers_hyper', 1,3)
    lr = trial.suggest_float('lr', 1e-7, 1e-2, log=True)
    kl_weight = trial.suggest_float('kl_weight', 1e-12, 1e-4, log=True)
    fw_weight = trial.suggest_float('fw_weight', 1e-12, 1e-4, log=True)
    fourier_feat_scale = trial.suggest_float('fourier_scale', 2, 40, log=False)
    partial_conv = trial.suggest_categorical('partial_conv', [True, False])
    #accumulation_steps = trial.suggest_int('accumulation_steps', 8, 128)
    accumulation_steps=32

    
    img_dataset = dataio.FastMRIBrainKspace(split='train', downsampled=True, image_resolution=image_resolution)
    coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=image_resolution, image=False)

    generalization_dataset = dataio.ImageGeneralizationWrapper(coord_dataset,
                                                            train_sparsity_range=train_sparsity_range,
                                                            test_sparsity= 'CS_cartesian',
                                                            generalization_mode=gmode,
                                                            device=device)

    dataloader = DataLoader(generalization_dataset, shuffle=True, batch_size=batch_size,
                            pin_memory=False, num_workers=0,)

    # VAL DATASET
    img_dataset_val = dataio.FastMRIBrainKspace(split='val_small', downsampled=True, image_resolution=image_resolution)
    coord_dataset_val = dataio.Implicit2DWrapper(img_dataset_val, sidelength=image_resolution, image=False)
    generalization_dataset_val = dataio.ImageGeneralizationWrapper(coord_dataset_val,
                                                            train_sparsity_range=train_sparsity_range,
                                                            test_sparsity= 'CS_cartesian',
                                                            generalization_mode=gmode,
                                                            device=device)
    dataloader_val = DataLoader(generalization_dataset_val, shuffle=True, batch_size=batch_size,
                            pin_memory=False, num_workers=0,)



    loss_fn = partial(loss_functions.image_hypernetwork_ift_loss, None, kl_weight, fw_weight)
    #loss_fn = partial(loss_functions.image_hypernetwork_ift_loss, kl_weight, fw_weight)
    summary_fn = partial(utils.write_image_summary_small, image_resolution, None)

    root_path = os.path.join(logging_root, experiment_name)

    fourier_transformer = GaussianFourierFeatureTransform(num_input_channels=2, mapping_size_spatial=num_fourier_features, 
                                                        scale=fourier_feat_scale, device=device)

    # Record the fourier feature transform matrix
    #fourier_transformer.save_B('current_B.pt')

    trial_val_all = 0
    try:
        for ii in range(n_trials):
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
                                                        conv_kernel_size=kernel_size)
            model.cuda(device)

            print(f'Parameter Hyperopt trial #: {ii}')
            trial_val = training.train(model=model, train_dataloader=dataloader,val_dataloader=dataloader_val, epochs=num_epochs,
                        lr=lr, steps_til_summary=steps_til_summary, epochs_til_checkpoint=num_epochs-1,
                        model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, clip_grad=True,
                        fourier_feat_transformer=fourier_transformer, device=device, hyperopt_run=True, accumulation_steps=accumulation_steps)
            trial_val_all += trial_val
        trial_val_all /= n_trials
    except:
        print('Exception raised. Error in training with these parameters')
        trial_val_all = 1e2

    print(f'OUTPUT TRIAL_VAL = {trial_val_all}')

    return trial_val_all



if __name__ == "__main__":
    study = optuna.load_study(
        storage = "sqlite:///db.sqlite3_with_conv_parallelhyp",
        study_name = 'hyperopt_with_conv_parallelhyp')
    
    p = configargparse.ArgumentParser()
    p.add('-d', '--device_id', required=True, help='CUDA device ID.')
    opt = p.parse_args()

    objective = partial(objective, device_id=opt.device_id)
    study.optimize(objective, n_trials=300, gc_after_trial=True)
    print(f"Best value: {study.best_value} (params: {study.best_params})")