
import torch
import torch.nn.functional as F

import diff_operators
import modules
import numpy as np
import dataio
import scipy.io as sio
import utils
import sys

def image_mse_log(mask, model_output, gt):
    # SAME as below, but no image domain loss/ fourier transforms 

    kspace_output_real = dataio.lin2img(model_output['model_out'])
    kspace_output = kspace_output_real[:,0,:,:] + 1j * kspace_output_real[:,1,:,:]
    kspace_output_mag = torch.abs(kspace_output)
    kspace_output_phase = torch.angle(kspace_output)

    kspace_gt_real = dataio.lin2img(gt['img'])
    kspace_gt = kspace_gt_real[:,0,:,:] + 1j * kspace_gt_real[:,1,:,:]
    kspace_gt_mag = torch.abs(kspace_gt)
    kspace_gt_phase = torch.angle(kspace_gt)

    eps = 1e-3

    kspace_pred_log = torch.log(torch.abs(kspace_output_mag)+eps)
    kspace_gt_log = torch.log(torch.abs(kspace_gt_mag)+eps)


    # add a kspace domain loss:
    mag_weight = 0.0001
    phase_weight = 0.0001
    test_mag_weight = (mag_weight *(kspace_pred_log-kspace_gt_log)**2).sum()
    test_phase_weight = (phase_weight * (2-torch.cos(kspace_output_phase-kspace_gt_phase))).sum()

    print(f'Calculated loss of mag: {test_mag_weight}, of phase: {test_phase_weight}')
    kspace_loss =  (mag_weight *(kspace_pred_log-kspace_gt_log)**2 + 
                    phase_weight * (2-torch.cos(kspace_output_phase-kspace_gt_phase))).sum()

    if mask is None:
        return {'img_loss': (kspace_loss)}
    else:
        return {'img_loss': (kspace_loss)}
    
def image_mse_cubic(mask, model_output, gt):

    kspace_output_real = dataio.lin2img(model_output['model_out'])

    kspace_gt_real = dataio.lin2img(gt['img'])

    eps = 1e-3
    kspace_pred_tx = torch.sign(kspace_output_real)*torch.pow(torch.abs(kspace_output_real)+eps,1/3)
    kspace_gt_tx = torch.sign(kspace_gt_real)*torch.pow(torch.abs(kspace_gt_real)+eps,1/3)

    # add a kspace domain loss:
    kspace_weight = 0.000001
    kspace_loss = kspace_weight * ((kspace_pred_tx-kspace_gt_tx)**2).sum()


    if mask is None:
        return {'img_loss': (kspace_loss)}
    else:
        return {'img_loss': (kspace_loss)}

def image_mse(mask, model_output, gt, high_freq=False):
    # SAME as below, but no image domain loss/ fourier transforms 

    kspace_output_real = dataio.lin2img(model_output['model_out'])

    kspace_gt_real = dataio.lin2img(gt['img'])

    # transform
    #kspace_output_real = torch.asinh(400*kspace_output_real)/6.7
    #kspace_gt_real = torch.asinh(400*kspace_gt_real)/6.7

    #kspace_pred = kspace_output_real[:,0,:,:] + 1j * kspace_output_real[:,1,:,:]
    #kspace_gt = kspace_gt_real[:,0,:,:] + 1j * kspace_gt_real[:,1,:,:]
    #cplx_diff = kspace_pred-kspace_gt

    if high_freq:
        high_freq_mask = utils.create_circular_mask_torch(129, 129,center=None, radius=20)
        high_freq_mask = 1-high_freq_mask
        high_freq_mask = 1 * high_freq_mask.to(kspace_gt_real.device)
        #cplx_diff = cplx_diff * mask

    # add a kspace domain loss:
    kspace_weight = 1/(128*128) # if using 3, 0.0025. If using 6, 0.02 # dim sizekspace_pred

    #kspace_loss = ((torch.real(cplx_diff))**2).sum() + ((torch.imag(cplx_diff))**2).sum()
    if high_freq:
        kspace_loss = (torch.abs(high_freq_mask*(kspace_output_real-kspace_gt_real))**2).sum()
    else:
        kspace_loss = (torch.abs((kspace_output_real-kspace_gt_real))**2).sum()
    #print(f'kspace loss (unweighted): {kspace_loss}')
    kspace_loss = kspace_loss * kspace_weight

    if mask is None:
        return {'img_loss': (kspace_loss)}
    else:
        return {'img_loss': (kspace_loss)}

def image_smape(mask, model_output, gt):
     # SAME as below, but no image domain loss/ fourier transforms 

    kspace_output_real = dataio.lin2img(model_output['model_out'])
    kspace_gt_real = dataio.lin2img(gt['img'])

    # add a kspace domain loss:
    kspace_weight = 1/(128*128)
    eps = 1e-3
    kspace_loss = kspace_weight * ((torch.abs(kspace_output_real-kspace_gt_real))/(eps + torch.abs(kspace_gt_real)+torch.abs(kspace_output_real))).sum()

    if mask is None:
        return {'img_loss': (kspace_loss)}
    else:
        return {'img_loss': (kspace_loss)}

def image_asinh(mask, model_output, gt):
     # SAME as below, but no image domain loss/ fourier transforms 

    kspace_output_real = dataio.lin2img(model_output['model_out'])
    kspace_gt_real = dataio.lin2img(gt['img'])

    output_before_tx = kspace_output_real
    gt_before_tx = kspace_gt_real
    kspace_output_real = torch.asinh(50 * kspace_output_real)/4.67
    kspace_gt_real = torch.asinh(50 * kspace_gt_real)/4.67

    # print('Evaluating asinh loss')
    # sio.savemat('evaluating_asinh_loss.mat',{'pred_no_tx':output_before_tx.detach().cpu().numpy(), 'pred_tx':kspace_output_real.detach().cpu().numpy(),
    #                                          'true_no_tx':gt_before_tx.detach().cpu().numpy(), 'true_tx':kspace_gt_real.detach().cpu().numpy()})
    # sys.exit()
    # add a kspace domain loss:
    kspace_weight = 1/(128*128)
    kspace_loss = kspace_weight * ((kspace_output_real-kspace_gt_real)**2).sum()

    if mask is None:
        return {'img_loss': (kspace_loss)}
    else:
        return {'img_loss': (kspace_loss)}

def image_perp(mask, model_output, gt):
     # SAME as below, but no image domain loss/ fourier transforms 

    kspace_output_real = dataio.lin2img(model_output['model_out'])
    kspace_gt_real = dataio.lin2img(gt['img'])
    
    kspace_pred = kspace_output_real[:,0,:,:] + 1j * kspace_output_real[:,1,:,:]
    kspace_gt = kspace_gt_real[:,0,:,:] + 1j * kspace_gt_real[:,1,:,:]


    output_before_tx = kspace_pred
    gt_before_tx = kspace_gt

    eps = 1e-8
    loss = torch.abs(torch.real(kspace_pred)*torch.imag(kspace_gt)-torch.imag(kspace_pred)*torch.real(kspace_gt))/(torch.abs(kspace_pred)+eps)

    # dynamic range shift
    #loss = loss**(1/3)
    kspace_weight = 1/(128*128)*1000
    kspace_loss = kspace_weight*loss.sum()

    if mask is None:
        return {'img_loss': (kspace_loss)}
    else:
        return {'img_loss': (kspace_loss)}
    
def kspace_l1(mask, model_output, gt):
    # SAME as below, but no image domain loss/ fourier transforms 

    kspace_output_real = dataio.lin2img(model_output['model_out'])
    kspace_output = kspace_output_real[:,0,:,:] + 1j * kspace_output_real[:,1,:,:]

    kspace_gt_real = dataio.lin2img(gt['img'])
    kspace_gt = kspace_gt_real[:,0,:,:] + 1j * kspace_gt_real[:,1,:,:]

    # add l1 reg in kspace dim to encourage sparsity
    l1_reg = 0
    l1_cost = l1_reg * torch.abs(kspace_output).sum()

    # add a kspace domain loss:
    kspace_weight = 1/(128*128) # if using 3, 0.0025. If using 6, 0.02 # dim sizekspace_pred
    kspace_loss = kspace_weight * (torch.abs(kspace_output_real-kspace_gt_real)).sum()


    if mask is None:
        return {'img_loss': (l1_cost + kspace_loss)}
    else:
        return {'img_loss': (l1_cost + kspace_loss)}

def ift_image_mse(mask, model_output, gt):
    
    # DC implementation: just set loss from those locations to 0. So apply mask to both
    # dc mask is sampled points, so want points OUTSIDE that set
    #dc_mask = gt['dc_mask']
    #dc_mask = torch.squeeze(dc_mask[:,1,:,:])
    #learned_data_mask = 1-dc_mask
    #sio.savemat('testing_masks.mat',{'dc_mask':dc_mask.cpu().detach().numpy(),
    #                                  'learned_data_mask':learned_data_mask.cpu().detach().numpy()})


    kspace_output_real = dataio.lin2img(model_output['model_out'])
    kspace_output = kspace_output_real[:,0,:,:] + 1j * kspace_output_real[:,1,:,:]
    #kspace_output = kspace_output * learned_data_mask

    kspace_gt_real = dataio.lin2img(gt['img'])
    kspace_gt = kspace_gt_real[:,0,:,:] + 1j * kspace_gt_real[:,1,:,:]

    # combine DC data and learned data
    #kspace_output = kspace_output + kspace_gt*dc_mask

    img_output = torch.fft.ifft2(kspace_output)
    pred_real = torch.real(img_output)
    pred_imag = torch.imag(img_output)

    img_gt = torch.fft.ifft2(kspace_gt)
    gt_real = torch.real(img_gt)
    gt_imag = torch.imag(img_gt)

    # add a kspace domain loss:
    img_weight = 1/(128*128)*1e4
    kspace_loss = 0

    #print(f'size of output in LOSS = {np.shape(kspace_gt)}')
    if mask is None:
        return {'img_loss': (img_weight*((pred_real - gt_real)**2+(pred_imag - gt_imag)**2)).sum()}
    else:
        return {'img_loss': (img_weight*((pred_real - gt_real)**2+(pred_imag - gt_imag)**2)).sum()}

def image_l1(mask, model_output, gt):
    if mask is None:
        return {'img_loss': torch.abs(model_output['model_out'] - gt['img']).mean()}
    else:
        return {'img_loss': (mask * torch.abs(model_output['model_out'] - gt['img'])).mean()}


def image_mse_TV_prior(mask, k1, model, model_output, gt):
    coords_rand = 2 * (torch.rand((model_output['model_in'].shape[0],
                                   model_output['model_in'].shape[1] // 2,
                                   model_output['model_in'].shape[2])).cuda() - 0.5)
    rand_input = {'coords': coords_rand}
    rand_output = model(rand_input)

    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(diff_operators.gradient(
                    rand_output['model_out'], rand_output['model_in']))).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(diff_operators.gradient(
                    rand_output['model_out'], rand_output['model_in']))).mean()}


def image_mse_FH_prior(mask, k1, model, model_output, gt):
    coords_rand = 2 * (torch.rand((model_output['model_in'].shape[0],
                                   model_output['model_in'].shape[1] // 2,
                                   model_output['model_in'].shape[2])).cuda() - 0.5)
    rand_input = {'coords': coords_rand}
    rand_output = model(rand_input)

    img_hessian, status = diff_operators.hessian(rand_output['model_out'],
                                                 rand_output['model_in'])
    img_hessian = img_hessian.view(*img_hessian.shape[0:2], -1)
    hessian_norm = img_hessian.norm(dim=-1, keepdim=True)

    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(hessian_norm)).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(hessian_norm)).mean()}


def latent_loss(model_output):
    return torch.mean(model_output['latent_vec'] ** 2)


def hypo_weight_loss(model_output):
    weight_sum = 0
    total_weights = 0

    for weight in model_output['hypo_params'].values():
        weight_sum += torch.sum(weight ** 2)
        total_weights += weight.numel()

    return weight_sum * (1 / total_weights)


def image_hypernetwork_img_domain_loss(mask, kl, fw, model_output, gt):
    print(f'Hypo weight loss = {fw * hypo_weight_loss(model_output)}')
    print(f'Latent loss = {kl * latent_loss(model_output)}')
    img_loss = ift_image_mse(mask, model_output, gt)['img_loss']
    print(f'Img loss = {img_loss}')
    return {'img_loss': ift_image_mse(mask, model_output, gt)['img_loss'],
            'latent_loss': kl * latent_loss(model_output),
            'hypo_weight_loss': fw * hypo_weight_loss(model_output)}

def image_hypernetwork_loss(mask, kl, fw, model_output, gt):
    return {'img_loss': image_mse(mask, model_output, gt)['img_loss'],
            'latent_loss': kl * latent_loss(model_output),
            'hypo_weight_loss': fw * hypo_weight_loss(model_output)}

def image_hypernetwork_perp_loss(mask, kl, fw, model_output, gt):
    return {'img_loss': image_perp(mask, model_output, gt)['img_loss'],
            'latent_loss': kl * latent_loss(model_output),
            'hypo_weight_loss': fw * hypo_weight_loss(model_output)}

def image_hypernetwork_asinh_loss(mask, kl, fw, model_output, gt):
    return {'img_loss': image_asinh(mask, model_output, gt)['img_loss'],
            'latent_loss': kl * latent_loss(model_output),
            'hypo_weight_loss': fw * hypo_weight_loss(model_output)}

def image_hypernetwork_l1_loss(mask, kl, fw, model_output, gt):
    return {'img_loss': kspace_l1(mask, model_output, gt)['img_loss'],
            'latent_loss': kl * latent_loss(model_output),
            'hypo_weight_loss': fw * hypo_weight_loss(model_output)}

def image_hypernetwork_log_loss(mask, kl, fw, model_output, gt):
    return {'img_loss': image_mse_log(mask, model_output, gt)['img_loss'],
            'latent_loss': kl * latent_loss(model_output),
            'hypo_weight_loss': fw * hypo_weight_loss(model_output)}

def image_hypernetwork_cubic_loss(mask, kl, fw, model_output, gt):
    return {'img_loss': image_mse_cubic(mask, model_output, gt)['img_loss'],
            'latent_loss': kl * latent_loss(model_output),
            'hypo_weight_loss': fw * hypo_weight_loss(model_output)}

def image_hypernetwork_ift_loss(mask, kl, fw, model_output, gt):
    return {'img_loss': ift_image_mse(mask, model_output, gt)['img_loss'],
            'latent_loss': kl * latent_loss(model_output),
            'hypo_weight_loss': fw * hypo_weight_loss(model_output)}


def function_mse(model_output, gt):
    return {'func_loss': ((model_output['model_out'] - gt['func']) ** 2).mean()}


def gradients_mse(model_output, gt):
    # compute gradients on the model
    gradients = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
    # compare them with the ground-truth
    gradients_loss = torch.mean((gradients - gt['gradients']).pow(2).sum(-1))
    return {'gradients_loss': gradients_loss}


def gradients_color_mse(model_output, gt):
    # compute gradients on the model
    gradients_r = diff_operators.gradient(model_output['model_out'][..., 0], model_output['model_in'])
    gradients_g = diff_operators.gradient(model_output['model_out'][..., 1], model_output['model_in'])
    gradients_b = diff_operators.gradient(model_output['model_out'][..., 2], model_output['model_in'])
    gradients = torch.cat((gradients_r, gradients_g, gradients_b), dim=-1)
    # compare them with the ground-truth
    weights = torch.tensor([1e1, 1e1, 1., 1., 1e1, 1e1]).cuda()
    gradients_loss = torch.mean((weights * (gradients[0:2] - gt['gradients']).pow(2)).sum(-1))
    return {'gradients_loss': gradients_loss}


def laplace_mse(model_output, gt):
    # compute laplacian on the model
    laplace = diff_operators.laplace(model_output['model_out'], model_output['model_in'])
    # compare them with the ground truth
    laplace_loss = torch.mean((laplace - gt['laplace']) ** 2)
    return {'laplace_loss': laplace_loss}


def wave_pml(model_output, gt):
    source_boundary_values = gt['source_boundary_values']
    x = model_output['model_in']  # (meta_batch_size, num_points, 3)
    y = model_output['model_out']  # (meta_batch_size, num_points, 1)
    squared_slowness = gt['squared_slowness']
    dirichlet_mask = gt['dirichlet_mask']
    batch_size = x.shape[1]

    du, status = diff_operators.jacobian(y, x)
    dudt = du[..., 0]

    if torch.all(dirichlet_mask):
        diff_constraint_hom = torch.Tensor([0])
    else:
        hess, status = diff_operators.jacobian(du[..., 0, :], x)
        lap = hess[..., 1, 1, None] + hess[..., 2, 2, None]
        dudt2 = hess[..., 0, 0, None]
        diff_constraint_hom = dudt2 - 1 / squared_slowness * lap

    dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]
    neumann = dudt[dirichlet_mask]

    return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 1e1,
            'neumann': torch.abs(neumann).sum() * batch_size / 1e2,
            'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}


def helmholtz_pml(model_output, gt):
    source_boundary_values = gt['source_boundary_values']

    if 'rec_boundary_values' in gt:
        rec_boundary_values = gt['rec_boundary_values']

    wavenumber = gt['wavenumber'].float()
    x = model_output['model_in']  # (meta_batch_size, num_points, 2)
    y = model_output['model_out']  # (meta_batch_size, num_points, 2)
    squared_slowness = gt['squared_slowness'].repeat(1, 1, y.shape[-1] // 2)
    batch_size = x.shape[1]

    full_waveform_inversion = False
    if 'pretrain' in gt:
        pred_squared_slowness = y[:, :, -1] + 1.
        if torch.all(gt['pretrain'] == -1):
            full_waveform_inversion = True
            pred_squared_slowness = torch.clamp(y[:, :, -1], min=-0.999) + 1.
            squared_slowness_init = torch.stack((torch.ones_like(pred_squared_slowness),
                                                 torch.zeros_like(pred_squared_slowness)), dim=-1)
            squared_slowness = torch.stack((pred_squared_slowness, torch.zeros_like(pred_squared_slowness)), dim=-1)
            squared_slowness = torch.where((torch.abs(x[..., 0, None]) > 0.75) | (torch.abs(x[..., 1, None]) > 0.75),
                                           squared_slowness_init, squared_slowness)
        y = y[:, :, :-1]

    du, status = diff_operators.jacobian(y, x)
    dudx1 = du[..., 0]
    dudx2 = du[..., 1]

    a0 = 5.0

    # let pml extend from -1. to -1 + Lpml and 1 - Lpml to 1.0
    Lpml = 0.5
    dist_west = -torch.clamp(x[..., 0] + (1.0 - Lpml), max=0)
    dist_east = torch.clamp(x[..., 0] - (1.0 - Lpml), min=0)
    dist_south = -torch.clamp(x[..., 1] + (1.0 - Lpml), max=0)
    dist_north = torch.clamp(x[..., 1] - (1.0 - Lpml), min=0)

    sx = wavenumber * a0 * ((dist_west / Lpml) ** 2 + (dist_east / Lpml) ** 2)[..., None]
    sy = wavenumber * a0 * ((dist_north / Lpml) ** 2 + (dist_south / Lpml) ** 2)[..., None]

    ex = torch.cat((torch.ones_like(sx), -sx / wavenumber), dim=-1)
    ey = torch.cat((torch.ones_like(sy), -sy / wavenumber), dim=-1)

    A = modules.compl_div(ey, ex).repeat(1, 1, dudx1.shape[-1] // 2)
    B = modules.compl_div(ex, ey).repeat(1, 1, dudx1.shape[-1] // 2)
    C = modules.compl_mul(ex, ey).repeat(1, 1, dudx1.shape[-1] // 2)

    a, _ = diff_operators.jacobian(modules.compl_mul(A, dudx1), x)
    b, _ = diff_operators.jacobian(modules.compl_mul(B, dudx2), x)

    a = a[..., 0]
    b = b[..., 1]
    c = modules.compl_mul(modules.compl_mul(C, squared_slowness), wavenumber ** 2 * y)

    diff_constraint_hom = a + b + c
    diff_constraint_on = torch.where(source_boundary_values != 0.,
                                     diff_constraint_hom - source_boundary_values,
                                     torch.zeros_like(diff_constraint_hom))
    diff_constraint_off = torch.where(source_boundary_values == 0.,
                                      diff_constraint_hom,
                                      torch.zeros_like(diff_constraint_hom))
    if full_waveform_inversion:
        data_term = torch.where(rec_boundary_values != 0, y - rec_boundary_values, torch.Tensor([0.]).cuda())
    else:
        data_term = torch.Tensor([0.])

        if 'pretrain' in gt:  # we are not trying to solve for velocity
            data_term = pred_squared_slowness - squared_slowness[..., 0]

    return {'diff_constraint_on': torch.abs(diff_constraint_on).sum() * batch_size / 1e3,
            'diff_constraint_off': torch.abs(diff_constraint_off).sum(),
            'data_term': torch.abs(data_term).sum() * batch_size / 1}


def sdf(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1]))
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
    # Exp      # Lapl
    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
            'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
            'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
            'grad_constraint': grad_constraint.mean() * 5e1}  # 1e1      # 5e1

# inter = 3e3 for ReLU-PE
