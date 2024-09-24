'''Implements a generic training loop. With DDP functionality
'''

import torch
import utils
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil

import tempfile
import os

import torch
from torch.distributed import init_process_group, destroy_process_group



def train_ddp(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None,
          fourier_feat_transformer=None, device=0, ddp_run=False, accumulation_steps=1,):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    if device==0:
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)
        else:
            os.makedirs(model_dir, exist_ok=True)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir, exist)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):

            # DDP: need to set epoch to make shuffling work properly
            train_dataloader.sampler.set_epoch(epoch)

            # TODO: currently model is saved across ALL ddp processes
            if not epoch % epochs_til_checkpoint and epoch and device==0:
                torch.save(model.module.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                # TODO: should also apply encoding to GT coords????? Not sure.
                if fourier_feat_transformer is None:
                    pass
                else:
                    model_input['coords'] = fourier_feat_transformer(model_input['coords'])

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
                
                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    # TODO: just disabling to see if DDP runs
                    #summary_fn(model, model_input, gt, model_output, writer, total_steps)
                del model_output, losses
                
                # Backward pass
                train_loss = train_loss/accumulation_steps
                train_loss.backward()

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                # weight update
                # accumulation steps for gradient accumulation
                if (step+1) % accumulation_steps == 0 or (step+1 ==len(train_dataloader)):
                    optim.step()
                    optim.zero_grad()


                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:

                                if fourier_feat_transformer is None:
                                    pass
                                else:
                                    model_input['coords'] = fourier_feat_transformer(model_input['coords'])

                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)

                                for loss_name, loss in val_loss.items():
                                    single_loss = val_loss['img_loss']
                                val_losses.append(single_loss)
                            
                            del model_output, gt

                            mean_val_loss = torch.mean(torch.stack(val_losses))
                            writer.add_scalar("val_loss", mean_val_loss, total_steps)
                        tqdm.write(f"val loss (img_loss_only): {mean_val_loss}")

                        model.train()

                total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))
        
        final_val = mean_val_loss
        return final_val
        

def ddp_setup(rank: int, world_size: int):
  """
  Args:
      rank: Unique identifier of each process
     world_size: Total number of processes
  """
  os.environ["MASTER_ADDR"] = "iis-cn1-aa57"
  os.environ["MASTER_PORT"] = "12355"
  torch.cuda.set_device(rank)
  init_process_group(backend="nccl", rank=rank, world_size=world_size)
