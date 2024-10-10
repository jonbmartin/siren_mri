import os 
import h5py
import numpy as np

split = 'train'

if split =='train':
    dir = 'multicoil_train_recon/'
    newdir = 'multicoil_train_recon_byslice/'
elif split =='test':
    dir = 'multicoil_test_recon/'
    newdir = 'multicoil_train_recon_byslice/'
elif split =='val':
    dir = 'multicoil_val_recon/'
    newdir = 'multicoil_train_recon_byslice/'
elif split =='val_small':
    dir = 'multicoil_val_recon_small/'
    newdir = 'multicoil_train_recon_byslice/'

root = '../fastMRIdata/' + dir
filenames = os.listdir(root)

for ii in range(len(filenames)):
    f = h5py.File(root + filenames[ii], "w")
    
    # return data as numpy array
    data = f['reconstruction'][()]
    f.close()

    slices, width, height = np.shape(data)

    for ss in range(slices):
        with h5py.File(newdir+'slice_'+str(ss)+"_"+filenames[ii]) as f:
            data = f.create_dataset("reconstruction",data[ss,:,:], dtype='float32')
            f.close()

