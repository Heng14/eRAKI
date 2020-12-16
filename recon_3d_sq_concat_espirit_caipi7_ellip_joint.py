#!/usr/bin/env python
import scipy.io as sio
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
import warnings
import nvgpu
from PIL import Image
import scipy.io as sio
import numpy as np
import numpy.matlib
import time
import os
#from grappa import *
#from raki_3d_sq_concat_polyfit import *
from raki_3d_sq_concat_espirit_5layers_caipi7_ellip_joint import *
#from raki_3d_sq_concat_espirit_caipi7_ellip import *
from utils import *
import h5py
import pandas as pd
from skimage.draw import ellipse
#from spark import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# # Read and Prepare the Data
# ## Read Fully Sampled Data
# Read the fully sampled image
def process_one(kdata, kdata_ref, acs_out, debug_path='debug_fig_sq'):

    # Transpose to get the channel index in position 0, and rotate the image (if necessary)
    #img_coils = np.rot90(np.transpose(img_coils, (2,0,1)),k=-1,axes=(1,2))
    #print (img_coils.shape)

    # Crop the image (if necessary)
    #img_coils = img_coils[:,:,:]

    # Get the dimensions
    kspace_fully_sampled = kdata
    [num_chan, N1, N2, Nt] = kspace_fully_sampled.shape


    #mosaic(kspace_fully_sampled[:,:,:], 4, 8, 1, [0,2e-5], fig_title='kspace_fully_sampled', fig_size=(15,15), num_rot=0)
    # Get the rssq image

    '''
    kspace_fully_sampled = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kspace_fully_sampled, axes=3), axis=3), axes=3)
    img_coils = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_fully_sampled, axes=(1,2)), axes=(1,2)), axes=(1,2))
    image_original0 = np.sqrt(np.sum(np.square(np.abs(img_coils)),axis=0))


    for i in range(75,90):
        image_original = image_original0[:,:,i]
        mosaic(image_original, 1, 1, 1, [0,150], fig_title=f'image_original_{i}', fig_base_path=debug_path, fig_size=(10,10), num_rot=0)
    raise
    '''


    #num_acs = (24, 24, 256)
    num_acs = (22, 22, 256)
    #num_acs = (N1, 30, 40)
    Rx = 4
    Ry = 2

    acs_x, acs_y, acs_t = num_acs

    acs_start_index_x = N1//2 - num_acs[0]//2 # inclusive
    acs_start_index_y = N2//2 - num_acs[1]//2 # inclusive

    acs_end_index_x = np.int(np.ceil(N1/2)) + num_acs[0]//2 # exclusive
    acs_end_index_y = np.int(np.ceil(N2/2)) + num_acs[1]//2 # exclusive

    acs_start_index_t = Nt//2 - num_acs[2]//2 # inclusive
    acs_end_index_t = np.int(np.ceil(Nt/2)) + num_acs[2]//2 # exclusive

    #print (acs_start_index_x, acs_end_index_x)
    #print (acs_start_index_y, acs_end_index_y)
    #print (acs_start_index_t, acs_end_index_t)

    #kspace_undersampled_zero_filled = np.zeros(kspace_fully_sampled.shape, dtype=kspace_fully_sampled.dtype)
    #kspace_undersampled_zero_filled[:,0:N1:Ry,0:N2:Rx,:] = kspace_fully_sampled[:,0:N1:Ry,0:N2:Rx,:]
    #print (kspace_undersampled_zero_filled.shape)

    print (kspace_fully_sampled.shape)
    c, w, h, z = kspace_fully_sampled.shape
    mask_ellip = np.zeros((w, h), dtype=np.uint8)
    rr, cc = ellipse(w//2, h//2, w//2, h//2)
    mask_ellip[rr, cc] = 1
    print (mask_ellip.shape)
    mask_ellip = np.expand_dims(mask_ellip, axis=0)
    mask_ellip = np.repeat(mask_ellip, c, axis=0)
    mask_ellip = np.expand_dims(mask_ellip, axis=-1)
    mask_ellip = np.repeat(mask_ellip, z, axis=-1)
    print (mask_ellip.shape)


    mask = np.zeros(kspace_fully_sampled.shape)

    mask[0:32, 0:N1:7, 0:N2:7, :] = 1
    mask[0:32, 1:N1:7, 3:N2:7, :] = 1
    mask[0:32, 2:N1:7, 6:N2:7, :] = 1
    mask[0:32, 3:N1:7, 2:N2:7, :] = 1
    mask[0:32, 4:N1:7, 5:N2:7, :] = 1
    mask[0:32, 5:N1:7, 1:N2:7, :] = 1
    mask[0:32, 6:N1:7, 4:N2:7, :] = 1

    mask[32:64, 1:N1:7, 1:N2:7, :] = 1
    mask[32:64, 2:N1:7, 4:N2:7, :] = 1
    mask[32:64, 3:N1:7, 0:N2:7, :] = 1
    mask[32:64, 4:N1:7, 3:N2:7, :] = 1
    mask[32:64, 5:N1:7, 6:N2:7, :] = 1
    mask[32:64, 6:N1:7, 2:N2:7, :] = 1
    mask[32:64, 0:N1:7, 5:N2:7, :] = 1

    mask[64:96, 2:N1:7, 2:N2:7, :] = 1
    mask[64:96, 3:N1:7, 5:N2:7, :] = 1
    mask[64:96, 4:N1:7, 1:N2:7, :] = 1
    mask[64:96, 5:N1:7, 4:N2:7, :] = 1
    mask[64:96, 6:N1:7, 0:N2:7, :] = 1
    mask[64:96, 0:N1:7, 3:N2:7, :] = 1
    mask[64:96, 1:N1:7, 6:N2:7, :] = 1


    '''
    mask[:, 0:N1:7, 0:N2:7, :] = 1
    mask[:, 1:N1:7, 3:N2:7, :] = 1
    mask[:, 2:N1:7, 6:N2:7, :] = 1
    mask[:, 3:N1:7, 2:N2:7, :] = 1
    mask[:, 4:N1:7, 5:N2:7, :] = 1
    mask[:, 5:N1:7, 1:N2:7, :] = 1
    mask[:, 6:N1:7, 4:N2:7, :] = 1
    #print (mask[10,0:16,0:16,10])
    #raise
    '''
    '''
    mask[0:16, 0:N1:7, 0:N2:7, :] = 1
    mask[0:16, 1:N1:7, 3:N2:7, :] = 1
    mask[0:16, 2:N1:7, 6:N2:7, :] = 1
    mask[0:16, 3:N1:7, 2:N2:7, :] = 1
    mask[0:16, 4:N1:7, 5:N2:7, :] = 1
    mask[0:16, 5:N1:7, 1:N2:7, :] = 1
    mask[0:16, 6:N1:7, 4:N2:7, :] = 1

    mask[16:32, 1:N1:7, 1:N2:7, :] = 1
    mask[16:32, 2:N1:7, 4:N2:7, :] = 1
    mask[16:32, 3:N1:7, 0:N2:7, :] = 1
    mask[16:32, 4:N1:7, 3:N2:7, :] = 1
    mask[16:32, 5:N1:7, 6:N2:7, :] = 1
    mask[16:32, 6:N1:7, 2:N2:7, :] = 1
    mask[16:32, 0:N1:7, 5:N2:7, :] = 1


    mask[32:48, 2:N1:7, 2:N2:7, :] = 1
    mask[32:48, 3:N1:7, 5:N2:7, :] = 1
    mask[32:48, 4:N1:7, 1:N2:7, :] = 1
    mask[32:48, 5:N1:7, 4:N2:7, :] = 1
    mask[32:48, 6:N1:7, 0:N2:7, :] = 1
    mask[32:48, 0:N1:7, 3:N2:7, :] = 1
    mask[32:48, 1:N1:7, 6:N2:7, :] = 1
    '''
    #print (mask.shape)
    #raise

    kspace_undersampled_zero_filled0 = kspace_fully_sampled * mask * mask_ellip
    #kspace_undersampled_zero_filled = kspace_undersampled_zero_filled0[0:16,...] + kspace_undersampled_zero_filled0[16:32,...] + kspace_undersampled_zero_filled0[32:48,...]
    kspace_undersampled_zero_filled = kspace_undersampled_zero_filled0[0:32,...] + kspace_undersampled_zero_filled0[32:64,...]


    kspace_acs_zero_filled = np.zeros(kspace_fully_sampled.shape, dtype=kspace_fully_sampled.dtype)
    kspace_acs_zero_filled[:, acs_start_index_x:acs_end_index_x, acs_start_index_y:acs_end_index_y,  acs_start_index_t:acs_end_index_t] = kspace_fully_sampled[:, acs_start_index_x:acs_end_index_x, acs_start_index_y:acs_end_index_y,  acs_start_index_t:acs_end_index_t]
    print (kspace_acs_zero_filled.shape)

    kspace_acs_cropped = np.zeros([num_chan, num_acs[0], num_acs[1], num_acs[2]], dtype=kspace_fully_sampled.dtype)
    kspace_acs_cropped[:,:,:,:] = kspace_fully_sampled[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y, acs_start_index_t:acs_end_index_t]


    #tmp = kdata_ref[acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y, acs_start_index_t:acs_end_index_t]
    #print (tmp.shape)
    #print (acs_out.shape)
    #print (tmp[:,12,100])
    #print (acs_out[0,:,12,100])
    #raise

    #kspace_coils_acs_out = acs_out
    kspace_coils_acs_out = acs_out[:,1:-1,1:-1,:]
    #kspace_coils_acs_out = acs_out[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y, acs_start_index_t:acs_end_index_t]
    #kspace_coils_acs_out = kspace_acs_cropped
    #kspace_coils_acs_out = np.expand_dims(kdata_ref[acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y, acs_start_index_t:acs_end_index_t], axis=0)


    '''
    tmp = np.expand_dims(kdata_ref[acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y, acs_start_index_t:acs_end_index_t], axis=0)
    print (kspace_coils_acs_out.shape)
    print (tmp.shape)
    print (kspace_coils_acs_out[0,10:12,10:12,5])
    print (tmp[0,10:12,10:12,5])
    raise
    '''

    #print (kspace_coils_acs_out.shape)
    #img_test = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_coils_acs_out[0,:,:,0], axes=(0,1)), axes=(0,1)), axes=(0,1))
    #mosaic(img_test, 1, 1, 1, [0,2e-5], fig_title='img_test', fig_base_path='debug_test', fig_size=(15,15), num_rot=0)


    #kdata_ref[acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y, acs_start_index_t:acs_end_index_t] = acs_out[0,...] #put acs back to ref

    img_coils_ref = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kdata_ref, axes=(0,1,2)), axes=(0,1,2)), axes=(0,1,2))
    #img_coils_ref = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kdata_ref, axes=(0,1)), axes=(0,1)), axes=(0,1))

    image_original_ref = np.abs(img_coils_ref)
    #image_original_ref = np.sqrt(np.sum(np.square(np.abs(img_coils_ref)),axis=0))
    print ('image_original_ref shape: ', image_original_ref.shape)



    start_t = time.time()

    name_image = f'{debug_path}/recon.mat'
    name_weight = f'{debug_path}/weight.mat'

    raki(num_chan, N1, N2, Nt, acs_start_index_x, acs_end_index_x, acs_start_index_y, acs_end_index_y, acs_start_index_t, acs_end_index_t, kspace_undersampled_zero_filled, kspace_acs_zero_filled, kspace_coils_acs_out, kspace_acs_cropped, Rx, Ry, debug_path, name_image, name_weight, train_flag=True)
    print (time.time()-start_t)
    #show_raki_results(image_original_ref, debug_path, name_image)
    show_raki_results_3d(image_original_ref, debug_path, name_image)
    #return rmse

def run(debug_path='debug_fig_acs_ks3/debug_fig_sq_concat_espirit_24-24-256_3d_Rx3Ry3_caipi7_ellip_joint_reg'):
    #dt2 = h5py.File('../meas_data_ori/k1.mat')
    #kdata_coils0 = dt2['k1'].value
    #kdata_coils = kdata_coils0['real'] + np.multiply(kdata_coils0['imag'], 1j)
    #kdata_coils = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kdata_coils, axes=3), axis=3), axes=3)
    #print (kdata_coils.shape)
    #np.save('../meas_data_256-np/k1', kdata_coils)
    #raise

    #dt2 = sio.loadmat('../meas_data_after-espirit/k1_89.mat')
    #kdata_coils = dt2['res_acs']
    ##kdata_coils = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kdata_coils, axes=2), axis=2), axes=2)
    #print (kdata_coils.shape)
    #np.save('../meas_data_after-espirit/k1_89', kdata_coils)
    #raise

    #for i in range(0,256):
    #    dt2 = sio.loadmat(f'../meas_data_after-espirit/full/k1_{i}.mat')
    #    kdata_coils = dt2['res_acs']
    #    print (kdata_coils.shape)
    #    np.save(f'../meas_data_after-espirit/full/k1_{i}', kdata_coils)
    #raise

    '''
    kdata_coils = np.load('../meas_data_after-espirit/full_3d/k1.npy')
    kdata_coils = np.expand_dims(kdata_coils, axis=-1)
    kspace_fully_sampled = np.transpose(kdata_coils, (3,1,0,2))

    kspace_fully_sampled = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kspace_fully_sampled, axes=3), axis=3), axes=3)
    img_coils = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_fully_sampled, axes=(1,2)), axes=(1,2)), axes=(1,2))
    image_original0 = np.sqrt(np.sum(np.square(np.abs(img_coils)),axis=0))


    for i in range(0,10):
        image_original = image_original0[:,:,i]
        mosaic(image_original, 1, 1, 1, [0,150], fig_title=f'image_original10_{i}', fig_base_path=debug_path, fig_size=(10,10), num_rot=0)
    raise
    '''
    '''
    acs_out0 = np.load('../meas_data_after-espirit/acs_30_40_3d_test_test/k1_3c.npy') #(240 ,30, 256), (30, 30, 256)
    acs_out1 = np.load('../meas_data_after-espirit/acs_30_40_3d_test_test/k1_2c.npy') #(240 ,30, 256), (30, 30, 256)
    print (acs_out1[10,10,10])
    acs_out1 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(acs_out1, axes=2), axis=2), axes=2)
    print (acs_out0.shape)
    print (acs_out1.shape)
    print (acs_out0[10,10,10])
    print (acs_out1[10,10,10])
    print (np.max(abs(acs_out0)))
    print (np.max(abs(acs_out1)))
    raise
    '''


    #acs_out0 = np.load('../meas_data_after-espirit/acs_30_40_3d_test/k1.npy') #(240 ,30, 256), (30, 30, 256)
    #acs_out0 = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(acs_out0, axes=2), axis=2), axes=2)
    #print (acs_out0.shape)
    #print (np.max(abs(acs_out0)))
    #raise


    os.makedirs(debug_path, exist_ok=True)
    #kdata_coils = np.load('../meas_data_np/k1.npy')
    #kdata_coils = np.load('../meas_data_np/k1.npy')

    #kdata_coils1 = np.load('../meas_data_np_gcc/calib_24_24_256_3d_16coils/k1.npy')
    #kdata_coils2 = np.load('../meas_data_np_gcc/calib_24_24_256_3d_16coils/k1.npy')
    #kdata_coils3 = np.load('../meas_data_np_gcc/calib_24_24_256_3d_16coils/k1.npy')
    #kdata_coils = np.concatenate((kdata_coils1, kdata_coils2, kdata_coils3), axis=2)
    #kdata_coils = kdata_coils1

    kdata_coils1 = np.load('../meas_data_np/k1.npy')
    kdata_coils2 = np.load('../meas_data_np/k2.npy')
    kdata_coils3 = np.load('../meas_data_np/k3.npy')
    #kdata_coils = np.concatenate((kdata_coils1, kdata_coils2), axis=2)
    kdata_coils = np.concatenate((kdata_coils1, kdata_coils2, kdata_coils3), axis=2)


    #kdata_coils1 = np.load('../meas_data_np_gcc/calib_24_24_256_3d_16coils/k1.npy')
    #kdata_coils2 = np.load('../meas_data_np_gcc/calib_24_24_256_3d_16coils_vc/k1.npy')
    #kdata_coils = np.concatenate((kdata_coils1, kdata_coils2), axis=2)
    #kdata_coils_vc = np.load('../meas_data_ori_vc/k1.npy')
    #kdata_coils = np.concatenate((kdata_coils, kdata_coils_vc), axis=2)
    #print (kdata_coils.shape)
    #print (kdata_coils_vc.shape)
    #raise

    acs_out = np.load('../meas_data_after-espirit/acs_24_24_256_3d_ks3/k1.npy') #(240 ,30, 256), (30, 30, 256))
    #acs_out = np.load('../meas_data_after-espirit/acs_24_24_256_3d_ks3/k1_zpad.npy') #(240 ,30, 256), (30, 30, 256))
    #acs_out = np.load('../meas_data_after-sumcompress/acs_30_30_256_3d/k1.npy') #(240 ,30, 256), (30, 30, 256))

    #acs_out = np.load('../meas_data_after-polycompress/acs_30_40_256_3d_img/im_coil_combined_acs.npy') #(240 ,30, 256), (30, 30, 256))
    #acs_out = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(acs_out, axes=(0,1,2)), axes=(0,1,2)), axes=(0,1,2))

    #acs_out = np.load('../meas_data_after-espirit/acs_30_40_3d_test/k1.npy') #(240 ,30, 256), (30, 30, 256)
    #acs_out = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(acs_out, axes=2), axis=2), axes=2)
    acs_out = np.expand_dims(acs_out, axis=-1)
    #print (acs_out.shape)
    #raise

    kdata_coils0_np = np.load('../meas_data_after-espirit/acs_24_24_256_3d_ks3/k1_full_samesens.npy') # data used for comparision
    #kdata_coils0_np = np.load('../meas_data_after-espirit/acs_24_24_256_3d_ks3/k1_full_samesens_zpad.npy') # data used for comparision
    #kdata_coils0_np = np.load('../meas_data_after-espirit/acs_24_24_256_3d/k1_full_samesens_zpad.npy') # data used for comparision

    #print (kdata_coils.shape)
    #kdata_coils0 = np.load('../meas_data_after-espirit/full/k1_75.npy') # data used for comparision
    #print (kdata_coils0.shape)
    ##kdata_coils0 = np.transpose(kdata_coils0, (1,0,2))
    #kdata_coils0 = np.transpose(kdata_coils0, (1,0))
    #print (kdata_coils0.shape)
    size_t = len(kdata_coils[0, 0, 0, :])
    rmse_list = []
    acs_out_list = []
    kdata_coils0_list = []
    #for i in range(75,90):
    #for i in range(size_t):
    #    kdata_coils0_tmp = np.load(f'../meas_data_after-espirit/full/k1_{i}.npy') # data used for comparision

    #    kdata_coils0_list.append(kdata_coils0_tmp)

        #acs_out_tmp = np.load(f'../meas_data_after-espirit/acs_30_240_6/k1_{i}.npy')
        #acs_out_tmp = np.expand_dims(acs_out_tmp, axis=-1)
        #acs_out_list.append(acs_out_tmp)
    #acs_out_np = np.array(acs_out_list)
    #kdata_coils0_np = np.array(kdata_coils0_list)

    '''
    kdata_coils = np.transpose(kdata_coils, (2,0,1,3))
    acs_out = np.transpose(acs_out, (3,1,0,2))
    #acs_out = np.transpose(acs_out_np, (3,2,1,0))
    kdata_coils0 = np.transpose(kdata_coils0_np, (1,0,2))
    #kdata_coils0 = kdata_coils
    '''

    kdata_coils = np.transpose(kdata_coils, (2,1,0,3))
    acs_out = np.transpose(acs_out, (3,0,1,2))
    #acs_out = np.transpose(acs_out_np, (3,2,1,0))
    kdata_coils0 = np.transpose(kdata_coils0_np, (0,1,2))
    #kdata_coils0 = kdata_coils


    print (kdata_coils.shape)
    print (acs_out.shape)
    print (kdata_coils0.shape)

    print (np.max(abs(kdata_coils)))
    print (np.max(abs(acs_out)))
    print (np.max(abs(kdata_coils0)))

    process_one(kdata_coils, kdata_coils0, acs_out, debug_path=debug_path)
    

if __name__ == '__main__':
    import fire
    fire.Fire(run)














