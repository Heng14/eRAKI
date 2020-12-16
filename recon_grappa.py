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
from skimage.metrics import structural_similarity as ssim
import numpy as np
import numpy.matlib
import time
import os
from grappa import *
from utils import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# # Read and Prepare the Data
# ## Read Fully Sampled Data
# Read the fully sampled image


def process_one(kdata, kdata0, sens_resize_i, i, debug_path='debug_fig_sq'):

    # Crop the image (if necessary)
    #img_coils = img_coils[:,:,:]

    # Get the dimensions
    kspace_fully_sampled = kdata
    [num_chan, N1, N2] = kspace_fully_sampled.shape


    #img_coils = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_fully_sampled, axes=(1,2)), axes=(1,2)), axes=(1,2))
    #image_original = np.sqrt(np.sum(np.square(np.abs(img_coils)),axis=0))

    img_coils0 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kdata0, axes=(0,1)), axes=(0,1)), axes=(0,1))
    image_original0 = np.abs(img_coils0)


    #mosaic(image_original, 1, 1, 1, [0,0.015], fig_title=f'image_original_{i}', fig_base_path=debug_path+'/all_slices', fig_size=(10,10), num_rot=0)
    #mosaic(image_original, 1, 1, 1, [0,0.015], fig_title=f'image_original_{i}', fig_base_path='debug_path', fig_size=(10,10), num_rot=0)
    #mosaic(image_original0, 1, 1, 1, [0,0.015], fig_title=f'image_original0_{i}', fig_base_path='debug_path', fig_size=(10,10), num_rot=0)
    #return

    #num_acs = (40,N2)
    num_acs = (24,24)
    Rx = 3
    Ry = 3 #acc direction


    acs_start_index_x = N1//2 - num_acs[0]//2 # inclusive
    acs_start_index_y = N2//2 - num_acs[1]//2 # inclusive
    acs_end_index_x = np.int(np.ceil(N1/2)) + num_acs[0]//2 # exclusive
    acs_end_index_y = np.int(np.ceil(N2/2)) + num_acs[1]//2 # exclusive

    kspace_acs_zero_filled = np.zeros(kspace_fully_sampled.shape, dtype=kspace_fully_sampled.dtype)
    kspace_acs_cropped = np.zeros([num_chan, num_acs[0], num_acs[1]], dtype=kspace_fully_sampled.dtype)

    kspace_acs_zero_filled[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y] = kspace_fully_sampled[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y]
    kspace_acs_cropped[:,:,:] = kspace_fully_sampled[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y]


    # ## Undersample
    # Note: The sampling pattern might look incorrect because of aliasing while displaying the figure in smaller sizes. The data in the array itself is correct

    kspace_undersampled_zero_filled = np.zeros(kspace_fully_sampled.shape, dtype=kspace_fully_sampled.dtype)
    kspace_undersampled_zero_filled[:,0:N1:Ry,0:N2:Rx] = kspace_fully_sampled[:,0:N1:Ry,0:N2:Rx]


    grappa_shift_x = np.zeros(num_chan, dtype=int)
    grappa_shift_y = np.zeros(num_chan, dtype=int)

    grappa_coil_images_without_sub = np.zeros(kspace_undersampled_zero_filled.shape, kspace_undersampled_zero_filled.dtype)
    grappa_coil_kspaces_without_sub = np.zeros(kspace_undersampled_zero_filled.shape, dtype=kspace_undersampled_zero_filled.dtype)

    start_t = time.time()
    grappa_coil_kspaces_without_sub, grappa_coil_images_without_sub = grappa(kspace_undersampled_zero_filled,kspace_acs_zero_filled, Ry, Rx, num_acs, grappa_shift_x, grappa_shift_y,kernel_size=np.array([3, 3]), lambda_tik=1e-9)
    print ("grappa time: ", time.time()- start_t)
    #np.save("grappa_coil_kspaces_without_sub.npy",grappa_coil_kspaces_without_sub)
    #np.save("grappa_coil_images_without_sub.npy",grappa_coil_images_without_sub)
    #grappa_coil_kspaces_without_sub = np.load("grappa_coil_kspaces_without_sub.npy")
    #grappa_coil_images_without_sub = np.load("grappa_coil_images_without_sub.npy")

    # Substitute ACS
    acs_mask = np.zeros((num_chan, N1, N2), dtype=bool)
    acs_mask[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y] = True

    grappa_coil_kspaces_sub = (grappa_coil_kspaces_without_sub * np.invert(acs_mask).astype(int)) + kspace_acs_zero_filled[:,:,:]

    #grappa_coil_images_sub = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grappa_coil_kspaces_sub, axes=(1,2)), axes=(1,2)), axes=(1,2))

    print (grappa_coil_kspaces_sub.shape)
    #print (grappa_coil_images_sub.shape)

    np.save(f'{debug_path}/grappa_coil_kspaces_sub_{i}.npy',grappa_coil_kspaces_sub)

    rmse = show_grappa_results(image_original0, grappa_coil_kspaces_sub, sens_resize_i, debug_path, i)
    return rmse



def show_raki_results_3d(kdata_coils0_np, debug_path0, sens_resize):
    kdata_coils0 = np.transpose(kdata_coils0_np, (1,0,2))
    print (kdata_coils0.shape)
    print (sens_resize.shape)
    img_coils0 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kdata_coils0, axes=(0,1)), axes=(0,1)), axes=(0,1))
    image_original0 = np.abs(img_coils0)
    size_t = len(kdata_coils0[0, 0, :])

    coil_images_raki_sub_list = [] 
    for i in range(size_t):
        name_image = f'{debug_path0}/grappa_coil_kspaces_sub_{i}.npy'
        kspace_recon = np.load(name_image)
        #kspace_recon = np.transpose(kspace_recon,(2,1,0))
        #print (kspace_recon.shape)
        #raise
        coil_images_raki_sub_tmp = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_recon, axes=(1,2)), axes=(1,2)), axes=(1,2))
        coil_images_raki_sub_list.append(coil_images_raki_sub_tmp)
    coil_images_raki_sub_np = np.array(coil_images_raki_sub_list)
    coil_images_raki_sub = np.transpose(coil_images_raki_sub_np,(1,2,3,0))
    print (coil_images_raki_sub.shape)
    #raise

    raki_rssq_image_sub0 = np.sum(np.conj(sens_resize) * coil_images_raki_sub, axis=0)/np.sum(np.square(np.abs(sens_resize)), axis=0)
    raki_rssq_image_sub0 = np.abs(raki_rssq_image_sub0)

    print (raki_rssq_image_sub0.shape)
    print (image_original0.shape)
    sio.savemat(osp.join(debug_path0, f'recon_image_3d.mat'),{'data':raki_rssq_image_sub0})
    #sio.savemat(osp.join(debug_path0, f'ori_image_3d.mat'),{'data':image_original0})
    #raise

    # using bet mask
    mask0 = sio.loadmat('../bet_mask/msk.mat')
    mask = mask0['msk']
    mask = np.transpose(mask,(1,0,2))

    #mask = image_original0 > 0.6 * np.mean(image_original0)
    #print (mask.shape)
    #raise

    image_original = image_original0 * mask
    raki_rssq_image_sub = raki_rssq_image_sub0 * mask

    sio.savemat(osp.join(debug_path0, f'recon_image_3d_bet.mat'),{'data':raki_rssq_image_sub})
    #sio.savemat(osp.join(debug_path0, f'ori_image_3d_bet.mat'),{'data':image_original})
    #raise

    #image_original = image_original[...,50:100]
    #raki_rssq_image_sub = raki_rssq_image_sub[...,50:100]

    rmse_raki_sub = np.sqrt(np.sum(np.square(np.abs(image_original - raki_rssq_image_sub)),axis=(0,1,2))) / np.sqrt(np.sum(np.square(np.abs(image_original)),axis=(0,1,2)))

    rmse_str = str(rmse_raki_sub)
    print('3d betmask rmse_raki: ' + str(rmse_str))

    ssim_e = ssim(raki_rssq_image_sub, image_original, data_range=image_original.max() - image_original.min())
    ssim_e_str = str(ssim_e)
    print ('3d betmask ssim: ', ssim_e_str)


    debug_path = os.path.join(debug_path0, f'3d_betmask_{rmse_str}')
    os.makedirs(debug_path, exist_ok=True)

    debug_path_ssim = os.path.join(debug_path0, f'3d_betmask_ssim_{ssim_e_str}')
    os.makedirs(debug_path_ssim, exist_ok=True)
    raise

    size_t = len(image_original0[0, 0, :])
    rmse_list = []
    #for i in range(75,90):
    #for i in range(50,150):
    for i in range(size_t):
        image_original_i = image_original[:,:,i]
        raki_rssq_image_sub_i = raki_rssq_image_sub[:,:,i]

        mosaic(raki_rssq_image_sub_i, 1, 1, 1, [0,150], fig_title=f'raki_rssq_image_sub_{i}', fig_base_path=debug_path, fig_size=(10,10), num_rot=0)
        mosaic(image_original_i, 1, 1, 1, [0,150], fig_title=f'image_original_{i}', fig_base_path=debug_path, fig_size=(10,10), num_rot=0)

        rmse_raki_sub = np.sqrt(np.sum(np.square(np.abs(image_original_i - raki_rssq_image_sub_i)),axis=(0,1))) / np.sqrt(np.sum(np.square(np.abs(image_original_i)),axis=(0,1)))
        print('rmse_raki: ' + str(rmse_raki_sub))
        rmse_list.append(rmse_raki_sub)
        #continue

        #imsave('img_ori.jpg', image_original)
        #imsave('img_recon_raki.jpg', raki_rssq_image_sub)

        comparison_epti = np.concatenate((image_original_i, raki_rssq_image_sub_i), axis=-1)
        im_recon_error_epti = np.abs(image_original_i - raki_rssq_image_sub_i)
        im_recon_error_epti_scale = im_recon_error_epti * 5
        im_gr = np.concatenate((comparison_epti, im_recon_error_epti_scale), axis=-1)
        mosaic(im_gr, 1, 1, 1, [0,150], fig_title=f'image_error_{i}', fig_base_path=debug_path, fig_size=(30,10), num_rot=0)
        #im_gr = np.concatenate((comparison_epti, im_recon_error_epti_scale), axis=-1)
        #imsave(osp.join(f'comparison_recon_raki.jpg'), comparison_epti)
        #imsave(osp.join(f'comparison_recon_raki_error.jpg'), im_recon_error_epti_scale)
        sio.savemat(osp.join(debug_path, f'recon_raki_error_{i}'),{'data':im_recon_error_epti})

    rmse_np = np.array(rmse_list)
    data = pd.DataFrame(rmse_np)
    writer = pd.ExcelWriter(f'{debug_path}/rmse.xlsx')
    data.to_excel(writer, 'rmse', float_format='%.6f')
    writer.save()
    writer.close()



def run(debug_path='debug_fig_grappa_24-24_Rx3Ry3_k3_tik1e-9'):
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

    os.makedirs(debug_path, exist_ok=True)
    kdata_coils = np.load('../meas_data_256-np/k3.npy')
    print (kdata_coils.shape)
    #kdata_coils0 = np.load('../meas_data_after-espirit/k1_75.npy') # data used for comparision
    #print (kdata_coils0.shape)
    ##kdata_coils0 = np.transpose(kdata_coils0, (1,0,2))
    #kdata_coils0 = np.transpose(kdata_coils0, (1,0))
    #print (kdata_coils0.shape)

    kdata_coils0_np = np.load('../meas_data_after-espirit/acs_24_24_256_3d_ks3/k3_full_samesens.npy') # data used for comparision
    kdata_coils0_np = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kdata_coils0_np, axes=2), axis=2), axes=2)

    size_t = len(kdata_coils[0, 0, 0, :])
    rmse_list = []

    sens_resize0 = np.load('../meas_data_after-espirit/acs_24_24_256_3d_ks3/sens_resize.npy')
    sens_resize = sens_resize0['real'] + np.multiply(sens_resize0['imag'], 1j)
    sens_resize = np.transpose(sens_resize,(3,1,0,2))

    #show_raki_results_3d(kdata_coils0_np, debug_path, sens_resize)
    #raise


    #for i in range(75,90):
    #for i in range(50,150):
    for i in range(size_t):
        kdata_tmp = kdata_coils[:, :, :, i]
        print (kdata_tmp.shape)
        #kdata_coils0 = np.load(f'../meas_data_after-espirit/full/k1_{i}.npy') # data used for comparision
        kdata_coils0 = kdata_coils0_np[:, :, i] # data used for comparision
        kdata_coils0 = np.transpose(kdata_coils0, (1,0))
        print (kdata_coils0.shape)

        kdata_i = np.transpose(kdata_tmp, (2,0,1))
        #np.save('test', kdata_i)
        
        rmse_tmp = process_one(kdata_i, kdata_coils0, sens_resize[:,:,:,i], i, debug_path=debug_path)
        rmse_list.append(rmse_tmp)
        #raise
    rmse_np = np.array(rmse_list)
    data = pd.DataFrame(rmse_np)
    writer = pd.ExcelWriter(f'{debug_path}/rmse.xlsx')
    data.to_excel(writer, 'rmse', float_format='%.6f')
    writer.save()
    writer.close()


if __name__ == '__main__':
    import fire
    fire.Fire(run)


















