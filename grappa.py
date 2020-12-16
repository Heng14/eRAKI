import scipy.io as sio
import scipy as sp
import numpy as np
import time
from scipy.linalg import svd
from utils import *


# ## Grappa
# Notes
#-------
# 1) Only use odd kernel size
# 2) samples.shape and acs.shape should be [num_chan, N1, N2]
# 3) Both samples and acs should be zero filled, and have the same matrix size as the original fully sampled image
# 4) The parity of each image dimension must match the parity of the corresponding acs dimension
# 5) This function doesn't substitute the ACS back after reconstruction

def grappa(samples, acs, Rx, Ry, num_acs, shift_x, shift_y, kernel_size=np.array([3,3]), lambda_tik=0):
    start_t = time.time()
    # Set Initial Parameters
    #------------------------------------------------------------------------------------------
    [num_chan, N1, N2] = samples.shape
    N = np.array([N1, N2]).astype(int)
    acs_start_index_x = N1//2 - num_acs[0]//2 #inclusive
    acs_start_index_y = N2//2 - num_acs[1]//2 #inclusive
    acs_end_index_x = np.int(np.ceil(N1/2)) + num_acs[0]//2
    acs_end_index_y = np.int(np.ceil(N2/2)) + num_acs[1]//2
    
    kspace_sampled = np.zeros(samples.shape, dtype=samples.dtype)
    kspace_sampled[:] = samples[:]
    
    kspace_acs = np.zeros(acs.shape, dtype=acs.dtype)
    kspace_acs[:] = acs[:]
    
    kspace_acs_crop = np.zeros([num_chan, num_acs[0], num_acs[1]], dtype=acs.dtype)
    kspace_acs_crop[:,:,:] = kspace_acs[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y]
    
    #Kernel Side Size
    kernel_hsize = (kernel_size // 2).astype(int)

    #Padding
    pad_size = (kernel_hsize * [Rx,Ry]).astype(int)
    N_pad = N + 2 * pad_size

    # Beginning/End indices for kernels in the acs region
    ky_begin_index = (Ry*kernel_hsize[1]).astype(int)
    ky_end_index = (num_acs[1] - Ry*kernel_hsize[1] - 1 - np.amax(shift_y)).astype(int)

    kx_begin_index = (Rx*kernel_hsize[0]).astype(int)
    kx_end_index = (num_acs[0] - Rx*kernel_hsize[0] - 1 - np.amax(shift_x)).astype(int)

    # Beginning/End indices for kernels in the full kspace
    Ky_begin_index = (Ry*kernel_hsize[1]).astype(int)
    Ky_end_index = (N_pad[1] - Ry*kernel_hsize[1] - 1).astype(int)

    Kx_begin_index = (Rx*kernel_hsize[0]).astype(int)
    Kx_end_index = (N_pad[0] - Rx*kernel_hsize[0] - 1).astype(int)

    # Count the number of kernels that fit the acs region
    ind = 0
    for i in range(ky_begin_index, ky_end_index+1):
        for j in range(kx_begin_index, kx_end_index+1):
            ind +=1

    num_kernels = ind

    # Initialize right hand size and acs_kernel matrices
    target_data = np.zeros([num_kernels, num_chan, Rx, Ry], dtype=samples.dtype)
    kernel_data = np.zeros([num_chan, kernel_size[0], kernel_size[1]], dtype=samples.dtype)
    acs_data = np.zeros([num_kernels, kernel_size[0] * kernel_size[1] * num_chan], dtype=samples.dtype)

    # Get kernel and target data from the acs region
    #------------------------------------------------------------------------------------------
    print('Collecting kernel and target data from the acs region')
    print('-'*50)
    kernel_num = 0
    for ky in range(ky_begin_index, ky_end_index + 1):
        print('ky: ' + str(ky))
        for kx in range(kx_begin_index, kx_end_index + 1):
            # Get kernel data
            for nchan in range(0,num_chan):
                kernel_data[nchan, :, :] = kspace_acs_crop[nchan, 
                                                           shift_x[nchan] + kx - (kernel_hsize[0]*Rx): shift_x[nchan] + kx + (kernel_hsize[0]*Rx)+1: Rx, 
                                                           shift_y[nchan] + ky - (kernel_hsize[1]*Ry): shift_y[nchan] + ky + (kernel_hsize[1]*Ry)+1: Ry]

            acs_data[kernel_num, :] = kernel_data.flatten()

            # Get target data
            for rx in range(0,Rx):
                for ry in range(0,Ry):
                    if rx != 0 or ry != 0:
                        for nchan in range(0,num_chan):
                            target_data[kernel_num,:,rx,ry] = kspace_acs_crop[:,
                                                                              shift_x[nchan] + kx - rx,
                                                                              shift_y[nchan] + ky - ry]

            # Move to the next kernel
            kernel_num += 1
            
    print()

    # Tikhonov regularization
    #------------------------------------------------------------------------------------------
    #U, S, Vh = sp.linalg.svd(acs_data, full_matrices=False)
    U, S, Vh = svd(acs_data, full_matrices=False)
    
    print('Condition number: ' + str(np.max(np.abs(S))/np.min(np.abs(S))))
    print()
    
    S_inv = np.conjugate(S) / (np.square(np.abs(S)) + lambda_tik)
    acs_data_inv = np.transpose(np.conjugate(Vh)) @ np.diag(S_inv) @ np.transpose(np.conjugate(U));

    # Get kernel weights
    #------------------------------------------------------------------------------------------
    print('Getting kernel weights')
    print('-'*50)
    kernel_weights = np.zeros([num_chan, kernel_size[0] * kernel_size[1] * num_chan, Rx, Ry], dtype=samples.dtype)

    for rx in range(0,Rx):
        print('rx: ' + str(rx))
        for ry in range(0,Ry):
            print('ry: ' + str(ry))
            if rx != 0 or ry != 0:
                for nchan in range(0,num_chan):
                    print('Channel: ' + str(nchan+1))
                    if lambda_tik == 0:
                        kernel_weights[nchan, :, rx, ry], resid, rank, s = np.linalg.lstsq(acs_data,target_data[:, nchan, rx, ry], rcond=None)
                    else:
                        kernel_weights[nchan, :, rx, ry] = acs_data_inv @ target_data[:, nchan, rx, ry]
                        
    print()
    print ("grappa learning time: ", time.time()- start_t)
    start_t = time.time()
    # Reconstruct unsampled points
    #------------------------------------------------------------------------------------------
    print('Reconstructing unsampled points')
    print('-'*50)
    kspace_recon = np.pad(kspace_sampled, ((0, 0), (pad_size[0],pad_size[0]), (pad_size[1],pad_size[1])), 'constant')
    data = np.zeros([num_chan, kernel_size[0] * kernel_size[1]], dtype=samples.dtype)

    for ky in range(Ky_begin_index, Ky_end_index+1, Ry):
        print('ky: ' + str(ky))
        for kx in range(Kx_begin_index, Kx_end_index+1, Rx):

            for nchan in range(0,num_chan):
                data[nchan, :] = (kspace_recon[nchan,
                                               shift_x[nchan] + kx - (kernel_hsize[0]*Rx): shift_x[nchan] + kx + (kernel_hsize[0]*Rx)+1: Rx, 
                                               shift_y[nchan] + ky - (kernel_hsize[1]*Ry): shift_y[nchan] + ky + (kernel_hsize[1]*Ry)+1: Ry]).flatten()


            for rx in range(0,Rx):
                for ry in range(0,Ry):
                    if rx != 0 or ry != 0:
                        for nchan in range(0,num_chan):
                            interpolation = np.dot(kernel_weights[nchan, :, rx, ry] , data.flatten())
                            kspace_recon[nchan, shift_x[nchan] + kx - rx, shift_y[nchan] + ky - ry] = interpolation

    # Get the image back
    #------------------------------------------------------------------------------------------
    kspace_recon = kspace_recon[:, pad_size[0]:-pad_size[0], pad_size[1]:-pad_size[1]]  
    img_grappa = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_recon, axes=(1,2)), axes=(1,2)), axes=(1,2))
    
    print()
    print('GRAPPA reconstruction complete.')
    print ("grappa recon time: ", time.time()- start_t)
    return kspace_recon, img_grappa


def show_grappa_results(image_original, kspace_recon, sens_resize_i, debug_path, i):

    # Get the RAKI coil images
    coil_images_raki_sub = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_recon, axes=(1,2)), axes=(1,2)), axes=(1,2))

    # Visualize
    print('Axis Labels: ' + '[num_coils, N1, N2]')
    print('RAKI Coil Images Shape: ' + str(coil_images_raki_sub.shape))
    print('RAKI Coil Images Data Type: ' + str(coil_images_raki_sub.dtype))

    #raki_rssq_image_sub = np.sqrt(np.sum(np.square(np.abs(coil_images_raki_sub)),axis=0))
    raki_rssq_image_sub = np.sum(np.conj(sens_resize_i) * coil_images_raki_sub, axis=0)/np.sum(np.square(np.abs(sens_resize_i)), axis=0)
    raki_rssq_image_sub = np.abs(raki_rssq_image_sub)


    image_ref0 = image_original.copy()
    mask = image_ref0 > 0.6 * np.mean(image_ref0)
    mask = 1.0 * mask
    image_original = image_original * mask
    raki_rssq_image_sub = raki_rssq_image_sub * mask

    #mosaic(raki_rssq_image_sub, 1, 1, 1, [0,150], fig_title=f'grappa_rssq_image_sub_{i}', fig_base_path=debug_path, fig_size=(10,10), num_rot=0)
    #mosaic(image_original, 1, 1, 1, [0,150], fig_title=f'image_original_{i}', fig_base_path=debug_path, fig_size=(10,10), num_rot=0)

    #imsave('img_ori.jpg', image_original)
    #imsave('img_recon_raki.jpg', raki_rssq_image_sub)

    rmse_raki_sub = np.sqrt(np.sum(np.square(np.abs(image_original - raki_rssq_image_sub)),axis=(0,1))) / np.sqrt(np.sum(np.square(np.abs(image_original)),axis=(0,1)))
    print('rmse_raki: ' + str(rmse_raki_sub))

    rmse_raki_sub_str = str(rmse_raki_sub)
    print('rmse_raki: ' + str(rmse_raki_sub))
    return rmse_raki_sub

    comparison_epti = np.concatenate((image_original, raki_rssq_image_sub), axis=-1)
    comparison_epti = comparison_epti
    im_recon_error_epti = np.abs(image_original - raki_rssq_image_sub)
    im_recon_error_epti_scale = im_recon_error_epti * 5
    im_gr = np.concatenate((comparison_epti, im_recon_error_epti_scale), axis=-1)
    mosaic(im_gr, 1, 1, 1, [0,150], fig_title=f'image_error_{i}', fig_base_path=debug_path, fig_size=(30,10), num_rot=0)
    #im_gr = np.concatenate((comparison_epti, im_recon_error_epti_scale), axis=-1)
    #imsave(osp.join(f'comparison_recon_raki.jpg'), comparison_epti)
    #imsave(osp.join(f'comparison_recon_raki_error.jpg'), im_recon_error_epti_scale)
    sio.savemat(osp.join(debug_path, f'recon_grappa_error_{i}'),{'data':im_recon_error_epti})
    return rmse_raki_sub





