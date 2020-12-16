import scipy.io as sio
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
import warnings
import nvgpu
from PIL import Image
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import scipy.io as sio
import numpy as np
import numpy.matlib
import time
import os
import os.path as osp
from utils import *
from skimage.io import imsave
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.disable_v2_behavior()
#RAKI
#(The code in this section is directly copied from the code supplied with the original paper)
# Reformat input data to match the supplied RAKI code


def weight_variable(shape,vari_name):
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial,name = vari_name)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv2d_dilate(x, W, Rx, Ry):
    return tf.nn.convolution(x, W,padding='VALID',dilation_rate = [Rx, Ry])

#### LEARNING FUNCTION ####
def learning(ACS,target_input,Rx, Ry,sess, ACS_dim_X, ACS_dim_Y, ACS_dim_Z, target_dim_X,target_dim_Y,target_dim_Z, target, kernel_x_1, kernel_x_2, kernel_y_1, kernel_y_2, layer1_channels, layer2_channels, kernel_last_x, kernel_last_y, LearningRate, MaxIteration):

    [target_dim0,target_dim1,target_dim2,target_dim3] = np.shape(target)
    input_ACS = tf.placeholder(tf.float32, [1, ACS_dim_X,ACS_dim_Y,ACS_dim_Z])
    input_Target = tf.placeholder(tf.float32, [1, target_dim_X,target_dim_Y,target_dim3])

    Input = tf.reshape(input_ACS, [1, ACS_dim_X, ACS_dim_Y, ACS_dim_Z])

    W_conv1 = weight_variable([kernel_x_1, kernel_y_1, ACS_dim_Z, layer1_channels],'W1')
    #h_conv1 = tf.nn.relu(conv2d_dilate(Input, W_conv1,accrate_input))
    h_conv1 = conv2d_dilate(Input, W_conv1, Rx, Ry)

    W_conv2 = weight_variable([kernel_x_2, kernel_y_2, layer1_channels, layer2_channels],'W2')
    h_conv2 = tf.nn.relu(conv2d_dilate(h_conv1, W_conv2, Rx, Ry))

    W_conv3 = weight_variable([kernel_last_x, kernel_last_y, layer2_channels, target_dim3],'W3')
    h_conv3 = conv2d_dilate(h_conv2, W_conv3, Rx, Ry)

    #error_norm = tf.norm(input_Target - h_conv3)

    #error_norm = (tf.norm(input_Target - h_conv3, ord=2) + tf.norm(input_Target - h_conv3, ord=1))*0.5

    error_norm = (tf.norm(input_Target - h_conv3, ord=2) + tf.norm(input_Target - h_conv3, ord=1))*0.5 + 0.2*(tf.nn.l2_loss(W_conv1)+0.9*tf.nn.l2_loss(W_conv2)+0.8*tf.nn.l2_loss(W_conv3))


    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(LearningRate, global_step=global_step,decay_steps=50,decay_rate=0.95)
    train_step = tf.train.AdamOptimizer(lr).minimize(error_norm)
    #train_step = tf.train.AdamOptimizer(LearningRate).minimize(error_norm)

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    error_prev = 1
    for i in range(MaxIteration+1):

        sess.run(train_step, feed_dict={input_ACS: ACS, input_Target: target, global_step:i})
        if i % 100 == 0:
            error_now=sess.run(error_norm,feed_dict={input_ACS: ACS, input_Target: target})
            print('The',i,'th iteration gives an error',error_now)



    #error = sess.run(error_norm,feed_dict={input_ACS: ACS, input_Target: target})
    return sess.run([W_conv1,W_conv2,W_conv3])


def cnn_3layer(input_kspace,w1,b1,w2,b2,w3,b3,Rx,Ry,sess):
    #h_conv1 = tf.nn.relu(conv2d_dilate(input_kspace, w1,acc_rate))
    h_conv1 = conv2d_dilate(input_kspace, w1,Rx,Ry)
    h_conv2 = tf.nn.relu(conv2d_dilate(h_conv1, w2,Rx,Ry))
    h_conv3 = conv2d_dilate(h_conv2, w3,Rx,Ry)
    return sess.run(h_conv3)



def raki(num_chan, N1, N2, acs_start_index_x, acs_end_index_x, acs_start_index_y, acs_end_index_y, kspace_undersampled_zero_filled, kspace_acs_zero_filled, kspace_coils_acs_out, kspace_acs_cropped, Rx, Ry, debug_path, name_image, name_weight,train_flag=True):

    acs_mask = np.zeros((num_chan, N1, N2), dtype=bool)
    acs_mask[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y] = True

    kspace = kspace_undersampled_zero_filled * np.invert(acs_mask).astype(int) + kspace_acs_zero_filled[:,:,:]
    #kspace = np.rot90(kspace,k=1,axes=(1,2))
    #mosaic(kspace, 4, 8, 1, [0,50], fig_title='kspace', fig_base_path=debug_path, fig_size=(15,15), num_rot=0)

    kspace = np.transpose(kspace,(2,1,0))
    print('kspace.shape: ', kspace.shape)

    #kspace_undersampled_zero_filled = np.rot90(kspace_undersampled_zero_filled,k=1,axes=(1,2))
    kspace_undersampled_zero_filled = np.transpose(kspace_undersampled_zero_filled,(2,1,0))

    #kspace_coils_acs_out = np.rot90(kspace_coils_acs_out,k=1,axes=(1,2))
    kspace_coils_acs_out = np.transpose(kspace_coils_acs_out,(2,1,0))

    #kspace_acs_cropped = np.rot90(kspace_acs_cropped,k=1,axes=(1,2))
    kspace_acs_cropped = np.transpose(kspace_acs_cropped,(2,1,0))


    # ## Parse Data
    #For convinience, everything are the same with Matlab version :)

    #### Network Parameters ####
    kernel_x_1 = 3
    kernel_y_1 = 3

    kernel_x_2 = 1
    kernel_y_2 = 1

    kernel_last_x = 1
    kernel_last_y = 1

    layer1_channels = 32
    layer2_channels = 32

    MaxIteration = 500
    LearningRate = 1e-2

    #### Input/Output Data ####
    #inputData = 'rawdata.mat'
    input_variable_name = 'kspace'
    resultName = 'RAKI_recon'
    recon_variable_name = 'kspace_recon'

    ######################################################################

    # Read data
    #kspace = sio.loadmat(inputData)
    #kspace = kspace[input_variable_name]
    #no_ACS_flag = 0;

    # Normalization
    normalize = 0.15/np.max(abs(kspace[:]))
    kspace = np.multiply(kspace,normalize)
    kspace_undersampled_zero_filled = np.multiply(kspace_undersampled_zero_filled,normalize)
    kspace_acs_cropped = np.multiply(kspace_acs_cropped,normalize)
    #return normalize
    kspace_coils_acs_out = np.multiply(kspace_coils_acs_out,normalize)


    # Get the shapes
    [m1,n1,no_ch] = np.shape(kspace)
    #no_inds = 1

    #kspace_all = kspace;


    #no_ACS_flag=0;

    ACS = kspace_acs_cropped
    ACS_out = kspace_coils_acs_out

    [ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS)
    ACS_re = np.zeros([ACS_dim_X,ACS_dim_Y,ACS_dim_Z*2])
    ACS_re[:,:,0:no_ch] = np.real(ACS)
    ACS_re[:,:,no_ch:no_ch*2] = np.imag(ACS)

    [ACS_out_dim_X, ACS_out_dim_Y, ACS_out_dim_Z] = np.shape(ACS_out)
    ACS_out_re = np.zeros([ACS_out_dim_X,ACS_out_dim_Y,ACS_out_dim_Z*2])
    ACS_out_re[:,:,0:ACS_out_dim_Z] = np.real(ACS_out)
    ACS_out_re[:,:,ACS_out_dim_Z:ACS_out_dim_Z*2] = np.imag(ACS_out)

    no_channels = ACS_dim_Z*2

    # What the !?
    b1_flag = 0;
    b2_flag = 0;
    b3_flag = 0;

    if (b1_flag == 1):
        b1_all = np.zeros([1,1, layer1_channels,no_channels]);
    else:
        b1 = []

    if (b2_flag == 1):
        b2_all = np.zeros([1,1, layer2_channels,no_channels])
    else:
        b2 = []

    if (b3_flag == 1):
        b3_all = np.zeros([1,1, layer3_channels, no_channels])
    else:
        b3 = []


    # ACS limits so that all kernels stay in the ACS region


    time_ALL_start = time.time()

    [ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS_re)
    ACS = np.reshape(ACS_re, [1,ACS_dim_X, ACS_dim_Y, ACS_dim_Z])
    ACS = np.float32(ACS)

    [ACS_out_dim_X, ACS_out_dim_Y, ACS_out_dim_Z] = np.shape(ACS_out_re)
    ACS_out = np.reshape(ACS_out_re, [1,ACS_out_dim_X, ACS_out_dim_Y, ACS_out_dim_Z])
    ACS_out = np.float32(ACS_out)

    target_x_start = np.int32((np.ceil(kernel_x_1/2)-1) + (np.ceil(kernel_x_2/2)-1) + (np.ceil(kernel_last_x/2)-1)) * Rx;
    target_x_end = ACS_out_dim_X  - np.int32((np.floor(kernel_x_1/2) + np.floor(kernel_x_2/2) + np.floor(kernel_last_x/2))) * Rx -1;


    target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * Ry;
    target_y_end = ACS_out_dim_Y  - np.int32((np.floor(kernel_y_1/2) + np.floor(kernel_y_2/2) + np.floor(kernel_last_y/2))) * Ry -1;

    target_dim_X = target_x_end - target_x_start + 1
    target_dim_Y = target_y_end - target_y_start + 1

    target_dim_Z = Rx * Ry - 1
    #target_dim_Z = Rx * Ry

    w1_all = np.zeros([kernel_x_1, kernel_y_1, no_channels, layer1_channels, no_channels],dtype=np.float32)
    w2_all = np.zeros([kernel_x_2, kernel_y_2, layer1_channels,layer2_channels,no_channels],dtype=np.float32)
    w3_all = np.zeros([kernel_last_x, kernel_last_y, layer2_channels,target_dim_Z, no_channels],dtype=np.float32)

    # ## Training
    if train_flag:
        print('go!')
        time_Learn_start = time.time()

        errorSum = 0;
        config = tf.ConfigProto()
        #config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True


        for ind_c in range(ACS_out_dim_Z):

            print('learning Channel #', ind_c+1)

            target = np.zeros([1,target_dim_X,target_dim_Y,target_dim_Z])

            for ind_acc_x in range(Rx):

                target_x_start = np.int32((np.ceil(kernel_x_1/2)-1) + (np.ceil(kernel_x_2/2)-1) + (np.ceil(kernel_last_x/2)-1)) * Rx + ind_acc_x;
                target_x_end = ACS_out_dim_X  - np.int32((np.floor(kernel_x_1/2) + np.floor(kernel_x_2/2) + np.floor(kernel_last_x/2))) * Rx + ind_acc_x - 1;
                #print ('target_x_start, target_x_end: ', target_x_start, target_x_end)
                for ind_acc_y in range(Ry):
                    target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * Ry + ind_acc_y
                    target_y_end = ACS_out_dim_Y  - np.int32((np.floor(kernel_y_1/2) + np.floor(kernel_y_2/2) + np.floor(kernel_last_y/2))) * Ry + ind_acc_y - 1;
                    if ind_acc_x == ind_acc_y == 0:
                        continue

                    target[0,:,:,ind_acc_y+ind_acc_x*Ry-1] = ACS_out[0,target_x_start:target_x_end + 1, target_y_start:target_y_end +1,ind_c];
                    #target[0,:,:,ind_acc_y+ind_acc_x*Ry] = ACS_out[0,target_x_start:target_x_end + 1, target_y_start:target_y_end +1,ind_c];

            sess = tf.Session(config=config)

            [w1,w2,w3]=learning(ACS,target,Rx, Ry, sess, ACS_dim_X, ACS_dim_Y, ACS_dim_Z, target_dim_X,target_dim_Y,target_dim_Z, target, kernel_x_1, kernel_x_2, kernel_y_1, kernel_y_2, layer1_channels, layer2_channels, kernel_last_x, kernel_last_y, LearningRate, MaxIteration)
            w1_all[:,:,:,:,ind_c] = w1
            w2_all[:,:,:,:,ind_c] = w2
            w3_all[:,:,:,:,ind_c] = w3


            sess.close()
            tf.reset_default_graph()

        time_Learn_end = time.time();
        print('lerning step costs:',(time_Learn_end - time_Learn_start),'s')
        sio.savemat(name_weight, {'w1': w1_all,'w2': w2_all,'w3': w3_all})

    # ## Reconstruction

    w_all = sio.loadmat(name_weight)
    w1_all = w_all['w1']
    w2_all = w_all['w2']
    w3_all = w_all['w3']

    #kspace_recon_all = np.copy(kspace_all)
    #kspace_recon_all_nocenter = np.copy(kspace_all)

    #kspace = np.copy(kspace_all)

    # Find oversampled lines and set them to zero
    #over_samp = np.setdiff1d(picks,np.int32([range(0, n1,acc_rate)]))

    kspace_und = kspace_undersampled_zero_filled
    [dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z] = np.shape(kspace_und)

    # Split real and imaginary parts
    kspace_und_re = np.zeros([dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
    kspace_und_re[:,:,0:dim_kspaceUnd_Z] = np.real(kspace_und)
    kspace_und_re[:,:,dim_kspaceUnd_Z:dim_kspaceUnd_Z*2] = np.imag(kspace_und)
    kspace_und_re = np.float32(kspace_und_re)
    kspace_und_re = np.reshape(kspace_und_re,[1,dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
    kspace_recon = kspace_und_re.copy()
    kspace_und_re_sq = kspace_und_re[:,::Rx,::Ry,:]

    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 1/3 ;

    if (b1_flag == 1):
        b1 = b1_all[:,:,:,ind_c];
    if (b2_flag == 1):
        b2 = b2_all[:,:,:,ind_c];
    if (b3_flag == 1):
        b3 = b3_all[:,:,:,ind_c];


    for ind_c in range(0,no_channels):
        print('Reconstruting Channel #',ind_c+1)

        sess = tf.Session(config=config)
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)

        # grab w and b
        w1 = np.float32(w1_all[:,:,:,:,ind_c])
        w2 = np.float32(w2_all[:,:,:,:,ind_c])
        w3 = np.float32(w3_all[:,:,:,:,ind_c])


        res = cnn_3layer(kspace_und_re_sq,w1,b1,w2,b2,w3,b3,1,1,sess)

        for ind_acc_x in range(Rx):

            target_x_start = np.int32((np.ceil(kernel_x_1/2)-1) + (np.ceil(kernel_x_2/2)-1) + (np.ceil(kernel_last_x/2)-1)) * Rx + ind_acc_x;
            target_x_end = dim_kspaceUnd_X  - np.int32((np.floor(kernel_x_1/2) + np.floor(kernel_x_2/2) + np.floor(kernel_last_x/2))) * Rx + ind_acc_x - 1;

            for ind_acc_y in range(Ry):
                target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * Ry + ind_acc_y
                target_y_end = dim_kspaceUnd_Y  - np.int32((np.floor(kernel_y_1/2) + np.floor(kernel_y_2/2) + np.floor(kernel_last_y/2))) * Ry + ind_acc_y - 1;

                if ind_acc_x == ind_acc_y == 0:
                    continue
                kspace_recon[0,target_x_start:target_x_end + 1:Rx, target_y_start:target_y_end +1:Ry,ind_c] = res[0,:,:,ind_acc_y+ind_acc_x*Ry-1]
                #kspace_recon[0,target_x_start:target_x_end + 1:Rx, target_y_start:target_y_end +1:Ry,ind_c] = res[0,:,:,ind_acc_y+ind_acc_x*Ry]

        sess.close()
        tf.reset_default_graph()

    kspace_recon = np.squeeze(kspace_recon)

    kspace_recon_complex = (kspace_recon[:,:,0:np.int32(no_channels/2)] + np.multiply(kspace_recon[:,:,np.int32(no_channels/2):no_channels],1j))
    #kspace_recon_all_nocenter[:,:,:] = np.copy(kspace_recon_complex);
    #kspace_recon_complex[:,center_start:center_end,:] = kspace_NEVER_TOUCH[:,center_start:center_end,:]

    kspace_recon_complex[acs_start_index_y:acs_end_index_y,acs_start_index_x:acs_end_index_x,:] = kspace_coils_acs_out

    #kspace_recon_all[:,:,:] = kspace_recon_complex;

    #for sli in range(0,no_ch):
    #    kspace_recon_all[:,:,sli] = np.fft.ifft2(kspace_recon_all[:,:,sli])

    #rssq = (np.sum(np.abs(kspace_recon_all)**2,2)**(0.5))
    sio.savemat(name_image,{recon_variable_name:kspace_recon_complex, 'normalize':normalize})

    time_ALL_end = time.time()
    print('All process costs ',(time_ALL_end-time_ALL_start),'s')
    #print('Error Average in Training is ',errorSum/no_channels)

def show_raki_results(image_original, sens_resize_i, debug_path, i, name_image):
    # ## Results
    # Read the kspace
    dt = sio.loadmat(name_image)

    kspace_recon = dt['kspace_recon']
    normalize = dt['normalize']
    kspace_recon = np.transpose(kspace_recon,(2,1,0))
    #kspace_recon = np.rot90(kspace_recon,k=-1,axes=(1,2))
    kspace_recon = kspace_recon/normalize

    #mosaic(kspace_recon, 4, 8, 1, [0,50], fig_title='kspace_recon', fig_base_path=debug_path, fig_size=(15,15), num_rot=0)

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

    #mosaic(raki_rssq_image_sub, 1, 1, 1, [0,150], fig_title=f'raki_rssq_image_sub_{i}', fig_base_path=debug_path, fig_size=(10,10), num_rot=0)
    #mosaic(image_original, 1, 1, 1, [0,150], fig_title=f'image_original_{i}', fig_base_path=debug_path, fig_size=(10,10), num_rot=0)

    #imsave('img_ori.jpg', image_original)
    #imsave('img_recon_raki.jpg', raki_rssq_image_sub)

    rmse_raki_sub = np.sqrt(np.sum(np.square(np.abs(image_original - raki_rssq_image_sub)),axis=(0,1))) / np.sqrt(np.sum(np.square(np.abs(image_original)),axis=(0,1)))

    rmse_raki_sub_str = str(rmse_raki_sub)
    print('rmse_raki: ' + str(rmse_raki_sub))
    return rmse_raki_sub

    comparison_epti = np.concatenate((image_original, raki_rssq_image_sub), axis=-1)
    comparison_epti = comparison_epti
    im_recon_error_epti = np.abs(image_original - raki_rssq_image_sub)
    im_recon_error_epti_scale = im_recon_error_epti * 5
    im_gr = np.concatenate((comparison_epti, im_recon_error_epti_scale), axis=-1)
    mosaic(im_gr, 1, 1, 1, [0,150], fig_title=f'image_error_{i}_{rmse_raki_sub_str}', fig_base_path=debug_path, fig_size=(30,10), num_rot=0)
    #im_gr = np.concatenate((comparison_epti, im_recon_error_epti_scale), axis=-1)
    #imsave(osp.join(f'comparison_recon_raki.jpg'), comparison_epti)
    #imsave(osp.join(f'comparison_recon_raki_error.jpg'), im_recon_error_epti_scale)
    sio.savemat(osp.join(debug_path, f'recon_raki_error_{i}'),{'data':im_recon_error_epti})
    return rmse_raki_sub

