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
from utils import *
from skimage.io import imsave
from operator import mul
from functools import reduce
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.disable_v2_behavior()
#tf.enable_eager_execution()
#RAKI
#(The code in this section is directly copied from the code supplied with the original paper)
# Reformat input data to match the supplied RAKI code

def weight_variable_partial(shape,vari_name,mode='k1'):
    mask = np.zeros(shape)
    #print (shape)
    #raise
    if mode=='k1':
        for i in range(4):
            mask[i,i*4,:,:,:] = 1
        for j in range(3):
            mask[j+6,j*4+2,:,:,:] = 1
    elif mode=='k2':
        for i in range(4):
            mask[i,i*4,:,:,:] = 1
        for j in range(3):
            mask[j+7,j*4+2,:,:,:] = 1
    else:
        print ('mode error !!!')
        raise

    #print (mask[:,:,0,0,0])

    mask = tf.convert_to_tensor(mask, dtype=tf.bool)
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    w = tf.Variable(initial,name = vari_name)
    w0 = tf.Variable(tf.zeros(shape))
    #return tf.Variable(initial,name = vari_name)
    return tf.where(mask, w, tf.stop_gradient(w0))

def weight_variable(shape,vari_name):
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial,name = vari_name)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv3d_dilate(x, W, strides=[1,1,1], dilation_rate=[1,1,1]):
    return tf.nn.convolution(x, W,padding='VALID',strides=strides, dilation_rate = dilation_rate)


#### LEARNING FUNCTION ####
def learning(ACS,target_input,Rx, Ry, Rt,sess, ACS_dim_X, ACS_dim_Y, ACS_dim_T, ACS_dim_Z, target_dim_X,target_dim_Y,target_dim_T,target_dim_Z, target, kernel_x_1, kernel_x_2, kernel_x_3, kernel_x_4, kernel_y_1, kernel_y_2, kernel_y_3, kernel_y_4, kernel_t_1, kernel_t_2, kernel_t_3, kernel_t_4, layer1_channels, layer2_channels, layer3_channels, layer4_channels, kernel_last_x, kernel_last_y, kernel_last_t, LearningRate, MaxIteration, finetune,slice_n,mode='k1'):

    input_ACS = tf.placeholder(tf.float32, [1, ACS_dim_X,ACS_dim_Y,ACS_dim_T,ACS_dim_Z])
    input_Target = tf.placeholder(tf.float32, [1, target_dim_X,target_dim_Y,target_dim_T,target_dim_Z])

    Input = tf.reshape(input_ACS, [1, ACS_dim_X, ACS_dim_Y, ACS_dim_T, ACS_dim_Z])

    [target_dim0,target_dim1,target_dim2,target_dim3,target_dim4] = np.shape(target)

    W_conv1 = weight_variable_partial([kernel_x_1, kernel_y_1, kernel_t_1, ACS_dim_Z, layer1_channels],'W1',mode=mode)
    #h_conv1 = tf.nn.relu(conv3d_dilate(Input, W_conv1,Rx, Ry))
    h_conv1 = conv3d_dilate(Input, W_conv1)

    W_conv2 = weight_variable([kernel_x_2, kernel_y_2, kernel_t_2, layer1_channels, layer2_channels],'W2')
    h_conv2 = conv3d_dilate(h_conv1, W_conv2)
    #h_conv2 = tf.nn.relu(conv3d_dilate(h_conv1, W_conv2, Rx=3, Ry=3, Rt=3))

    W_conv3 = weight_variable([kernel_x_3, kernel_y_3, kernel_t_3, layer2_channels, layer3_channels],'W3')
    h_conv3 = tf.nn.relu(conv3d_dilate(h_conv2, W_conv3))

    W_conv4 = weight_variable([kernel_x_4, kernel_y_4, kernel_t_4, layer3_channels, layer4_channels],'W4')
    h_conv4 = conv3d_dilate(h_conv3, W_conv4)

    W_conv5 = weight_variable([kernel_last_x, kernel_last_y, kernel_last_t, layer4_channels, target_dim4],'W5')
    h_conv5 = conv3d_dilate(h_conv4, W_conv5)


    #error_norm = tf.norm(input_Target - h_conv3)
    error_norm = (tf.norm(input_Target - h_conv5, ord=2) + tf.norm(input_Target - h_conv5, ord=1))*0.5
    #error_norm = tf.norm(input_Target - h_conv5, ord=2)*0.7 + tf.norm(input_Target - h_conv5, ord=1)*0.3
    #error_norm = tf.norm(input_Target - h_conv3, ord=1)

    #error_norm_tmp = tf.sqrt(tf.reduce_sum(tf.pow(input_Target - h_conv3,2), axis=(0,1,2,4)))
    #error_norm = tf.reduce_mean(error_norm_tmp)
    #error_norm0 = error_norm_tmp[90]



    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(LearningRate, global_step=global_step,decay_steps=50,decay_rate=0.95)
    #lr = tf.train.cosine_decay(LearningRate, global_step=global_step,decay_steps=50,alpha=0.001)
    train_step = tf.train.AdamOptimizer(lr).minimize(error_norm)
    #train_step = tf.train.AdamOptimizer(LearningRate).minimize(error_norm, global_step=global_step)

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    error_min = 1

    saver = tf.train.Saver()
    #saver = tf.train.Saver({"W1": W_conv1, "W2": W_conv2})

    checkpoint_dir = f'checkpoint/{slice_n}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    if finetune:
        restore_dir = f'checkpoint/{slice_n-1}'
        ckpt = tf.train.get_checkpoint_state(restore_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print (f'load pre-train model {restore_dir} / {ckpt_name} !!!!!!!!!!!!!!!!!!')
            #W_conv3.initializer.run()
            saver.restore(sess, os.path.join(restore_dir, ckpt_name))

    error_now=sess.run(error_norm,feed_dict={input_ACS: ACS, input_Target: target})
    print('The init an error',error_now)

    for i in range(MaxIteration+1):
        sess.run(train_step, feed_dict={input_ACS: ACS, input_Target: target, global_step:i})

        if i % 100 == 0:
            error_now=sess.run(error_norm,feed_dict={input_ACS: ACS, input_Target: target})

            #error_now0=sess.run(error_norm0,feed_dict={input_ACS: ACS, input_Target: target})
            #if error_now < error_min:
            #    w_list = [sess.run(W_conv1),sess.run(W_conv2),sess.run(W_conv3)]

            print('The',i,'th iteration gives an error',error_now)

    #model_path = os.path.join(checkpoint_dir, f'model_{slice_n}')
    #saver.save(sess, model_path)

    #error = sess.run(error_norm,feed_dict={input_ACS: ACS, input_Target: target})
    #return w_list
    return [sess.run(W_conv1),sess.run(W_conv2),sess.run(W_conv3),sess.run(W_conv4),sess.run(W_conv5)]

def cnn_3layer(input_kspace,w1,b1,w2,b2,w3,b3,w4,w5,Rx,Ry,Rt,sess):

    input_phdr = tf.placeholder(tf.float32, input_kspace.shape)
    w1_phdr = tf.placeholder(tf.float32, w1.shape)
    w2_phdr = tf.placeholder(tf.float32, w2.shape)
    w3_phdr = tf.placeholder(tf.float32, w3.shape)
    w4_phdr = tf.placeholder(tf.float32, w4.shape)
    w5_phdr = tf.placeholder(tf.float32, w5.shape)


    #h_conv1 = tf.nn.relu(conv3d_dilate(input_kspace, w1,Rx,Ry))
    h_conv1 = conv3d_dilate(input_phdr, w1_phdr, strides=[Rx,Ry,Rt])
    h_conv2 = conv3d_dilate(h_conv1, w2_phdr)
    #h_conv2 = tf.nn.relu(conv3d_dilate(h_conv1, w2_phdr))
    h_conv3 = tf.nn.relu(conv3d_dilate(h_conv2, w3_phdr))
    h_conv4 = conv3d_dilate(h_conv3, w4_phdr)
    h_conv5 = conv3d_dilate(h_conv4, w5_phdr)

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    return sess.run(h_conv5, feed_dict={input_phdr: input_kspace, w1_phdr: w1, w2_phdr: w2, w3_phdr: w3, w4_phdr: w4, w5_phdr: w5})

'''
def cnn_3layer(input_kspace,w1,b1,w2,b2,w3,b3,w4,w5,Rx,Ry,Rt,sess):

    h_conv1 = conv3d_dilate(input_kspace, w1, strides=[Rx,Ry,Rt])
    h_conv2 = conv3d_dilate(h_conv1, w2)
    h_conv3 = tf.nn.relu(conv3d_dilate(h_conv2, w3))
    h_conv4 = conv3d_dilate(h_conv3, w4)
    h_conv5 = conv3d_dilate(h_conv4, w5)
    return sess.run(h_conv5)
'''

def raki(num_chan, N1, N2, Nt, acs_start_index_x, acs_end_index_x, acs_start_index_y, acs_end_index_y, acs_start_index_t, acs_end_index_t, kspace_undersampled_zero_filled, kspace_acs_zero_filled, kspace_coils_acs_out, kspace_acs_cropped, Rx, Ry, Rt, debug_path, name_image, name_weight1, name_weight2, train_flag_1=True, train_flag_2=True, finetune=False, slice_n=0):

    acs_mask = np.zeros((num_chan, N1, N2, Nt), dtype=bool)
    acs_mask[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y, acs_start_index_t:acs_end_index_t] = True

    kspace = kspace_undersampled_zero_filled * np.invert(acs_mask).astype(int) + kspace_acs_zero_filled[:,:,:,:]

    #kspace = np.rot90(kspace,k=1,axes=(1,2))
    kspace = np.transpose(kspace,(2,1,3,0))

    #kspace_undersampled_zero_filled = np.rot90(kspace_undersampled_zero_filled,k=1,axes=(1,2))
    kspace_undersampled_zero_filled = np.transpose(kspace_undersampled_zero_filled,(2,1,3,0))

    #kspace_coils_acs_out = np.rot90(kspace_coils_acs_out,k=1,axes=(1,2))
    kspace_coils_acs_out = np.transpose(kspace_coils_acs_out,(2,1,3,0))

    #kspace_acs_cropped = np.rot90(kspace_acs_cropped,k=1,axes=(1,2))
    kspace_acs_cropped = np.transpose(kspace_acs_cropped,(2,1,3,0))

    #print('kspace.shape: ', kspace.shape)

    #dt = {'kspace' : kspace}
    #sio.savemat('rawdata.mat', dt)

    #print (kspace_coils_acs_out.shape)
    #raise
    # ## Parse Data
    #For convinience, everything are the same with Matlab version :)

    MaxIteration = 500
    LearningRate = 1e-2

    #### Input/Output Data ####
    #inputData = 'rawdata.mat'
    #input_variable_name = 'kspace'
    resultName = 'RAKI_recon'
    recon_variable_name = 'kspace_recon'

    ######################################################################

    # Read data
    #kspace = sio.loadmat(inputData)
    #kspace = kspace[input_variable_name]
    no_ACS_flag = 0;

    #print (np.max(abs(kspace[:])))
    #print (np.max(abs(kspace_coils_acs_out[:])))

    # Normalization
    normalize = 0.015/max(np.max(abs(kspace[:])), np.max(abs(kspace_coils_acs_out[:])))
    #normalize = 0.15/np.max(abs(kspace[:]))
    kspace = np.multiply(kspace,normalize)
    kspace_undersampled_zero_filled = np.multiply(kspace_undersampled_zero_filled,normalize)
    kspace_acs_cropped = np.multiply(kspace_acs_cropped,normalize)

    #normalize_out = 0.015/np.max(abs(kspace_coils_acs_out[:]))
    #kspace_coils_acs_out = np.multiply(kspace_coils_acs_out,normalize_out)
    kspace_coils_acs_out = np.multiply(kspace_coils_acs_out,normalize)

    #print (np.max(abs(kspace[:])))
    #print (np.max(abs(kspace_coils_acs_out[:])))
    #raise

    #return normalize

    # Get the shapes
    [m1,n1,no_t,no_ch] = np.shape(kspace)
    no_inds = 1

    kspace_all = np.copy(kspace);


    no_ACS_flag=0;
    print('ACS signal found in the input data')
    ACS = kspace_acs_cropped

    #ACS_out =ACS[:,:,:,0]
    #ACS_out = np.expand_dims(ACS_out, axis=-1)
    ACS_out = kspace_coils_acs_out
    #print (ACS_out.shape)
    #print (kspace_coils_acs_out.shape)


    [ACS_dim_X, ACS_dim_Y, ACS_dim_T, ACS_dim_Z] = np.shape(ACS)
    ACS_re = np.zeros([ACS_dim_X,ACS_dim_Y,ACS_dim_T,ACS_dim_Z*2])
    ACS_re[:,:,:,0:no_ch] = np.real(ACS)
    ACS_re[:,:,:,no_ch:no_ch*2] = np.imag(ACS)

    [ACS_out_dim_X, ACS_out_dim_Y, ACS_out_dim_t, ACS_out_dim_Z] = np.shape(ACS_out)
    ACS_out_re = np.zeros([ACS_out_dim_X,ACS_out_dim_Y,ACS_out_dim_t,ACS_out_dim_Z*2])
    ACS_out_re[:,:,:,0:ACS_out_dim_Z] = np.real(ACS_out)
    ACS_out_re[:,:,:,ACS_out_dim_Z:ACS_out_dim_Z*2] = np.imag(ACS_out)



    # Get the acceleration rate
    no_channels = ACS_out_dim_Z*2

    #return normalize, name_image

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

    ####### kernel 1 ########

    #### Network Parameters ####
    kernel_x_1 = 9  #nx
    kernel_y_1 = 13  #npe
    kernel_t_1 = 9  #nt

    kernel_x_2 = 1
    kernel_y_2 = 1
    kernel_t_2 = 3

    kernel_x_3 = 1
    kernel_y_3 = 1
    kernel_t_3 = 1

    kernel_x_4 = 1
    kernel_y_4 = 1
    kernel_t_4 = 1

    kernel_last_x = 1
    kernel_last_y = 1
    kernel_last_t = 1

    layer1_channels = 32
    layer2_channels = 32
    layer3_channels = 64
    layer4_channels = 32

    #target_t_start = np.int32(np.ceil(kernel_t_1/2) + np.floor(kernel_t_2/2) + np.floor(kernel_last_t/2) -1);
    target_t_start = np.int32((np.ceil(kernel_t_1/2)-1) + (np.ceil(kernel_t_2/2)-1) + (np.ceil(kernel_t_3/2)-1) + (np.ceil(kernel_t_4/2)-1) + (np.ceil(kernel_last_t/2)-1));
    target_t_end = np.int32(ACS_out_dim_t - target_t_start -1);
    #print (target_t_start, target_t_end)

    time_ALL_start = time.time()

    [ACS_dim_X, ACS_dim_Y, ACS_dim_T, ACS_dim_Z] = np.shape(ACS_re)
    ACS = np.reshape(ACS_re, [1,ACS_dim_X, ACS_dim_Y, ACS_dim_T, ACS_dim_Z])
    ACS = np.float32(ACS)


    [ACS_out_dim_X, ACS_out_dim_Y, ACS_out_dim_t, ACS_out_dim_Z] = np.shape(ACS_out_re)
    ACS_out = np.reshape(ACS_out_re, [1,ACS_out_dim_X, ACS_out_dim_Y,ACS_out_dim_t, ACS_out_dim_Z])
    ACS_out = np.float32(ACS_out)

    Rx_offset = 2
    Ry_offset = 4
    Rx_fill = 6
    Ry_fill = 5

    target_x_start = np.int32(Rx_offset)
    target_x_end = ACS_out_dim_X  - np.int32(kernel_x_1-Rx_offset)

    target_y_start = np.int32(Ry_offset)
    target_y_end = ACS_out_dim_Y  - np.int32(kernel_y_1-Ry_offset)

    print (target_x_start, target_x_end)
    print (target_y_start, target_y_end)
    print (target_t_start, target_t_end)


    target_dim_X = target_x_end - target_x_start + 1
    target_dim_Y = target_y_end - target_y_start + 1
    target_dim_T = target_t_end - target_t_start + 1
    #target_dim_Z = (acc_rate - 1)*ACS_out_dim_Z
    #target_dim_Z = Rx_fill * Ry_fill * ACS_out_dim_Z
    target_dim_Z = Rx_fill * Ry_fill - 1
    #print (target_dim_X, target_dim_Y, target_dim_T, target_dim_Z)


    w1_all = np.zeros([kernel_x_1, kernel_y_1, kernel_t_1, ACS_dim_Z, layer1_channels, ACS_out_dim_Z],dtype=np.float32)
    w2_all = np.zeros([kernel_x_2, kernel_y_2, kernel_t_2, layer1_channels,layer2_channels,ACS_out_dim_Z],dtype=np.float32)
    w3_all = np.zeros([kernel_x_3, kernel_y_3, kernel_t_3, layer2_channels,layer3_channels,ACS_out_dim_Z],dtype=np.float32)
    w4_all = np.zeros([kernel_x_4, kernel_y_4, kernel_t_4, layer3_channels,layer4_channels,ACS_out_dim_Z],dtype=np.float32)
    w5_all = np.zeros([kernel_last_x, kernel_last_y, kernel_last_t, layer4_channels,target_dim_Z, ACS_out_dim_Z],dtype=np.float32)



    # ## Training
    if train_flag_1:

        print('go!')

        time_Learn_start = time.time()

        errorSum = 0;
        config = tf.ConfigProto()
        #config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True


        for ind_c in range(ACS_out_dim_Z):
            target = np.zeros([1,target_dim_X,target_dim_Y,target_dim_T,target_dim_Z])
            print('learning channel #',ind_c+1)
            idx = 0
            for ind_acc_x in range(Rx_fill):

                target_x_start = np.int32(Rx_offset) + ind_acc_x;
                target_x_end = ACS_out_dim_X  - np.int32(kernel_x_1-Rx_offset) + ind_acc_x
                #print ('target_x_start, target_x_end: ', target_x_start, target_x_end)

                for ind_acc_y in range(Ry_fill):
                    target_y_start = np.int32(Ry_offset) + ind_acc_y
                    target_y_end = ACS_out_dim_Y  - np.int32(kernel_y_1-Ry_offset) + ind_acc_y
                    #print ('target_y_start, target_y_end: ', target_y_start, target_y_end)
                    #print (ind_acc_y+ind_acc_x*Ry+ind_c*Ry*Rx)
                    if (ind_acc_x == 0 and ind_acc_y == 4) or (ind_acc_x == 5 and ind_acc_y == 2):
                        continue


                    target[0,:,:,:,idx] = ACS_out[0,target_x_start:target_x_end + 1, target_y_start:target_y_end +1,target_t_start:target_t_end + 1,ind_c];
                    idx += 1

            sess = tf.Session(config=config)

            [w1,w2,w3,w4,w5]=learning(ACS,target,Rx, Ry, Rt, sess, ACS_dim_X, ACS_dim_Y, ACS_dim_T, ACS_dim_Z, target_dim_X,target_dim_Y,target_dim_T,target_dim_Z, target, kernel_x_1, kernel_x_2, kernel_x_3, kernel_x_4, kernel_y_1, kernel_y_2, kernel_y_3, kernel_y_4, kernel_t_1, kernel_t_2, kernel_t_3, kernel_t_4, layer1_channels, layer2_channels, layer3_channels, layer4_channels, kernel_last_x, kernel_last_y, kernel_last_t, LearningRate, MaxIteration, finetune,slice_n)
            w1_all[:,:,:,:,:,ind_c] = w1
            w2_all[:,:,:,:,:,ind_c] = w2
            w3_all[:,:,:,:,:,ind_c] = w3
            w4_all[:,:,:,:,:,ind_c] = w4
            w5_all[:,:,:,:,:,ind_c] = w5

            sess.close()
            tf.reset_default_graph()

        time_Learn_end = time.time();
        print('lerning step costs:',(time_Learn_end - time_Learn_start),'s')
        sio.savemat(name_weight1, {'w1': w1_all,'w2': w2_all,'w3': w3_all,'w4': w4_all,'w5': w5_all})



    #kspace_recon_all = np.copy(kspace_all)
    #kspace_recon_all_nocenter = np.copy(kspace_all)

    #kspace = np.copy(kspace_all)

    # Find oversampled lines and set them to zero
    kspace_und = kspace_undersampled_zero_filled

    [dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_T,dim_kspaceUnd_Z] = np.shape(kspace_und)
    kspace_und_re = np.zeros([dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_T,dim_kspaceUnd_Z*2])
    kspace_und_re[:,:,:,0:dim_kspaceUnd_Z] = np.real(kspace_und)
    kspace_und_re[:,:,:,dim_kspaceUnd_Z:dim_kspaceUnd_Z*2] = np.imag(kspace_und)
    kspace_und_re = np.float32(kspace_und_re)
    kspace_und_re = np.reshape(kspace_und_re,[1,dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_T,dim_kspaceUnd_Z*2])

    #kspace_und_re_sq = kspace_und_re[:,::Rx,::Ry,:,:]
    '''
    kspace_und_out = kspace_und[:,:,:,0]
    kspace_und_out = np.expand_dims(kspace_und_out, axis=-1)
    [dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_T,dim_kspaceUnd_Z] = np.shape(kspace_und_out)
    kspace_und_out_re = np.zeros([dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_T,dim_kspaceUnd_Z*2])
    #kspace_und_out_re[:,:,:,0:dim_kspaceUnd_Z] = np.real(kspace_und_out)
    #kspace_und_out_re[:,:,:,dim_kspaceUnd_Z:dim_kspaceUnd_Z*2] = np.imag(kspace_und_out)
    #kspace_und_out_re = np.float32(kspace_und_out_re)
    kspace_und_out_re = np.reshape(kspace_und_out_re,[1,dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_T,dim_kspaceUnd_Z*2])
    '''
    #kspace_recon = kspace_und_out_re
    kspace_recon = kspace_und_re.copy()

    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 1/3 ;

    print (np.max(abs(kspace_und_re[:])))
    print (np.max(abs(ACS[:])))

    ###### Reconstruction k1 #########

    w_all = sio.loadmat(name_weight1)
    w1_all = w_all['w1']
    w2_all = w_all['w2']
    w3_all = w_all['w3']
    w4_all = w_all['w4']
    w5_all = w_all['w5']

    if (b1_flag == 1):
        b1 = b1_all[:,:,:,ind_c];
    if (b2_flag == 1):
        b2 = b2_all[:,:,:,ind_c];
    if (b3_flag == 1):
        b3 = b3_all[:,:,:,ind_c];

    for i in range(Rx): # Ry_cycle=12
        x_i = i
        y_i = i*4



        target_t_start = np.int32((np.ceil(kernel_t_1/2)-1) + (np.ceil(kernel_t_2/2)-1) + (np.ceil(kernel_t_3/2)-1) + (np.ceil(kernel_t_4/2)-1) + (np.ceil(kernel_last_t/2)-1));
        target_t_end = np.int32(dim_kspaceUnd_T - target_t_start -1);

        for ind_c in range(0,no_channels):

            print('Reconstruting Channel #', ind_c+1)
            sess = tf.Session(config=config)
            # grab w and b
            w1 = np.float32(w1_all[:,:,:,:,:,ind_c])
            w2 = np.float32(w2_all[:,:,:,:,:,ind_c])
            w3 = np.float32(w3_all[:,:,:,:,:,ind_c])
            w4 = np.float32(w4_all[:,:,:,:,:,ind_c])
            w5 = np.float32(w5_all[:,:,:,:,:,ind_c])

            res_i = cnn_3layer(kspace_und_re[:,x_i:,y_i:,:,:],w1,b1,w2,b2,w3,b3,w4,w5,Rx,Ry,1,sess)
            idx = 0

            for ind_acc_x in range(Rx_fill):

                target_x_start = np.int32(Rx_offset) + ind_acc_x;
                target_x_end = dim_kspaceUnd_X  - np.int32(kernel_x_1-Rx_offset) + ind_acc_x

                for ind_acc_y in range(Ry_fill):
                    target_y_start = np.int32(Ry_offset) + ind_acc_y
                    target_y_end = dim_kspaceUnd_Y  - np.int32(kernel_y_1-Ry_offset) + ind_acc_y

                    if (ind_acc_x == 0 and ind_acc_y == 4) or (ind_acc_x == 5 and ind_acc_y == 2):
                        continue

                    kspace_recon[0,target_x_start+x_i:target_x_end + 1:Rx, target_y_start+y_i:target_y_end +1:Ry,target_t_start:target_t_end + 1,ind_c] = res_i[0,:,:,:,idx]
                    idx += 1

            sess.close()
            tf.reset_default_graph()


    ####### kernel 2 ########

    #### Network Parameters ####
    kernel_x_1 = 10  #nx
    kernel_y_1 = 13  #npe
    kernel_t_1 = 9  #nt

    kernel_x_2 = 1
    kernel_y_2 = 1
    kernel_t_2 = 3

    kernel_x_3 = 1
    kernel_y_3 = 1
    kernel_t_3 = 1

    kernel_x_4 = 1
    kernel_y_4 = 1
    kernel_t_4 = 1

    kernel_last_x = 1
    kernel_last_y = 1
    kernel_last_t = 1

    layer1_channels = 32
    layer2_channels = 32
    layer3_channels = 64
    layer4_channels = 32

    #target_t_start = np.int32(np.ceil(kernel_t_1/2) + np.floor(kernel_t_2/2) + np.floor(kernel_last_t/2) -1);
    target_t_start = np.int32((np.ceil(kernel_t_1/2)-1) + (np.ceil(kernel_t_2/2)-1) + (np.ceil(kernel_t_3/2)-1) + (np.ceil(kernel_t_4/2)-1) + (np.ceil(kernel_last_t/2)-1));
    target_t_end = np.int32(ACS_out_dim_t - target_t_start -1);
    #print (target_t_start, target_t_end)

    Rx_offset = 2
    Ry_offset = 4
    Rx_fill = 7
    Ry_fill = 5

    target_x_start = np.int32(Rx_offset)
    target_x_end = ACS_out_dim_X  - np.int32(kernel_x_1-Rx_offset)

    target_y_start = np.int32(Ry_offset)
    target_y_end = ACS_out_dim_Y  - np.int32(kernel_y_1-Ry_offset)

    print (target_x_start, target_x_end)
    print (target_y_start, target_y_end)
    print (target_t_start, target_t_end)


    target_dim_X = target_x_end - target_x_start + 1
    target_dim_Y = target_y_end - target_y_start + 1
    target_dim_T = target_t_end - target_t_start + 1
    #target_dim_Z = (acc_rate - 1)*ACS_out_dim_Z
    #target_dim_Z = Rx_fill * Ry_fill * ACS_out_dim_Z
    target_dim_Z = Rx_fill * Ry_fill - 1
    #print (target_dim_X, target_dim_Y, target_dim_T, target_dim_Z)


    w1_all = np.zeros([kernel_x_1, kernel_y_1, kernel_t_1, ACS_dim_Z, layer1_channels, ACS_out_dim_Z],dtype=np.float32)
    w2_all = np.zeros([kernel_x_2, kernel_y_2, kernel_t_2, layer1_channels,layer2_channels,ACS_out_dim_Z],dtype=np.float32)
    w3_all = np.zeros([kernel_x_3, kernel_y_3, kernel_t_3, layer2_channels,layer3_channels,ACS_out_dim_Z],dtype=np.float32)
    w4_all = np.zeros([kernel_x_4, kernel_y_4, kernel_t_4, layer3_channels,layer4_channels,ACS_out_dim_Z],dtype=np.float32)
    w5_all = np.zeros([kernel_last_x, kernel_last_y, kernel_last_t, layer4_channels,target_dim_Z, ACS_out_dim_Z],dtype=np.float32)



    # ## Training
    if train_flag_2:

        print('go!')

        time_Learn_start = time.time()

        errorSum = 0;
        config = tf.ConfigProto()
        #config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True




        for ind_c in range(ACS_out_dim_Z):
            target = np.zeros([1,target_dim_X,target_dim_Y,target_dim_T,target_dim_Z])
            print('learning channel #',ind_c+1)
            idx = 0

            for ind_acc_x in range(Rx_fill):

                target_x_start = np.int32(Rx_offset) + ind_acc_x;
                target_x_end = ACS_out_dim_X  - np.int32(kernel_x_1-Rx_offset) + ind_acc_x
                #print ('target_x_start, target_x_end: ', target_x_start, target_x_end)
                for ind_acc_y in range(Ry_fill):
                    target_y_start = np.int32(Ry_offset) + ind_acc_y
                    target_y_end = ACS_out_dim_Y  - np.int32(kernel_y_1-Ry_offset) + ind_acc_y
                    #print ('target_y_start, target_y_end: ', target_y_start, target_y_end)
                    #print (ind_acc_y+ind_acc_x*Ry+ind_c*Ry*Rx)

                    if (ind_acc_x == 0 and ind_acc_y == 4) or (ind_acc_x == 6 and ind_acc_y == 2):
                        continue

                    target[0,:,:,:,idx] = ACS_out[0,target_x_start:target_x_end + 1, target_y_start:target_y_end +1,target_t_start:target_t_end + 1,ind_c];
                    idx += 1

            sess = tf.Session(config=config)

            [w1,w2,w3,w4,w5]=learning(ACS,target,Rx, Ry, Rt, sess, ACS_dim_X, ACS_dim_Y, ACS_dim_T, ACS_dim_Z, target_dim_X,target_dim_Y,target_dim_T,target_dim_Z, target, kernel_x_1, kernel_x_2, kernel_x_3, kernel_x_4, kernel_y_1, kernel_y_2, kernel_y_3, kernel_y_4, kernel_t_1, kernel_t_2, kernel_t_3, kernel_t_4, layer1_channels, layer2_channels, layer3_channels, layer4_channels, kernel_last_x, kernel_last_y, kernel_last_t, LearningRate, MaxIteration, finetune,slice_n,mode='k2')
            w1_all[:,:,:,:,:,ind_c] = w1
            w2_all[:,:,:,:,:,ind_c] = w2
            w3_all[:,:,:,:,:,ind_c] = w3
            w4_all[:,:,:,:,:,ind_c] = w4
            w5_all[:,:,:,:,:,ind_c] = w5

            sess.close()
            tf.reset_default_graph()

        time_Learn_end = time.time();
        print('lerning step costs:',(time_Learn_end - time_Learn_start),'s')
        sio.savemat(name_weight2, {'w1': w1_all,'w2': w2_all,'w3': w3_all,'w4': w4_all,'w5': w5_all})



    ###### Reconstruction k2 #########

    w_all = sio.loadmat(name_weight2)
    w1_all = w_all['w1']
    w2_all = w_all['w2']
    w3_all = w_all['w3']
    w4_all = w_all['w4']
    w5_all = w_all['w5']




    if (b1_flag == 1):
        b1 = b1_all[:,:,:,ind_c];
    if (b2_flag == 1):
        b2 = b2_all[:,:,:,ind_c];
    if (b3_flag == 1):
        b3 = b3_all[:,:,:,ind_c];

    for i in range(Rx): # Ry_cycle=12

        x_i = (i+6)%Rx
        y_i = i*4+2

        #print (kspace_und_re[:,x_i:x_i+10,y_i:y_i+13,0,0])
        #print ('*'*30)
        #print (w1.shape)
        #print (w1[:,:,0,0,0])
        #raise



        #target_t_start = np.int32(np.ceil(kernel_t_1/2) + np.floor(kernel_t_2/2) + np.floor(kernel_last_t/2) -1);
        target_t_start = np.int32((np.ceil(kernel_t_1/2)-1) + (np.ceil(kernel_t_2/2)-1) + (np.ceil(kernel_t_3/2)-1) + (np.ceil(kernel_t_4/2)-1) + (np.ceil(kernel_last_t/2)-1));
        target_t_end = np.int32(dim_kspaceUnd_T - target_t_start -1);

        for ind_c in range(0,no_channels):


            sess = tf.Session(config=config)
            # grab w and b
            w1 = np.float32(w1_all[:,:,:,:,:,ind_c])
            w2 = np.float32(w2_all[:,:,:,:,:,ind_c])
            w3 = np.float32(w3_all[:,:,:,:,:,ind_c])
            w4 = np.float32(w4_all[:,:,:,:,:,ind_c])
            w5 = np.float32(w5_all[:,:,:,:,:,ind_c])

            res_i = cnn_3layer(kspace_und_re[:,x_i:,y_i:,:,:],w1,b1,w2,b2,w3,b3,w4,w5,Rx,Ry,1,sess)
            idx = 0

            for ind_acc_x in range(Rx_fill):

                target_x_start = np.int32(Rx_offset) + ind_acc_x;
                target_x_end = dim_kspaceUnd_X  - np.int32(kernel_x_1-Rx_offset) + ind_acc_x

                for ind_acc_y in range(Ry_fill):
                    target_y_start = np.int32(Ry_offset) + ind_acc_y
                    target_y_end = dim_kspaceUnd_Y  - np.int32(kernel_y_1-Ry_offset) + ind_acc_y

                    if (ind_acc_x == 0 and ind_acc_y == 4) or (ind_acc_x == 6 and ind_acc_y == 2):
                        continue

                    kspace_recon[0,target_x_start+x_i:target_x_end + 1:Rx, target_y_start+y_i:target_y_end +1:Ry,target_t_start:target_t_end + 1,ind_c] = res_i[0,:,:,:,idx]
                    idx += 1

            sess.close()
            tf.reset_default_graph()

    kspace_recon = np.squeeze(kspace_recon)


    kspace_recon_complex = (kspace_recon[:,:,:,0:np.int32(no_channels/2)] + np.multiply(kspace_recon[:,:,:,np.int32(no_channels/2):no_channels],1j))
    #kspace_recon_all_nocenter[:,:,:,:] = np.copy(kspace_recon_complex);

    #put back acs

    #print (kspace_recon_complex.shape)
    #print (kspace_coils_acs_out.shape)
    #print (acs_start_index_x, acs_end_index_x, acs_start_index_y, acs_end_index_y)
    #print (kspace_recon_complex.shape)
    #print (kspace_coils_acs_out.shape)
    #raise

    #kspace_recon_complex[acs_start_index_y:acs_end_index_y,acs_start_index_x:acs_end_index_x,acs_start_index_t:acs_end_index_t,:] = kspace_coils_acs_out

    #kspace_recon_all[:,:,:] = kspace_recon_complex;

    #for sli in range(0,no_ch):
    #    kspace_recon_all[:,:,:,sli] = np.fft.ifft2(kspace_recon_all[:,:,:,sli])

    #rssq = (np.sum(np.abs(kspace_recon_all)**2,2)**(0.5))
    sio.savemat(name_image,{recon_variable_name:kspace_recon_complex, 'normalize':normalize})

    time_ALL_end = time.time()
    print('All process costs ',(time_ALL_end-time_ALL_start),'s')
    #print('Error Average in Training is ',errorSum/no_channels)

def show_raki_results(kdata_ref, sens_resize, debug_path, name_image):
    # ## Results
    # Read the kspace
    dt = sio.loadmat(name_image)

    kspace_recon = dt['kspace_recon']
    #print (np.max(abs(kspace_recon[:])), np.min(abs(kspace_recon[:])))

    normalize = dt['normalize']
    kspace_recon = np.transpose(kspace_recon,(3,1,2,0))
    #kspace_recon = np.rot90(kspace_recon,k=-1,axes=(1,2))
    kspace_recon = kspace_recon/normalize

    kdata_ref0 = np.transpose(kdata_ref,(0,1,3,2))
    sens_resize = np.transpose(sens_resize,(0,1,3,2))

    #print (np.max(abs(kdata_ref0[:])), np.min(abs(kdata_ref0[:])))
    #print (np.max(abs(kspace_recon[:])), np.min(abs(kspace_recon[:])))
    #raise

    #mosaic(kspace_recon, 4, 8, 1, [0,50], fig_title='kspace_recon', fig_base_path=debug_path, fig_size=(15,15), num_rot=0)

    #kspace_recon = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kspace_recon, axes=3), axis=3), axes=3)

    # Get the RAKI coil images
    coil_images_raki_sub = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_recon, axes=(1,2)), axes=(1,2)), axes=(1,2))
    #coil_images_raki_sub = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace_recon, axes=(1,2,3)), axes=(1,2,3)), axes=(1,2,3))

    #coil_images_raki_sub = kspace_recon

    raki_rssq_image_sub0 = np.sum(np.conj(sens_resize) * coil_images_raki_sub, axis=0)/np.sum(np.square(np.abs(sens_resize)), axis=0)
    raki_rssq_image_sub0 = np.abs(raki_rssq_image_sub0)
    #raki_rssq_image_sub0 = np.sqrt(np.sum(np.square(np.abs(coil_images_raki_sub)),axis=0))
    #raki_rssq_image_sub0 = np.abs(coil_images_raki_sub)[0]

    image_original = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kdata_ref0, axes=(1,2)), axes=(1,2)), axes=(1,2))
    image_original0 = np.sqrt(np.sum(np.square(np.abs(image_original)),axis=0))
    #image_original0 = np.abs(image_original)[0]


    print (image_original0.shape)
    print (raki_rssq_image_sub0.shape)

    rmse_list = []


    image_original = image_original0[:,:,0]
    image_ref0 = image_original.copy()
    mask = image_ref0 > 0.6 * np.mean(image_ref0)
    #mask = 1.0 * mask
    t = image_original0.shape[2]
    mask = np.expand_dims(mask, axis=-1)
    mask = np.tile(mask,(1,1,t))
    mask = 1.0 * mask
    #print (mask.shape)

    image_ori = image_original0 * mask
    recon_ori = raki_rssq_image_sub0 * mask

    rmse_raki = np.sqrt(np.sum(np.square(np.abs(image_ori - recon_ori)),axis=(0,1))) / np.sqrt(np.sum(np.square(np.abs(image_ori)),axis=(0,1)))
    print('rmse_raki: ' + str(rmse_raki))
    
    #raise
    #for i in range(50,150):
    for i in range(len(image_original0[0,0,:])):
        image_original = image_original0[:,:,i]
        raki_rssq_image_sub = raki_rssq_image_sub0[:,:,i]

        image_original = image_original * mask[:,:,i]
        raki_rssq_image_sub = raki_rssq_image_sub * mask[:,:,i]

        #print (image_original.max())
        #print (raki_rssq_image_sub.max())
        #continue

        mosaic(raki_rssq_image_sub, 1, 1, 1, [0,0.02], fig_title=f'raki_rssq_image_sub_{i}', fig_base_path=debug_path, fig_size=(10,10), num_rot=0)
        mosaic(image_original, 1, 1, 1, [0,0.02], fig_title=f'image_original_{i}', fig_base_path=debug_path, fig_size=(10,10), num_rot=0)

        #imsave('img_ori.jpg', image_original)
        #imsave('img_recon_raki.jpg', raki_rssq_image_sub)

        rmse_raki_sub = np.sqrt(np.sum(np.square(np.abs(image_original - raki_rssq_image_sub)),axis=(0,1))) / np.sqrt(np.sum(np.square(np.abs(image_original)),axis=(0,1)))
        print('rmse_raki: ' + str(rmse_raki_sub))

        '''
        comparison_epti = np.concatenate((image_original, raki_rssq_image_sub), axis=-1)
        comparison_epti = comparison_epti
        im_recon_error_epti = np.abs(image_original - raki_rssq_image_sub)
        im_recon_error_epti_scale = im_recon_error_epti * 5
        im_gr = np.concatenate((comparison_epti, im_recon_error_epti_scale), axis=-1)
        mosaic(im_gr, 1, 1, 1, [0,150], fig_title=f'image_error_{i}', fig_base_path=debug_path, fig_size=(30,10), num_rot=0)
        #im_gr = np.concatenate((comparison_epti, im_recon_error_epti_scale), axis=-1)
        #imsave(osp.join(f'comparison_recon_raki.jpg'), comparison_epti)
        #imsave(osp.join(f'comparison_recon_raki_error.jpg'), im_recon_error_epti_scale)
        sio.savemat(osp.join(debug_path, f'recon_raki_error_{i}'),{'data':im_recon_error_epti})
        '''

        rmse_list.append(rmse_raki_sub)
    rmse_np = np.array(rmse_list)
    data = pd.DataFrame(rmse_np)
    writer = pd.ExcelWriter(f'{debug_path}/rmse.xlsx')
    data.to_excel(writer, 'rmse', float_format='%.6f')
    writer.save()
    writer.close()

