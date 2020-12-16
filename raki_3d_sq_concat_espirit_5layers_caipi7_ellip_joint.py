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
import cv2
from utils import *
from skimage.io import imsave
from skimage.metrics import structural_similarity as ssim
from operator import mul
from functools import reduce
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.disable_v2_behavior()
#RAKI
#(The code in this section is directly copied from the code supplied with the original paper)
# Reformat input data to match the supplied RAKI code


def weight_variable_partial(shape, vari_name):
    mask = np.zeros(shape)
    #print (mask.shape)


    mask[0, 0, :,0:32,:] = 1
    mask[3, 1, :,0:32,:] = 1
    mask[6, 2, :,0:32,:] = 1
    mask[2, 3, :,0:32,:] = 1
    mask[5, 4, :,0:32,:] = 1
    mask[1, 5, :,0:32,:] = 1
    mask[4, 6, :,0:32,:] = 1
    mask[0, 7, :,0:32,:] = 1
    mask[7, 0, :,0:32,:] = 1
    mask[7, 7, :,0:32,:] = 1

    mask[1, 1, :,32:64,:] = 1
    mask[4, 2, :,32:64,:] = 1
    mask[7, 3, :,32:64,:] = 1
    mask[3, 4, :,32:64,:] = 1
    mask[6, 5, :,32:64,:] = 1
    mask[2, 6, :,32:64,:] = 1
    mask[5, 7, :,32:64,:] = 1  
    mask[5, 0, :,32:64,:] = 1  
    mask[0, 3, :,32:64,:] = 1  

    mask[2, 2, :,64:96,:] = 1
    mask[5, 3, :,64:96,:] = 1
    mask[4, 5, :,64:96,:] = 1
    mask[7, 6, :,64:96,:] = 1
    mask[3, 7, :,64:96,:] = 1
    mask[6, 1, :,64:96,:] = 1  
    mask[1, 4, :,64:96,:] = 1  
    mask[3, 0, :,64:96,:] = 1  
    mask[0, 6, :,64:96,:] = 1  



    mask[0, 0, :,32*3:32*4,:] = 1
    mask[3, 1, :,32*3:32*4,:] = 1
    mask[6, 2, :,32*3:32*4,:] = 1
    mask[2, 3, :,32*3:32*4,:] = 1
    mask[5, 4, :,32*3:32*4,:] = 1
    mask[1, 5, :,32*3:32*4,:] = 1
    mask[4, 6, :,32*3:32*4,:] = 1
    mask[0, 7, :,32*3:32*4,:] = 1
    mask[7, 0, :,32*3:32*4,:] = 1
    mask[7, 7, :,32*3:32*4,:] = 1

    mask[1, 1, :,32*4:32*5,:] = 1
    mask[4, 2, :,32*4:32*5,:] = 1
    mask[7, 3, :,32*4:32*5,:] = 1
    mask[3, 4, :,32*4:32*5,:] = 1
    mask[6, 5, :,32*4:32*5,:] = 1
    mask[2, 6, :,32*4:32*5,:] = 1
    mask[5, 7, :,32*4:32*5,:] = 1  
    mask[5, 0, :,32*4:32*5,:] = 1  
    mask[0, 3, :,32*4:32*5,:] = 1  

    mask[2, 2, :,32*5:32*6,:] = 1
    mask[5, 3, :,32*5:32*6,:] = 1
    mask[4, 5, :,32*5:32*6,:] = 1
    mask[7, 6, :,32*5:32*6,:] = 1
    mask[3, 7, :,32*5:32*6,:] = 1
    mask[6, 1, :,32*5:32*6,:] = 1  
    mask[1, 4, :,32*5:32*6,:] = 1  
    mask[3, 0, :,32*5:32*6,:] = 1  
    mask[0, 6, :,32*5:32*6,:] = 1  




    '''
    mask[0, 0, ...] = 1
    mask[0, 7, ...] = 1
    mask[7, 0, ...] = 1
    mask[7, 7, ...] = 1
    mask[3, 1, ...] = 1
    mask[6, 2, ...] = 1
    mask[2, 3, ...] = 1
    mask[5, 4, ...] = 1
    mask[1, 5, ...] = 1
    mask[4, 6, ...] = 1
    '''
    '''
    mask[0, 0, :,0:16,:] = 1
    mask[3, 1, :,0:16,:] = 1
    mask[6, 2, :,0:16,:] = 1
    mask[2, 3, :,0:16,:] = 1
    mask[5, 4, :,0:16,:] = 1
    mask[1, 5, :,0:16,:] = 1
    mask[4, 6, :,0:16,:] = 1
    mask[0, 7, :,0:16,:] = 1
    mask[7, 0, :,0:16,:] = 1
    mask[7, 7, :,0:16,:] = 1

    mask[0, 0, :,16:32,:] = 1
    mask[3, 1, :,16:32,:] = 1
    mask[6, 2, :,16:32,:] = 1
    mask[2, 3, :,16:32,:] = 1
    mask[5, 4, :,16:32,:] = 1
    mask[1, 5, :,16:32,:] = 1
    mask[4, 6, :,16:32,:] = 1
    mask[0, 7, :,16:32,:] = 1
    mask[7, 0, :,16:32,:] = 1
    mask[7, 7, :,16:32,:] = 1
    '''
    '''
    mask[0, 0, :,0:16,:] = 1
    mask[3, 1, :,0:16,:] = 1
    mask[6, 2, :,0:16,:] = 1
    mask[2, 3, :,0:16,:] = 1
    mask[5, 4, :,0:16,:] = 1
    mask[1, 5, :,0:16,:] = 1
    mask[4, 6, :,0:16,:] = 1
    mask[0, 7, :,0:16,:] = 1
    mask[7, 0, :,0:16,:] = 1
    mask[7, 7, :,0:16,:] = 1

    mask[1, 1, :,16:32,:] = 1
    mask[4, 2, :,16:32,:] = 1
    mask[7, 3, :,16:32,:] = 1
    mask[3, 4, :,16:32,:] = 1
    mask[6, 5, :,16:32,:] = 1
    mask[2, 6, :,16:32,:] = 1
    mask[5, 7, :,16:32,:] = 1  
    mask[5, 0, :,16:32,:] = 1  
    mask[0, 3, :,16:32,:] = 1  

    mask[2, 2, :,32:48,:] = 1
    mask[5, 3, :,32:48,:] = 1
    mask[4, 5, :,32:48,:] = 1
    mask[7, 6, :,32:48,:] = 1
    mask[3, 7, :,32:48,:] = 1
    mask[6, 1, :,32:48,:] = 1  
    mask[1, 4, :,32:48,:] = 1  
    mask[3, 0, :,32:48,:] = 1  
    mask[0, 6, :,32:48,:] = 1  


    mask[0, 0, :,48:64,:] = 1
    mask[3, 1, :,48:64,:] = 1
    mask[6, 2, :,48:64,:] = 1
    mask[2, 3, :,48:64,:] = 1
    mask[5, 4, :,48:64,:] = 1
    mask[1, 5, :,48:64,:] = 1
    mask[4, 6, :,48:64,:] = 1
    mask[0, 7, :,48:64,:] = 1
    mask[7, 0, :,48:64,:] = 1
    mask[7, 7, :,48:64,:] = 1

    mask[1, 1, :,64:80,:] = 1
    mask[4, 2, :,64:80,:] = 1
    mask[7, 3, :,64:80,:] = 1
    mask[3, 4, :,64:80,:] = 1
    mask[6, 5, :,64:80,:] = 1
    mask[2, 6, :,64:80,:] = 1
    mask[5, 7, :,64:80,:] = 1  
    mask[5, 0, :,64:80,:] = 1  
    mask[0, 3, :,64:80,:] = 1  

    mask[2, 2, :,80:96,:] = 1
    mask[5, 3, :,80:96,:] = 1
    mask[4, 5, :,80:96,:] = 1
    mask[7, 6, :,80:96,:] = 1
    mask[3, 7, :,80:96,:] = 1
    mask[6, 1, :,80:96,:] = 1  
    mask[1, 4, :,80:96,:] = 1  
    mask[3, 0, :,80:96,:] = 1  
    mask[0, 6, :,80:96,:] = 1  
    '''


    mask = tf.convert_to_tensor(mask, dtype=tf.bool)
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    w = tf.Variable(initial,name = vari_name)
    w0 = tf.Variable(tf.zeros(shape))
    #return tf.Variable(initial,name = vari_name)
    return tf.where(mask, w, tf.stop_gradient(w0))

def weight_variable(shape,vari_name):
    #initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    initial = tf.random_normal(shape, stddev=0.1,dtype=tf.float32)
    #initial = tf.random_uniform(shape, minval=-0.1,maxval=-0.1,dtype=tf.float32)
    return tf.Variable(initial,name = vari_name)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv3d_dilate(x, W, strides=[1,1,1], dilate_rate=[1,1,1]):
    return tf.nn.convolution(x, W,padding='VALID', strides=strides, dilation_rate = dilate_rate)


#### LEARNING FUNCTION ####
def learning(ACS,target_input,Rx, Ry,sess, ACS_dim_X, ACS_dim_Y, ACS_dim_T, ACS_dim_Z, target_dim_X,target_dim_Y,target_dim_T,target_dim_Z, target, kernel_x_1, kernel_x_2, kernel_x_3, kernel_x_4, kernel_y_1, kernel_y_2, kernel_y_3, kernel_y_4, kernel_t_1, kernel_t_2, kernel_t_3, kernel_t_4, layer1_channels, layer2_channels, layer3_channels, layer4_channels, kernel_last_x, kernel_last_y, kernel_last_t, LearningRate, MaxIteration, finetune, slice_n):

    input_ACS = tf.placeholder(tf.float32, [1, ACS_dim_X,ACS_dim_Y,ACS_dim_T,ACS_dim_Z])
    input_Target = tf.placeholder(tf.float32, [1, target_dim_X,target_dim_Y,target_dim_T,target_dim_Z])

    Input = tf.reshape(input_ACS, [1, ACS_dim_X, ACS_dim_Y, ACS_dim_T, ACS_dim_Z])

    [target_dim0,target_dim1,target_dim2,target_dim3,target_dim4] = np.shape(target)

    W_conv1 = weight_variable_partial([kernel_x_1, kernel_y_1, kernel_t_1, ACS_dim_Z, layer1_channels],'W1')
    #h_conv1 = tf.nn.relu(conv3d_dilate(Input, W_conv1, Rx, Ry))
    h_conv1 = conv3d_dilate(Input, W_conv1)

    W_conv2 = weight_variable([kernel_x_2, kernel_y_2, kernel_t_2, layer1_channels, layer2_channels],'W2')
    #h_conv2 = tf.nn.relu(conv3d_dilate(h_conv1, W_conv2, Rx, Ry))
    h_conv2 = conv3d_dilate(h_conv1, W_conv2)

    W_conv3 = weight_variable([kernel_x_3, kernel_y_3, kernel_t_3, layer2_channels, layer3_channels],'W3')
    #h_conv3 = tf.nn.relu(conv3d_dilate(h_conv2, W_conv3, Rx, Ry))
    h_conv3 = conv3d_dilate(h_conv2, W_conv3)

    W_conv4 = weight_variable([kernel_x_4, kernel_y_4, kernel_t_4, layer3_channels, layer4_channels],'W4')
    h_conv4 = tf.nn.relu(conv3d_dilate(h_conv3, W_conv4))
    #h_conv4 = conv3d_dilate(h_conv3, W_conv4, Rx, Ry)

    W_conv5 = weight_variable([kernel_last_x, kernel_last_y, kernel_last_t, layer4_channels, target_dim4],'W5')
    h_conv5 = conv3d_dilate(h_conv4, W_conv5)

    #l = h_conv5.shape[-1]
    #print (tf.signal.fftshift(tf.signal.ifft3d(tf.signal.ifftshift(tf.complex(h_conv5[...,:l//2], h_conv5[...,l//2:])[0,...],axes=(0,1,2))),axes=(0,1,2)))

    #input_Target_ifft = tf.signal.ifft3d(tf.complex(input_Target[...,:l//2], input_Target[...,l//2:])[0,...])
    #h_conv5_ifft = tf.signal.ifft3d(tf.complex(h_conv5[...,:l//2], h_conv5[...,l//2:])[0,...])
    #print (tf.cast(tf.norm(input_Target_ifft - h_conv5_ifft, ord=2), dtype=tf.float32))
    #sub_ifft = input_Target_ifft - h_conv5_ifft
    #print (tf.real(sub_ifft))
    #print (tf.imag(sub_ifft))
    #print (tf.concat([tf.real(sub_ifft),tf.imag(sub_ifft)], -1))
    #raise

    #error_norm = tf.norm(input_Target - h_conv4)
    #error_norm = (tf.norm(input_Target - h_conv5, ord=2) + tf.norm(input_Target - h_conv5, ord=1))*0.5
    #error_norm = tf.norm(input_Target - h_conv5, ord=2)*0.7 + tf.norm(input_Target - h_conv5, ord=1)*0.3

    #error_norm = tf.norm(input_Target - h_conv5, ord=2)*0.7 + tf.norm(input_Target - h_conv5, ord=1)*0.3 + 1.5*(tf.nn.l2_loss(W_conv1)+tf.nn.l2_loss(W_conv2)+tf.nn.l2_loss(W_conv3))

    error_norm = tf.norm(input_Target - h_conv5, ord=2)*0.7 + tf.norm(input_Target - h_conv5, ord=1)*0.3 + 2.4*(tf.nn.l2_loss(W_conv1)+0.95*tf.nn.l2_loss(W_conv2)+0.95*tf.nn.l2_loss(W_conv3)+0.9*tf.nn.l2_loss(W_conv4)+0.8*tf.nn.l2_loss(W_conv5))

    #error_norm = tf.norm((input_Target - h_conv5)*np.abs(target)/np.max(np.abs(target)), ord=2)*0.7 + tf.norm((input_Target - h_conv5)*np.abs(target)/np.max(np.abs(target)), ord=1)*0.3 + 2.4*(tf.nn.l2_loss(W_conv1)+0.95*tf.nn.l2_loss(W_conv2)+0.95*tf.nn.l2_loss(W_conv3)+0.9*tf.nn.l2_loss(W_conv4)+0.8*tf.nn.l2_loss(W_conv5))

    #error_norm = tf.norm(input_Target - h_conv5, ord=2)*0.7 + tf.norm(input_Target - h_conv5, ord=1)*0.3 + 1.5*(tf.nn.l2_loss(W_conv1)+tf.nn.l2_loss(W_conv2)+tf.nn.l2_loss(W_conv3)+tf.nn.l2_loss(W_conv4)+tf.nn.l2_loss(W_conv5)) + 0.3*tf.cast(tf.norm(input_Target_ifft - h_conv5_ifft, ord=2), dtype=tf.float32)

    #error_norm = tf.norm(input_Target - h_conv5, ord=2)*0.7 + tf.norm(input_Target - h_conv5, ord=1)*0.3 + 2.5*(tf.nn.l2_loss(W_conv1)+tf.nn.l2_loss(W_conv2)+tf.nn.l2_loss(W_conv3)+tf.nn.l2_loss(W_conv4)+tf.nn.l2_loss(W_conv5)) + 1.5*tf.nn.l2_loss(tf.concat([tf.real(sub_ifft),tf.imag(sub_ifft)], -1))


    #error_norm = tf.norm(input_Target - h_conv5, ord=2)*0.7 + tf.norm(input_Target - h_conv5, ord=1)*0.3 + 2*(tf.norm(W_conv1, ord=1)+tf.norm(W_conv2, ord=1)+tf.norm(W_conv3, ord=1)+tf.norm(W_conv4, ord=1)+tf.norm(W_conv5, ord=1))

    #error_norm = tf.norm(input_Target - h_conv5, ord=2)*0.7 + tf.norm(input_Target - h_conv5, ord=1)*0.3 + 2*(tf.nn.l2_loss(W_conv1)+tf.nn.l2_loss(W_conv2)+tf.nn.l2_loss(W_conv3)+tf.nn.l2_loss(W_conv4)+tf.nn.l2_loss(W_conv5)) + 0.05*(tf.norm(W_conv1, ord=1)+tf.norm(W_conv2, ord=1)+tf.norm(W_conv3, ord=1)+tf.norm(W_conv4, ord=1)+tf.norm(W_conv5, ord=1))

    #error_norm = tf.norm(input_Target - h_conv5, ord=2)*0.7 + tf.norm(input_Target - h_conv5, ord=1)*0.3 + 0.5*(tf.norm(W_conv1, ord=2)+tf.norm(W_conv2, ord=2)+tf.norm(W_conv3, ord=2)+tf.norm(W_conv4, ord=2)+tf.norm(W_conv5, ord=2))

    #error_norm = tf.nn.l2_loss(input_Target - h_conv5)
    #error_norm = tf.norm(input_Target - h_conv4, ord=1)

    #error_norm_tmp = tf.sqrt(tf.reduce_sum(tf.pow(input_Target - h_conv3,2), axis=(0,1,2,4)))
    #error_norm_tmp = tf.sqrt(tf.reduce_sum(tf.pow(input_Target - h_conv5,2), axis=(0,1,2,3)))
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
        #print (type(ACS))
        #print (ACS.shape)
        #print (ACS.max(), ACS.min())
        #noise = np.random.normal(0, ACS.max()*0.0001,ACS.shape)
        #print (noise.shape)
        #print (noise.max(), noise.min())
        #raise
        #ACS = ACS + noise

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

def cnn_3layer(input_kspace,w1,b1,w2,b2,w3,b3,w4,w5,Rx,Ry,sess):
    print ('input_kspace shape: ', input_kspace.shape)
    input_phdr = tf.placeholder(tf.float32, input_kspace.shape)
    w1_phdr = tf.placeholder(tf.float32, w1.shape)
    w2_phdr = tf.placeholder(tf.float32, w2.shape)
    w3_phdr = tf.placeholder(tf.float32, w3.shape)
    w4_phdr = tf.placeholder(tf.float32, w4.shape)
    w5_phdr = tf.placeholder(tf.float32, w5.shape)

    #h_conv1 = tf.nn.relu(conv3d_dilate(input_kspace, w1,Rx,Ry))
    h_conv1 = conv3d_dilate(input_phdr, w1_phdr, strides=[7,7,1])
    h_conv2 = conv3d_dilate(h_conv1, w2_phdr)
    #h_conv2 = tf.nn.relu(conv3d_dilate(h_conv1, w2, Rx,Ry))
    #h_conv3 = tf.nn.relu(conv3d_dilate(h_conv2, w3,Rx,Ry))
    h_conv3 = conv3d_dilate(h_conv2, w3_phdr)

    h_conv4 = tf.nn.relu(conv3d_dilate(h_conv3, w4_phdr))
    #h_conv4 = conv3d_dilate(h_conv3, w4, Rx,Ry)
    h_conv5 = conv3d_dilate(h_conv4, w5_phdr)

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    return sess.run(h_conv5, feed_dict={input_phdr: input_kspace, w1_phdr: w1, w2_phdr: w2, w3_phdr: w3, w4_phdr: w4, w5_phdr: w5})



def raki(num_chan, N1, N2, Nt, acs_start_index_x, acs_end_index_x, acs_start_index_y, acs_end_index_y, acs_start_index_t, acs_end_index_t, kspace_undersampled_zero_filled, kspace_acs_zero_filled, kspace_coils_acs_out, kspace_acs_cropped, Rx, Ry, debug_path, name_image, name_weight,train_flag=True, finetune=False, slice_n=0):

    acs_mask = np.zeros((num_chan, N1, N2, Nt), dtype=bool)
    acs_mask[:,acs_start_index_x:acs_end_index_x,acs_start_index_y:acs_end_index_y, acs_start_index_t:acs_end_index_t] = True

    #kspace = kspace_undersampled_zero_filled * np.invert(acs_mask).astype(int) + kspace_acs_zero_filled[:,:,:,:]

    #kspace = np.rot90(kspace,k=1,axes=(1,2))
    #kspace = np.transpose(kspace,(2,1,3,0))

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

    #### Network Parameters ####
    kernel_x_1 = 8
    kernel_y_1 = 8
    kernel_t_1 = 9

    kernel_x_2 = 1
    kernel_y_2 = 1
    kernel_t_2 = 7

    kernel_x_3 = 1
    kernel_y_3 = 1
    kernel_t_3 = 5

    kernel_x_4 = 1 
    kernel_y_4 = 1
    kernel_t_4 = 3

    kernel_last_x = 1 
    kernel_last_y = 1
    kernel_last_t = 1

    layer1_channels = 64
    layer2_channels = 64
    layer3_channels = 64 
    layer4_channels = 64

    MaxIteration = 300
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

    # Normalization
    normalize = 0.15/max(np.max(abs(kspace_undersampled_zero_filled[:])), np.max(abs(kspace_acs_zero_filled[:])))
    #normalize = 0.15/np.max(abs(kspace[:]))
    #kspace = np.multiply(kspace,normalize)
    kspace_undersampled_zero_filled = np.multiply(kspace_undersampled_zero_filled,normalize)
    kspace_acs_cropped = np.multiply(kspace_acs_cropped,normalize)

    #normalize_out = 0.015/np.max(abs(kspace_coils_acs_out[:]))
    #kspace_coils_acs_out = np.multiply(kspace_coils_acs_out,normalize_out)
    kspace_coils_acs_out = np.multiply(kspace_coils_acs_out,normalize)

    #print (np.max(abs(kspace[:])))
    #print (np.max(abs(kspace_coils_acs_out[:])))


    #return normalize

    # Get the shapes
    #[m1,n1,no_t,no_ch] = np.shape(kspace)
    #no_inds = 1

    #kspace_all = np.copy(kspace);


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
    ACS_re[:,:,:,0:ACS_dim_Z] = np.real(ACS)
    ACS_re[:,:,:,ACS_dim_Z:ACS_dim_Z*2] = np.imag(ACS)

    [ACS_out_dim_X, ACS_out_dim_Y, ACS_out_dim_t, ACS_out_dim_Z] = np.shape(ACS_out)
    ACS_out_re = np.zeros([ACS_out_dim_X,ACS_out_dim_Y,ACS_out_dim_t,ACS_out_dim_Z*2])
    ACS_out_re[:,:,:,0:ACS_out_dim_Z] = np.real(ACS_out)
    ACS_out_re[:,:,:,ACS_out_dim_Z:ACS_out_dim_Z*2] = np.imag(ACS_out)



    # Get the acceleration rate
    no_channels = ACS_out_dim_Z*2

    #return normalize, name_image


    #w1_all = np.zeros([kernel_x_1, kernel_y_1, kernel_t_1, no_channels, layer1_channels, no_channels],dtype=np.float32)
    #w2_all = np.zeros([kernel_x_2, kernel_y_2, kernel_t_2, layer1_channels,layer2_channels,no_channels],dtype=np.float32)
    #w3_all = np.zeros([kernel_last_x, kernel_last_y, kernel_last_t, layer2_channels,acc_rate - 1, no_channels],dtype=np.float32)

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

    #target_t_start = np.int32(np.ceil(kernel_t_1/2) + np.floor(kernel_t_2/2) + np.floor(kernel_t_3/2) + np.floor(kernel_t_4/2) + np.floor(kernel_last_t/2) -1) 
    #target_t_end = np.int32(ACS_out_dim_t - target_t_start -1)

    target_t_start = np.int32((np.ceil(kernel_t_1/2)-1) + (np.ceil(kernel_t_2/2)-1) + (np.ceil(kernel_t_3/2)-1) + (np.ceil(kernel_t_4/2)-1) + (np.ceil(kernel_last_t/2)-1)) * 1;
    target_t_end = ACS_out_dim_t  - np.int32((np.floor(kernel_t_1/2) + np.floor(kernel_t_2/2) + np.floor(kernel_t_3/2) + np.floor(kernel_t_4/2) + np.floor(kernel_last_t/2))) * 1 -1;


    time_ALL_start = time.time()

    [ACS_dim_X, ACS_dim_Y, ACS_dim_T, ACS_dim_Z] = np.shape(ACS_re)
    ACS = np.reshape(ACS_re, [1,ACS_dim_X, ACS_dim_Y, ACS_dim_T, ACS_dim_Z])
    ACS = np.float32(ACS)


    [ACS_out_dim_X, ACS_out_dim_Y, ACS_out_dim_t, ACS_out_dim_Z] = np.shape(ACS_out_re)
    ACS_out = np.reshape(ACS_out_re, [1,ACS_out_dim_X, ACS_out_dim_Y,ACS_out_dim_t, ACS_out_dim_Z])
    ACS_out = np.float32(ACS_out)



    target_x_start = 2
    target_x_end = ACS_out_dim_X  - 5 - 1;

    target_y_start = 3
    target_y_end = ACS_out_dim_Y  -  4 - 1;

    #print (target_x_start, target_x_end)
    #print (target_y_start, target_y_end)
    #print (target_t_start, target_t_end)

    target_dim_X = target_x_end - target_x_start + 1
    target_dim_Y = target_y_end - target_y_start + 1
    target_dim_T = target_t_end - target_t_start + 1
    #target_dim_Z = (acc_rate - 1)*ACS_out_dim_Z
    target_dim_Z = (Rx * Ry - 1) * ACS_out_dim_Z
    #print (Rx, Ry, target_dim_Z)

    # ## Training
    if train_flag:
        print('go!')

        time_Learn_start = time.time()

        errorSum = 0;
        config = tf.ConfigProto()
        #config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        target = np.zeros([1,target_dim_X,target_dim_Y,target_dim_T,target_dim_Z])
        print (target.shape)

        for ind_c in range(ACS_out_dim_Z):
            #for ind_acc in range(acc_rate-1):
            for ind_acc_x in range(Rx):

                target_x_start = 2 + ind_acc_x;
                target_x_end = ACS_out_dim_X  - 5 + ind_acc_x - 1;
                #print ('target_x_start, target_x_end: ', target_x_start, target_x_end)
                for ind_acc_y in range(Ry):
                    target_y_start = 3 + ind_acc_y
                    target_y_end = ACS_out_dim_Y  - 4 + ind_acc_y - 1;
                    #print ('target_y_start, target_y_end: ', target_y_start, target_y_end)
                    #print (ind_acc_y+ind_acc_x*Ry+ind_c*Ry*Rx)
                    if ind_acc_x == 3 and ind_acc_y == 1:
                        continue
                    #print (ind_acc_y+ind_acc_x*Ry+ind_c*(Ry*Rx-1))
                    target[0,:,:,:,ind_acc_y+ind_acc_x*Ry+ind_c*(Ry*Rx-1)] = ACS_out[0,target_x_start:target_x_end + 1, target_y_start:target_y_end +1,target_t_start:target_t_end + 1,ind_c];

        sess = tf.Session(config=config)

        [w1,w2,w3,w4,w5]=learning(ACS,target,Rx, Ry, sess, ACS_dim_X, ACS_dim_Y, ACS_dim_T, ACS_dim_Z, target_dim_X,target_dim_Y,target_dim_T,target_dim_Z, target, kernel_x_1, kernel_x_2, kernel_x_3, kernel_x_4, kernel_y_1, kernel_y_2, kernel_y_3, kernel_y_4, kernel_t_1, kernel_t_2, kernel_t_3, kernel_t_4, layer1_channels, layer2_channels, layer3_channels, layer4_channels, kernel_last_x, kernel_last_y, kernel_last_t, LearningRate, MaxIteration, finetune,slice_n)
        w1_all = w1
        w2_all = w2
        w3_all = w3
        w4_all = w4
        w5_all = w5


        sess.close()
        tf.reset_default_graph()

        time_Learn_end = time.time();
        print('lerning step costs:',(time_Learn_end - time_Learn_start),'s')
        sio.savemat(name_weight, {'w1': w1_all,'w2': w2_all,'w3': w3_all,'w4': w4_all,'w5': w5_all})

    # ## Reconstruction

    w_all = sio.loadmat(name_weight)
    w1_all = w_all['w1']
    w2_all = w_all['w2']
    w3_all = w_all['w3']
    w4_all = w_all['w4']
    w5_all = w_all['w5']

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

    kspace_und_out = kspace_und[:,:,:,0]
    kspace_und_out = np.expand_dims(kspace_und_out, axis=-1)
    [dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_T,dim_kspaceUnd_Z] = np.shape(kspace_und_out)
    kspace_und_out_re = np.zeros([dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_T,dim_kspaceUnd_Z*2])
    #kspace_und_out_re[:,:,:,0:dim_kspaceUnd_Z] = np.real(kspace_und_out)
    #kspace_und_out_re[:,:,:,dim_kspaceUnd_Z:dim_kspaceUnd_Z*2] = np.imag(kspace_und_out)
    #kspace_und_out_re = np.float32(kspace_und_out_re)
    kspace_und_out_re = np.reshape(kspace_und_out_re,[1,dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_T,dim_kspaceUnd_Z*2])

    #kspace_recon = kspace_und_out_re
    kspace_recon = kspace_und_out_re.copy()

    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 1/3 ;


    print('Reconstruting Channel #')

    sess = tf.Session(config=config)
    #if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    #    init = tf.initialize_all_variables()
    #else:
    #    init = tf.global_variables_initializer()
    #sess.run(init)

    # grab w and b
    w1 = np.float32(w1_all)
    w2 = np.float32(w2_all)
    w3 = np.float32(w3_all)
    w4 = np.float32(w4_all)
    w5 = np.float32(w5_all)

    w1_0 = w1[:,:,:,0:32,:] + w1[:,:,:,32:32*2,:] + w1[:,:,:,32*2:32*3,:]
    w1_1 = w1[:,:,:,32*3:32*4,:] + w1[:,:,:,32*4:32*5,:] + w1[:,:,:,32*5:32*6,:]
    w1 = np.concatenate((w1_0, w1_1), axis=3)

    #w1_0 = w1[:,:,:,0:32,:] + w1[:,:,:,32:64,:] 
    #w1_1 = w1[:,:,:,64:96,:] + w1[:,:,:,96:128,:]
    #w1 = np.concatenate((w1_0, w1_1), axis=3)

    '''
    print (w1.shape)
    print (kspace_und_re.shape)
    print (w1[:,:,0,0,0])
    print ('*'*10)
    print (kspace_und_re[0,100:110,100:110,0,0])
    raise
    '''

    if (b1_flag == 1):
        b1 = b1_all[:,:,:,ind_c];
    if (b2_flag == 1):
        b2 = b2_all[:,:,:,ind_c];
    if (b3_flag == 1):
        b3 = b3_all[:,:,:,ind_c];


    offset_list = [(0,0), (3,1), (6,2), (2,3), (5,4), (1,5), (4,6)]
    for offset in offset_list:
        offset_x = offset[0]
        offset_y = offset[1]

        res = cnn_3layer(kspace_und_re[:,offset_x:,offset_y:,:,:],w1,b1,w2,b2,w3,b3,w4,w5,1,1,sess)

        target_t_start = np.int32((np.ceil(kernel_t_1/2)-1) + (np.ceil(kernel_t_2/2)-1) + (np.ceil(kernel_t_3/2)-1) + (np.ceil(kernel_t_4/2)-1) + (np.ceil(kernel_last_t/2)-1)) * 1;
        target_t_end = ACS_out_dim_t  - np.int32((np.floor(kernel_t_1/2) + np.floor(kernel_t_2/2) + np.floor(kernel_t_3/2) + np.floor(kernel_t_4/2) + np.floor(kernel_last_t/2))) * 1 -1;

        for ind_c in range(0,no_channels):

            for ind_acc_x in range(Rx):

                target_x_start = 2 + ind_acc_x;
                target_x_end = dim_kspaceUnd_X  - 5 + ind_acc_x - 1;

                for ind_acc_y in range(Ry):
                    target_y_start = 3 + ind_acc_y
                    target_y_end = dim_kspaceUnd_Y  - 4+ ind_acc_y - 1;
                    if ind_acc_x == 3 and ind_acc_y == 1:
                        continue
                    kspace_recon[0,target_x_start+offset_x:target_x_end + 1:7, target_y_start+offset_y:target_y_end +1:7,target_t_start:target_t_end + 1,ind_c] = res[0,:,:,:,ind_acc_y+ind_acc_x*Ry+ind_c*(Ry*Rx-1)]



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
    kspace_recon_complex[acs_start_index_y:acs_end_index_y,acs_start_index_x:acs_end_index_x,acs_start_index_t:acs_end_index_t,:] = kspace_coils_acs_out

    #kspace_recon_all[:,:,:] = kspace_recon_complex;

    #for sli in range(0,no_ch):
    #    kspace_recon_all[:,:,:,sli] = np.fft.ifft2(kspace_recon_all[:,:,:,sli])

    #rssq = (np.sum(np.abs(kspace_recon_all)**2,2)**(0.5))
    sio.savemat(name_image,{recon_variable_name:kspace_recon_complex, 'normalize':normalize})

    time_ALL_end = time.time()
    print('All process costs ',(time_ALL_end-time_ALL_start),'s')
    #print('Error Average in Training is ',errorSum/no_channels)

def show_raki_results(image_original0, debug_path, name_image):
    # ## Results
    # Read the kspace
    dt = sio.loadmat(name_image)

    kspace_recon = dt['kspace_recon']
    normalize = dt['normalize']
    kspace_recon = np.transpose(kspace_recon,(3,1,0,2))
    #kspace_recon = np.rot90(kspace_recon,k=-1,axes=(1,2))
    kspace_recon = kspace_recon/normalize

    #mosaic(kspace_recon, 4, 8, 1, [0,50], fig_title='kspace_recon', fig_base_path=debug_path, fig_size=(15,15), num_rot=0)

    #kspace_recon = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kspace_recon, axes=3), axis=3), axes=3)

    # Get the RAKI coil images
    #coil_images_raki_sub = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace_recon, axes=(1,2)), axes=(1,2)), axes=(1,2))
    coil_images_raki_sub = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace_recon, axes=(1,2,3)), axes=(1,2,3)), axes=(1,2,3))

    #coil_images_raki_sub = kspace_recon

    #raki_rssq_image_sub0 = np.sqrt(np.sum(np.square(np.abs(coil_images_raki_sub)),axis=0))
    raki_rssq_image_sub0 = np.abs(coil_images_raki_sub)[0]

    print (image_original0.shape)
    print (raki_rssq_image_sub0.shape)

    rmse_list = []

    #for i in range(75,90):
    for i in range(50,150):
    #for i in range(len(image_original0[0,0,:])):
        image_original = image_original0[:,:,i]
        raki_rssq_image_sub = raki_rssq_image_sub0[:,:,i]

        image_ref0 = image_original.copy()
        mask = image_ref0 > 0.6 * np.mean(image_ref0)
        mask = 1.0 * mask
        image_original = image_original * mask
        raki_rssq_image_sub = raki_rssq_image_sub * mask

        #mosaic(raki_rssq_image_sub, 1, 1, 1, [0,150], fig_title=f'raki_rssq_image_sub_{i}', fig_base_path=debug_path, fig_size=(10,10), num_rot=0)
        #mosaic(image_original, 1, 1, 1, [0,150], fig_title=f'image_original_{i}', fig_base_path=debug_path, fig_size=(10,10), num_rot=0)

        rmse_raki_sub = np.sqrt(np.sum(np.square(np.abs(image_original - raki_rssq_image_sub)),axis=(0,1))) / np.sqrt(np.sum(np.square(np.abs(image_original)),axis=(0,1)))
        print('rmse_raki: ' + str(rmse_raki_sub))
        rmse_list.append(rmse_raki_sub)
        continue
        
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
    #raise
    rmse_np = np.array(rmse_list)
    data = pd.DataFrame(rmse_np)
    writer = pd.ExcelWriter(f'{debug_path}/rmse.xlsx')
    data.to_excel(writer, 'rmse', float_format='%.6f')
    writer.save()
    writer.close()


def show_raki_results_3d(image_original0, debug_path0, name_image):


    # ## Results
    # Read the kspace
    dt = sio.loadmat(name_image)

    kspace_recon = dt['kspace_recon']
    normalize = dt['normalize']
    kspace_recon = np.transpose(kspace_recon,(3,1,0,2))
    #kspace_recon = np.rot90(kspace_recon,k=-1,axes=(1,2))
    kspace_recon = kspace_recon/normalize

    #kspace_recon = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kspace_recon, axes=3), axis=3), axes=3)

    # Get the RAKI coil images
    coil_images_raki_sub = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(kspace_recon, axes=(1,2,3)), axes=(1,2,3)), axes=(1,2,3))
    #raki_rssq_image_sub0 = np.sqrt(np.sum(np.square(np.abs(coil_images_raki_sub)),axis=0))

    raki_rssq_image_sub0 = np.abs(coil_images_raki_sub)[0]

    #print (coil_images_raki_sub.shape)
    #print (sens_resize.shape)
    print (raki_rssq_image_sub0.shape)
    print (image_original0.shape)
    sio.savemat(osp.join(debug_path0, f'recon_image_3d.mat'),{'data':raki_rssq_image_sub0})

    # using bet mask
    mask0 = sio.loadmat('../bet_mask/msk.mat')
    mask = mask0['msk']
    #mask = np.transpose(mask,(1,0,2))

    mask = np.transpose(mask,(0,1,2))


    #print ('bet mask: ', mask.shape)
    #mask = image_original0 > 0.6 * np.mean(image_original0)
    #print ('mean mask: ', mask.shape)
    #raise

    image_original = image_original0 * mask
    raki_rssq_image_sub = raki_rssq_image_sub0 * mask
    #sio.savemat(osp.join(debug_path0, f'recon_image_3d_bet.mat'),{'data':raki_rssq_image_sub})

    #image_original = image_original[...,50:150]
    #raki_rssq_image_sub = raki_rssq_image_sub[...,50:150]

    rmse_raki_sub = np.sqrt(np.sum(np.square(np.abs(image_original - raki_rssq_image_sub)),axis=(0,1,2))) / np.sqrt(np.sum(np.square(np.abs(image_original)),axis=(0,1,2)))

    rmse_str = str(rmse_raki_sub)
    print ('3d betmask rmse: ', rmse_raki_sub)

    ssim_e = ssim(raki_rssq_image_sub, image_original, data_range=image_original.max() - image_original.min())
    ssim_e_str = str(ssim_e)
    print ('3d betmask ssim: ', ssim_e_str)
    #raise

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
        #print (raki_rssq_image_sub_i.shape)
        #raki_rssq_image_sub_i = cv2.GaussianBlur(raki_rssq_image_sub_i,(3,3),0)
        #print (raki_rssq_image_sub_i.shape)
        #raise

        mosaic(raki_rssq_image_sub_i, 1, 1, 1, [0,150], fig_title=f'raki_rssq_image_sub_{i}', fig_base_path=debug_path, fig_size=(10,10), num_rot=0)
        #mosaic(image_original_i, 1, 1, 1, [0,150], fig_title=f'image_original_{i}', fig_base_path=debug_path, fig_size=(10,10), num_rot=0)

        rmse_raki_sub = np.sqrt(np.sum(np.square(np.abs(image_original_i - raki_rssq_image_sub_i)),axis=(0,1))) / np.sqrt(np.sum(np.square(np.abs(image_original_i)),axis=(0,1)))
        print('rmse_raki: ' + str(rmse_raki_sub))
        rmse_list.append(rmse_raki_sub)
        continue 

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










