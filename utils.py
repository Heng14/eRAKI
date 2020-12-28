import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

# Notes
#-------
# 1) Given an input image of shape [num_images, N1, N2], concatenates all images in a mosaic and displays it
# 2) num_images must match num_row * num_col
def mosaic(img, num_row, num_col, fig_num, clim, fig_title='', fig_base_path='debug_fig', num_rot=0, fig_size=(18, 16)):
    
    fig = plt.figure(fig_num, figsize=fig_size)
    fig.patch.set_facecolor('black')
    title_str = osp.join(fig_base_path, fig_title)

    img = np.abs(img);
    #print ('img max:', np.max(img))
    img = img.astype(float)
        
    if img.ndim < 3:
        img = np.rot90(img, k=num_rot, axes=(0,1))
        img_res = img
        plt.imshow(img_res)
        plt.gray()        
        #plt.clim(clim)
        #title_str = fig_title
        plt.savefig(title_str + '.png')

    else: 
        img = np.rot90(img, k=num_rot,axes=(1,2))
        
        if img.shape[0] != (num_row * num_col):
            print('sizes do not match')    
        else:   
            img_res = np.zeros((img.shape[1]*num_row, img.shape[2]*num_col))
            
            idx = 0
            for r in range(0, num_row):
                for c in range(0, num_col):
                    img_res[r*img.shape[1] : (r+1)*img.shape[1], c*img.shape[2] : (c+1)*img.shape[2]] = img[idx,:,:]
                    idx = idx + 1
               
        plt.imshow(img_res)
        plt.gray()        
        #plt.clim(clim)
        plt.title(fig_title, color='white')
        #title_str = fig_title
        plt.savefig(title_str + '.png')

