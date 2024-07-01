# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:10:21 2022

@author: User
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def feature_maps(dset,info,howmany,save_path,model3):
    

    import os


    # redefine model to output right after the first hidden layer
    ixs = [2, 5, 9, 13, 17, 25, 35, 45, 60]
    outputs = [model3.layers[i].output for i in ixs]
    model2 = tf.keras.Model(inputs=model3.inputs, outputs=outputs)
    # load the image with the required shape
    
    for i in range(howmany):
    
        img = dset[i,:,:,:]
        img = np.expand_dims(img, axis=0)
        # expand dimensions so that it represents a single 'sample'
        # prepare the image (e.g. scale pixel values for the vgg)
        # img = preprocess_input(img)
        # get feature map for first hidden layer
        
        
        
        feature_maps = model2.predict(img)
        # plot the output from each block
        square = 3
        o = 1
        for fmap in feature_maps:
         	# plot all 64 maps in an 8x8 squares
             ix = 1
             for _ in range(square):
                 for _ in range(square):
                     ax = plt.subplot(square, square, ix)
                     ax.set_xticks([])
                     ax.set_yticks([]); fig = plt.gcf();
                     plt.imshow(fmap[0, :, :, ix-1], cmap='gist_gray')
                     ix += 1
             o=o+1
             
             the_id = info[i,0]
             name = '{}_featuremap.png'.format(the_id)
             the_save_path = os.path.join(save_path,name)
             fig.savefig(the_save_path)
             plt.close(fig)