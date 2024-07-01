# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:48:38 2022

@author: User
"""

#%% GRad CAM

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf
import numpy as np



# to load an external image
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array1, model3, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model3.inputs], [model3.get_layer(last_conv_layer_name).output, model3.output]
    )
    
    #  grad_model = tf.keras.models.Model(
    #     [model3.inputs], [model3.get_layer('model').get_layer(last_conv_layer_name).output, model3.output]
    # )   

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array1)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# # Display heatmap
# plt.matshow(heatmap)
# plt.show()


def save_and_display_gradcam(img, heatmap, cam_path="no_name.jpg", alpha=0.4, save=True):
    # Load the original image
    #img = keras.preprocessing.image.load_img(img_path)
    #img = keras.preprocessing.image.img_to_array(img)
    # Rescale heatmap to a range 0-255
    heatmap2 = np.uint8(255 * heatmap)
    img  = img*255

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap2]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    if not save:
        default_path = "temp.jpg"
        # Save the superimposed image
        superimposed_img.save(default_path)
    
        # Display Grad CAM
        display(Image(default_path))
    else:
        superimposed_img.save(cam_path)
    
        # Display Grad CAM
        #display(Image(cam_path))        



def GRADCAM ( model3, img_list, info, data,labels, predictions_all,
             last_conv_layer_name, alpha = 0.4, save = True ):
    
    '''
        model3: the saved model (tf file)
        img_lsit: a list of img numbers that corresponds to the data array. E.g [0,1,200]
        labels: the whole labels list
        last_conv_layer_name: found with mode3.layer_names
        alpha: defaults to 0.4
        save: True \ False
        base_name_save: the base name.The final files will be saved with "base_name_save_grad_x_label"
    
    '''
    

    # Remove last layer's softmax
    model3.layers[-1].activation = None
    
    for img_no in img_list:
    
        # display(Image(img_path))
        img_array1 = data[img_no,:,:,:]
        img_array1 = np.expand_dims(img_array1, axis=0)
 
        
        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(img_array1, model3, last_conv_layer_name)
        
        
        if save:
            
            if labels[img_no][0]==1:
                label = 'ISnormal'
            else:
                label = 'ISparathyroid'
                
            if predictions_all[img_no] == 1:
                predicted = 'predPARATHYR'
            else:
                predicted = 'predNORMAL'
            
            name = '{}_all_{}_{}.jpg'.format('a',label,predicted)
            save_and_display_gradcam(img = data[img_no,:,:,:], heatmap=heatmap,cam_path=name, alpha=alpha, save=True)
            
            img_array1 = tf.keras.preprocessing.image.array_to_img(img_array1[0,:,:,:])
            
            the_path = '{}_all.jpg'.format('a')
            
            img_array1.save(the_path)
            
            print ('Label: {}'.format(labels[img_no]))
            print ('Predicted: {}'.format(predictions_all[img_no]))
        else:
            name = 'temp.jpg'
            save_and_display_gradcam(img = data[img_no,:,:,:], heatmap=heatmap,cam_path=name, alpha=alpha, save=True)
            print ('Label: {}'.format(labels[img_no]))
            print ('Predicted: {}'.format(predictions_all[img_no]))