# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:53:20 2022

@author: User
"""

'''






LIME




'''

    

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
# https://lime-ml.readthedocs.io/en/latest/lime.html
from skimage.segmentation import mark_boundaries

import numpy as np
import matplotlib.pyplot as plt
import os


# segmentation_fn = SegmentationAlgorithm('quickshift',kernel_size=4,
#                                                     max_dist=200, ratio=0.01,
#                                                     random_seed=12)

def explanation_heatmap(explanation, exp_class,save=False,show=True,name = 'foo.png',path='C:\\Users\\User\\'):
    '''
    Using heat-map to highlight the importance of each super-pixel for the model prediction
    '''
    dict_heatmap = dict(explanation.local_exp[exp_class])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 
    plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
    plt.colorbar()
    plt.title("Heatmap of segments")
    
    if show:
        plt.show()
    if save:
        path = os.path.join(path,name)
        plt.savefig(path)
        plt.close()


def explanation_visual (temp_1,mask_1,predicted_label_ling,true_label_ling,no,save=False,show=True,name = 'foo.png',path='C:\\Users\\User\\'):
    
    fig, ax1 = plt.subplots(1,1,figsize=(15,15))
    ax1.imshow(mark_boundaries(temp_1, mask_1))
    ax1.axis('off')

    
    
    ax1.set_title('Most confident segment',fontsize = 18)
    fig.suptitle('\n{} {}\n True Label: {} \n Predicted: {}'.format(no[0],no[1],true_label_ling,predicted_label_ling), fontsize=22)
    
    
    if show:
        plt.show()
    if save:
        path = os.path.join(path,name)
        plt.show()
        fig.savefig(path)  
        plt.close(fig)
        plt.close()



def the_lime (items_no,predictions_all,labels,data,info,model3,verbose = False,show=False, save = True, base_path='C:\\Users\\User\\'):

    # https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html#sphx-glr-download-auto-examples-segmentation-plot-segmentations-py
    
    # segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    # segments_slic = slic(img, n_segments=250, compactness=10, sigma=1,
    #                  start_label=1)
    # segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    # gradient = sobel(rgb2gray(img))
    # segments_watershed = watershed(gradient, markers=250, compactness=0.001)



    # segmentation_fn = SegmentationAlgorithm('slic',scale=100, sigma=0.5, min_size=50)
    # segmentation_fn2 = SegmentationAlgorithm('slic',scale=100, sigma=0.5, min_size=50)
    
    segmentation_fn = SegmentationAlgorithm('felzenszwalb',scale=18, sigma=0.5, min_size=5)
    #segmentation_fn2 = SegmentationAlgorithm('felzenszwalb',scale=100, sigma=0.5, min_size=50)    
    
    explainer = lime_image.LimeImageExplainer(verbose=True, feature_selection='auto')
    
    n = 1
    for item in items_no:
        predicted_label = predictions_all[item]
        predicted_label_ling = str(predicted_label)
        true_label = labels[item,1]
        true_label_ling = str(true_label)
        no = ['a{}'.format(n),'v{}'.format(n)]
        n+=1
        
        explanation = explainer.explain_instance(data[item].astype('double'), model3.predict,  
                                                 top_labels=3, hide_color=0, num_samples=1000, num_features = 100000,segmentation_fn = segmentation_fn)
        
        
        # explanation2 = explainer.explain_instance(data[item].astype('double'), model3.predict,  
        #                                          top_labels=2, hide_color=0, num_samples=1000, num_features = 100000,segmentation_fn = segmentation_fn2)        
        
        
        temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True,negative_only = False, num_features=4, hide_rest=False)
        #temp_2, mask_2 = explanation2.get_image_and_mask(explanation2.top_labels[0], positive_only=True,negative_only = False, num_features=3, hide_rest=False)
        
        
        if verbose:
            print ('PREDICTED: {} || TRUE: {}'.format(predicted_label,true_label))
            print(no)
            
        
        name = '{}_pred{}_is{}.png'.format(no[0],predicted_label_ling,true_label_ling)
        
        
        save_path = os.path.join(base_path,'LIME')
    
    
        if not os.path.exists (save_path) : os.mkdir(save_path)
        explanation_visual (temp_1,mask_1,predicted_label_ling,true_label_ling,no,save,show,name = name,path=save_path)    
            
         
        plt.close()
        
        exp_class = 0   
        #explanation_heatmap(explanation, exp_class,save,show,name,save_path)
