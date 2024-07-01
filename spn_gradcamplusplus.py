


# https://keisen.github.io/tf-keras-vis-docs/






import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.gradcam import Gradcam
import os


#%%

# Create GradCAM++ object

def gradcamplusplus (items_no,predictions_all,labels,data,model3,verbose = False,show=False, save = True, base_path='C:\\Users\\User\\'):
    gradcam = GradcamPlusPlus(model3,
                              model_modifier=ReplaceToLinear(),
                              clone=True)
    
    def score_function(output):
    # The `output` variable refers to the output of the model,
    # so, in this case, `output` shape is `(3, 1000)` i.e., (samples, classes).
        return (output[0,0],output[0,1])
    
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])   
    
    

    n = 121
    for item in items_no:
        
        predicted_label = predictions_all[item]
        if predicted_label == 0:
            predicted_label_ling = 'Benign'
        elif predicted_label == 1:
            predicted_label_ling = 'Malignant'
        
        if labels[item,0] == 1:
            true_label_ling = 'Benign'
        if labels[item,1] == 1:
            true_label_ling = 'Malignant'

        no = ['a{}'.format(n),'v{}'.format(n)]
        n+=1     
        
        instance = data[item,:,:,:]
        the_image = instance
        instance = np.expand_dims(instance, axis=0)
        # Generate cam with GradCAM++
        
        score = CategoricalScore([int(predicted_label)])
        cam = gradcam(score, instance)
        #score = CategoricalScore([0,1])
        #cam = gradcam(score, img_tensor)
    
        colored = the_image
        # the_image =  rgb2gray(the_image)
        
        f, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
        heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
        ax1.imshow(the_image)
        ax1.imshow(heatmap, alpha=0.6)
        ax1.axis('off')
        
        ax2.imshow(colored)
        ax2.axis('off')
        
        
        ax1.set_title('Fused image',fontsize = 18)
        ax2.set_title('Original',fontsize = 18)
        f.suptitle('\n{} {}\n True Label: {} \n Predicted: {}'.format(no[0],no[0],true_label_ling,predicted_label_ling), fontsize=22)        
        
        plt.tight_layout()
        plt.show()
        
        name = '{}_pred{}_is{}_GC++'.format(no[0],predicted_label_ling,true_label_ling)
        
        save_path = os.path.join(base_path,'gradcamplusplus')
        if not os.path.exists (save_path) : os.mkdir(save_path)        
        
        f.savefig(os.path.join(save_path,name))
        plt.close()


#%%

def scorecam (items_no,predictions_all,labels,data,model3,verbose = False,show=False, save = True, base_path='C:\\Users\\User\\'):
    
    from tf_keras_vis.scorecam import Scorecam
    from tf_keras_vis.utils import num_of_gpus    
    
    scorecam = Scorecam(model3)
    
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])   
    
    

    n = 121
    for item in items_no:
        
        predicted_label = int(predictions_all[item])
        if predicted_label == 0:
            predicted_label_ling = 'Benign'
        elif predicted_label == 1:
            predicted_label_ling = 'Malignant'
        
        if labels[item,0] == 1:
            true_label_ling = 'Benign'
        if labels[item,1] == 1:
            true_label_ling = 'Malignant'

        no = ['a{}'.format(n),'v{}'.format(n)]
        n+=1     
        
        instance = data[item,:,:,:]
        the_image = instance
        instance = np.expand_dims(instance, axis=0)
        
        
        score = CategoricalScore([predicted_label])
        cam = scorecam(score, instance, penultimate_layer=-1)
        #score = CategoricalScore([0,1])
        #cam = gradcam(score, img_tensor)
    
        colored = the_image
        # the_image =  rgb2gray(the_image)
        
        f, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
        heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
        ax1.imshow(the_image)
        ax1.imshow(heatmap, alpha=0.6)
        ax1.axis('off')
        
        ax2.imshow(colored)
        ax2.axis('off')
        
        
        ax1.set_title('Fused image',fontsize = 18)
        ax2.set_title('Original',fontsize = 18)
        f.suptitle('\n{} {}\n True Label: {} \n Predicted: {}'.format(no[0],no[0],true_label_ling,predicted_label_ling), fontsize=22)        
        
        plt.tight_layout()
        plt.show()
        
        name = '{}_pred{}_is{}_SC'.format(no[0],predicted_label_ling,true_label_ling)
        
        save_path = os.path.join(base_path,'scorecam')
        if not os.path.exists (save_path) : os.mkdir(save_path)        
        
        f.savefig(os.path.join(save_path,name))
        plt.close()



def gradcam (items_no,predictions_all,labels,data,model3,verbose = False,show=False, save = True, base_path='C:\\Users\\User\\'):
    
    from tf_keras_vis.utils import num_of_gpus    
    
    
    gradcam = Gradcam(model3)
    
    
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])   
    
    

    n = 222
    for item in items_no:
        
        predicted_label = int(predictions_all[item])
        if predicted_label == 0:
            predicted_label_ling = 'Benign'
        elif predicted_label == 1:
            predicted_label_ling = 'Malignant'

        
        if labels[item,0] == 1:
            true_label_ling = 'Benign'
        if labels[item,1] == 1:
            true_label_ling = 'Malignant'


        no = ['a{}'.format(n),'v{}'.format(n)]
        n+=1     
        
        instance = data[item,:,:,:]
        the_image = instance
        instance = np.expand_dims(instance, axis=0)
        # Generate cam with GradCAM++
        score = CategoricalScore([predicted_label])
        cam = gradcam(score, instance, penultimate_layer=-1)
        #score = CategoricalScore([0,1])
        #cam = gradcam(score, img_tensor)
    
        colored = the_image
        # the_image =  rgb2gray(the_image)
        
        f, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
        heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
        ax1.imshow(the_image)
        ax1.imshow(heatmap, alpha=0.6)
        ax1.axis('off')
        
        ax2.imshow(colored)
        ax2.axis('off')
        
        
        ax1.set_title('Fused image',fontsize = 18)
        ax2.set_title('Original',fontsize = 18)
        f.suptitle('\n{} {}\n True Label: {} \n Predicted: {}'.format(no[0],no[0],true_label_ling,predicted_label_ling), fontsize=22)        
        
        plt.tight_layout()
        plt.show()
        
        name = '{}_pred{}_is{}_GC'.format(no[0],predicted_label_ling,true_label_ling)
        
        save_path = os.path.join(base_path,'GradCAM')
        if not os.path.exists (save_path) : os.mkdir(save_path)        
        
        f.savefig(os.path.join(save_path,name))
        plt.close()



#%%


def saliency (items_no,predictions_all,labels,data,model3,verbose = False,show=False, save = True, base_path='C:\\Users\\User\\'):
    
    from tf_keras_vis.saliency import Saliency
    from tf_keras_vis.utils import num_of_gpus    
    
    
    
    saliency = Saliency(model3,
                        model_modifier=ReplaceToLinear(),
                        clone=True)
    
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])   
    
    

    n = 222
    for item in items_no:
        
        predicted_label = int(predictions_all[item])
        if predicted_label == 0:
            predicted_label_ling = 'Benign'
        elif predicted_label == 1:
            predicted_label_ling = 'Malignant'

        
        if labels[item,0] == 1:
            true_label_ling = 'Benign'
        if labels[item,1] == 1:
            true_label_ling = 'Malignant'


        no = ['a{}'.format(n),'v{}'.format(n)]
        n+=1     
        
        instance = data[item,:,:,:]
        the_image = instance
        instance = np.expand_dims(instance, axis=0)
        # Generate cam with GradCAM++
        score = CategoricalScore([predicted_label])
        cam = saliency(score, instance)
        #score = CategoricalScore([0,1])
        #cam = gradcam(score, img_tensor)
    
        colored = the_image
        # the_image =  rgb2gray(the_image)
        
        f, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
        heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
        ax1.imshow(the_image)
        ax1.imshow(heatmap, alpha=0.6)
        ax1.axis('off')
        
        ax2.imshow(colored)
        ax2.axis('off')
        
        
        ax1.set_title('Fused image',fontsize = 18)
        ax2.set_title('Original',fontsize = 18)
        f.suptitle('\n{} {}\n True Label: {} \n Predicted: {}'.format(no[0],no[0],true_label_ling,predicted_label_ling), fontsize=22)        
        
        plt.tight_layout()
        plt.show()
        
        name = '{}_pred{}_is{}_SM'.format(no[0],predicted_label_ling,true_label_ling)
        
        save_path = os.path.join(base_path,'Saliency')
        if not os.path.exists (save_path) : os.mkdir(save_path)        
        
        f.savefig(os.path.join(save_path,name))
        plt.close()




#%%


def smoothgrad (items_no,predictions_all,labels,data,model3,verbose = False,show=False, save = True, base_path='C:\\Users\\User\\'):
    
    from tf_keras_vis.saliency import Saliency
    from tf_keras_vis.utils import num_of_gpus    
    
    
    
    saliency = Saliency(model3,
                        model_modifier=ReplaceToLinear(),
                        clone=True)
    
    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])   
    
    

    n = 222
    for item in items_no:
        
        predicted_label = int(predictions_all[item])
        if predicted_label == 0:
            predicted_label_ling = 'Benign'
        elif predicted_label == 1:
            predicted_label_ling = 'Malignant'

        
        if labels[item,0] == 1:
            true_label_ling = 'Benign'
        if labels[item,1] == 1:
            true_label_ling = 'Malignant'


        no = ['a{}'.format(n),'v{}'.format(n)]
        n+=1     
        
        instance = data[item,:,:,:]
        the_image = instance
        instance = np.expand_dims(instance, axis=0)
        # Generate cam with GradCAM++
        score = CategoricalScore([predicted_label])
        cam = saliency(score, instance,smooth_samples=5,smooth_noise=0.1)
        
        #score = CategoricalScore([0,1])
        #cam = gradcam(score, img_tensor)
    
        colored = the_image
        # the_image =  rgb2gray(the_image)
        
        f, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
        heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
        ax1.imshow(the_image)
        ax1.imshow(heatmap, alpha=0.6)
        ax1.axis('off')
        
        ax2.imshow(colored)
        ax2.axis('off')
        
        
        ax1.set_title('Fused image',fontsize = 18)
        ax2.set_title('Original',fontsize = 18)
        f.suptitle('\n{} {}\n True Label: {} \n Predicted: {}'.format(no[0],no[0],true_label_ling,predicted_label_ling), fontsize=22)        
        
        plt.tight_layout()
        plt.show()
        
        name = '{}_pred{}_is{}_SG'.format(no[0],predicted_label_ling,true_label_ling)
        
        save_path = os.path.join(base_path,'smoothgrad')
        if not os.path.exists (save_path) : os.mkdir(save_path)        
        
        f.savefig(os.path.join(save_path,name))
        plt.close()






