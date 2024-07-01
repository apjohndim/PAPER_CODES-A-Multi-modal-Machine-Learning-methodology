# -*- coding: utf-8 -*-

''' LOAD BASIC LIBRARIRES'''

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import random
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder

''' LOAD CUSTOM MODULES'''

import spn_model_maker
import spn_metrics
import spn_model_evaluation_plots


def model_save_load(data,labels,epochs,batch_size, model_name, in_shape, tune, classes,n_split,augmentation,verbose):
    
    model3 = spn_model_maker.selector(model_name)
    model3.save('C:\\Users\\User\\{}.h5'.format(model_name))
    loaded_trained_model = tf.keras.models.load_model('C:\\Users\\User\\{}.h5'.format(model_name))
    
    return loaded_trained_model


def add_noise(img):
    '''Add random noise to an image'''
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    img = np.random.poisson(img * vals) / float(vals)
    
    return img
  
    
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def prepare_clinical_data(predictions_ct,predictions_pet,rest_clinical,labels,selected_features):
    
    def convert_to_numeric(column):
        try:
            return pd.to_numeric(column)
        except ValueError as e:
            print (e)
            return column
    
    # covnert labels to dummies if not
    if labels.shape[1]<2:
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
    
    # Drop inwanted columns, keep index
    the_index = rest_clinical['ID']
    rest_clinical=rest_clinical.drop('ID',axis=1)
    rest_clinical=rest_clinical.drop('LABEL BASED ON BIOPSY (1), FOLLOW-UP (2), DOCTOR (3)',axis=1)
    rest_clinical=rest_clinical.drop('LABEL',axis=1)
    
    # recognize numeric columns (essential for getting the dummies later)
    for col in rest_clinical.columns:
        rest_clinical[col] = convert_to_numeric(rest_clinical[col])
        
    rest_clinical_dummies = pd.get_dummies(rest_clinical, columns=rest_clinical.select_dtypes(include=['object']).columns)
    
    ml_data = rest_clinical_dummies[selected_features]
    
    ml_data['CTimg_pred_benign'] = predictions_ct[:,0]
    ml_data['CTimg_pred_malignant'] = predictions_ct[:,1]
    ml_data['PETimg_pred_benign'] = predictions_pet[:,0]
    ml_data['PETimg_pred_malignant'] = predictions_pet[:,1]    
    ml_data.index = the_index

    # recognize numeric columns (essential for getting the dummies later)
    for col in ml_data.columns:
        ml_data[col] = convert_to_numeric(ml_data[col])    
    
    for col in ml_data.columns:
        if ml_data[col].dtype == 'float32':
            ml_data[col] = ml_data[col].astype('float64')
        elif ml_data[col].dtype == 'uint32' or 'uint8':
            ml_data[col] = ml_data[col].astype('int64')
            
    return ml_data, labels
        
def early_stopping(modality, learning_type):
    
    if modality=='pet':
        if learning_type == '10f':
            val_acc_thres = 0.86
            train_acc_thres = 0.90
        else:
            val_acc_thres = 0.88
            train_acc_thres = 0.90   
    if modality == 'ct':
        if learning_type == '10f':
            val_acc_thres = 0.90
            train_acc_thres = 0.92
        else:
            val_acc_thres = 0.90
            train_acc_thres = 0.92            
        
    # callback = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_accuracy',  # Monitor validation accuracy
    #     patience=10,               # Number of epochs with no improvement after which training will be stopped
    #     min_delta=0.03,          # Minimum change in the monitored quantity to qualify as an improvement
    #     mode='auto',               # Mode can be 'min', 'max', or 'auto'. In 'max' mode, training will stop when the quantity monitored has stopped increasing.
    #     baseline=val_acc_thres,            # Baseline value for the monitored quantity. Training will stop if the model doesn't show improvement over the baseline.
    #     restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    # )    
    
    
    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            #print (epoch)
            
            val_acc = logs.get('val_accuracy')
            train_acc = logs.get('accuracy')
            if val_acc is not None and train_acc is not None:
                        if val_acc >= val_acc_thres and train_acc >= train_acc_thres:
                            self.model.stop_training = True
                            print('######## - STOPPED AT {} EPOCH - ########'.format(epoch))
    callback = CustomCallback() 
    
    
    return callback





'''


MODEL FIT


'''

def train_fit(data, labels, OPTIONS):
    import warnings
    warnings.filterwarnings("ignore")    
    
    
    # EARLY STOPPING
    if OPTIONS['EARLY_STOP']:
        callback = early_stopping(modality='ct', learning_type='fit')
    
    ORIGINAL_LABELS = labels
    model = spn_model_maker.selector(OPTIONS['MODEL'], OPTIONS['IN_SHAPE'], OPTIONS['TUNE'], OPTIONS['CLASSES'])
    model.summary()
    if OPTIONS['MODEL'] in ['ioapi_vit','ioapi_perceiver','ioapi_fnet','ioapi_gmlp','ioapi_mlpmixer', 
                 'ioapi_involutional','ioapi_convmixer','ioapi_big_transfer']:
        labels = labels[:,1]    
    
    if OPTIONS['AUGMENTATION']:
        print ('Activating Augmentations')
        aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=5, width_shift_range=5,
                                                              height_shift_range = 5,
                                                              #shear_range = 1.0,
                                                              horizontal_flip=True,
                                                              featurewise_center = True,
                                                              featurewise_std_normalization = True,
                                                              validation_split=0.1
                                                              )
        
        # here we use the callback to for immediate training stop on val_acc
        history = model.fit_generator(aug.flow(data, labels, batch_size=OPTIONS['BATCH_SIZE']),validation_data=aug.flow(data, labels, batch_size=8, subset='validation'), epochs=OPTIONS['EPOCHS'],callbacks=[callback], steps_per_epoch=len(data)//OPTIONS['BATCH_SIZE'])
    
    else:
        # here we use the callback to for immediate training stop on val_acc
        history = model.fit(data, labels, validation_split=0.1, epochs=OPTIONS['EPOCHS'], batch_size=OPTIONS['BATCH_SIZE'],callbacks=[callback])
    
    
    
    predictions = model.predict(data)
    predictions_binary = np.argmax(predictions, axis=-1)
    
    
    if OPTIONS['PLOTS']:

        spn_model_evaluation_plots.plot_learning_curves_manual(labels=ORIGINAL_LABELS,predictions_all_num=predictions_binary,history=history,class_names=OPTIONS['CLASS_NAMES'],model_name=OPTIONS['MODEL'])
        
        spn_model_evaluation_plots.plot_roc_scikit(all_predictions_proba=predictions,all_true_labels=ORIGINAL_LABELS)
        

    
    
    return model, predictions, predictions_binary
    

 

'''


K-fold


'''

def train_kfold(data,labels,OPTIONS):
    import warnings
    warnings.filterwarnings("ignore")    
    
    # EARLY STOPPING
    if OPTIONS['EARLY_STOP']:
        callback = early_stopping(callback = early_stopping(modality='ct', learning_type='10f'))
    
    fold_metrics = []
    predictions_total = []
    predictions_total_binary = []
    true_labels = []
    
    for train_index, test_index in KFold(OPTIONS['N_SPLIT']).split(data):
        trainX, testX = data[train_index], data[test_index]
        trainY, testY = labels[train_index], labels[test_index]
        
        
        if OPTIONS['MODEL'] in ['ioapi_vit','ioapi_perceiver','ioapi_fnet','ioapi_gmlp','ioapi_mlpmixer', 
                     'ioapi_involutional','ioapi_convmixer','ioapi_big_transfer']:
            trainY = trainY[:,1]

        model = spn_model_maker.selector(OPTIONS['MODEL'], OPTIONS['IN_SHAPE'], OPTIONS['TUNE'], OPTIONS['CLASSES'])

        if OPTIONS['AUGMENTATION']:
            print ('Activating Augmentations')
            aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, width_shift_range=4,
                                                                  height_shift_range = 4,
                                                                  shear_range = 3.0,
                                                                  horizontal_flip=True,
                                                                  featurewise_center = True,
                                                                  featurewise_std_normalization = True,
                                                                  validation_split=0.1
                                                                  )
            # here we use the callback to for immediate training stop on val_acc
            history = model.fit_generator(aug.flow(trainX, trainY, batch_size=OPTIONS['BATCH_SIZE']),validation_data=aug.flow(trainX, trainY, batch_size=8, subset='validation'), epochs=OPTIONS['EPOCHS'],callbacks=[callback], steps_per_epoch=len(trainX)//OPTIONS['BATCH_SIZE'])
        
        else:
            # here we use the callback to for immediate training stop on val_acc
            history = model.fit(trainX, trainY, validation_split=0.1, epochs=OPTIONS['EPOCHS'], batch_size=OPTIONS['BATCH_SIZE'],callbacks=[callback])
        

        predictions = model.predict(testX)
        predictions_binary = np.argmax(predictions, axis=-1)




        metrics_dict, metrics_df, metric_names, metric_values = spn_metrics.calculate_metrics(testY, predictions, predictions_binary, positive_class=1)
        fold_metrics.append(metrics_dict)
        predictions_total.append(predictions)
        predictions_total_binary.append(predictions_binary)
        true_labels.append(testY)
    
    # Concatenate the arrays along the first axis
    PREDICTIONS_TOTAL = np.concatenate(predictions_total, axis=0)
    PREDICTIONS_BINARY_TOTAL = np.concatenate(predictions_total_binary, axis=0)
    LABELS_TOTAL = np.concatenate(true_labels, axis=0)
    
    
    metrics_dict, metrics_df, metric_names, metric_values = spn_metrics.calculate_metrics(LABELS_TOTAL, PREDICTIONS_TOTAL, PREDICTIONS_BINARY_TOTAL, positive_class=1)
    

    if OPTIONS['PLOTS']:

        spn_model_evaluation_plots.plot_learning_curves_manual(labels=LABELS_TOTAL,predictions_all_num=PREDICTIONS_TOTAL,history=history,class_names=OPTIONS['CLASS_NAMES'],model_name=OPTIONS['MODEL'])
        
        spn_model_evaluation_plots.plot_roc_scikit(all_predictions_proba=PREDICTIONS_TOTAL,all_true_labels=LABELS_TOTAL)
        
  
    return history, model, PREDICTIONS_TOTAL, PREDICTIONS_BINARY_TOTAL, LABELS_TOTAL, fold_metrics, metrics_dict 