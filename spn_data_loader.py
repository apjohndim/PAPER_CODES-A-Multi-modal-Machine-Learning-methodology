# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:51:50 2021

@author: John
"""


''' LOAD BASIC LIBRARIRES'''

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from imutils import paths
import numpy as np
import random
import cv2
import os
from PIL import Image 
import numpy
import tensorflow as tf
import pandas as pd
from scipy.ndimage import rotate
import logging
import re
import ast
SEED = 2   # set random seed



# print = logging.info

def normalize_from_pixels (scan):
    
    MIN_BOUND = scan.min()
    MAX_BOUND = scan.max()
    
    scan = (scan - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    scan[scan>1] = 1.
    scan[scan<0] = 0.
    return scan

def rotate_img(img, angle, bg_patch=(5,5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img


def gaussian_noise(img, mean=0, sigma=0.03):
    img = img.copy()
    noise = np.random.normal(mean, sigma, img.shape)
    mask_overflow_upper = img+noise >= 1.0
    mask_overflow_lower = img+noise < 0
    noise[mask_overflow_upper] = 1.0
    noise[mask_overflow_lower] = 0
    img += noise
    return img
#imgplot = plt.imshow(img)


def make_square(img,s):
    
    s1 = max(img.shape[0:2])
    #Creating a dark square with NUMPY  
    f = np.zeros((s1,s1,3),np.uint8)
    
    #Getting the centering position
    ax,ay = (s1 - img.shape[1])//2,(s1 - img.shape[0])//2
    
    #Pasting the 'image' in a centering position
    f[ay:img.shape[0]+ay,ax:ax+img.shape[1]] = img
    f = cv2.resize(f, (s, s))
    return f


def extract_patient_id_from_image_path(image_path):
    # Extract the file name from the full path
    image_name = os.path.basename(image_path)

    # Define a regular expression pattern to extract numbers
    pattern = re.compile(r'\d+')

    # Use the pattern to find all matches in the image name
    matches = pattern.findall(image_name)

    # Check if there are any matches
    if matches:
        # Take the first match as the patient ID
        patient_id = int(matches[0])
        return patient_id
    else:
        return None


def find_image_path_by_patient_id(image_paths, target_patient_id):
    for image_path in image_paths:
        # Extract patient ID from the current image path
        current_patient_id = extract_patient_id_from_image_path(image_path)

        # Check if the current image's patient ID matches the target ID
        if current_patient_id == target_patient_id:
            return image_path

    # Return None if no matching image is found
    return None


def load_spn (path, in_shape, verbose):
    
    # Load PET and CT data
    
    petpath = 'pet'
    ctpath = 'ct'
    labelfile = 'labels_updated_dec23.xlsx'

    width = in_shape[1]
    height = in_shape[0]
    if verbose:
        print("[INFO] loading images")
        
        
    data_ct_doctor = [] # Here, data will be stored in numpy array
    data_ct_follow = [] # Here, data will be stored in numpy array
    data_ct_biopsy = [] # Here, data will be stored in numpy array
    data_pet_doctor = [] # Here, data will be stored in numpy array
    data_pet_follow = [] # Here, data will be stored in numpy array
    data_pet_biopsy = [] # Here, data will be stored in numpy array
    labels_doctor = [] # Here, the lables of each image are stored
    labels_follow = [] # Here, the lables of each image are stored
    labels_biopsy = [] # Here, the lables of each image are stored
    
    
    imagePaths_ct = sorted(list(paths.list_images(os.path.join(path,ctpath))))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths_ct) # Shuffle the image data
    
    imagePaths_pet = sorted(list(paths.list_images(os.path.join(path,petpath))))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths_pet) # Shuffle the image data    
    
    
    labels_file = pd.read_excel(os.path.join(path,labelfile))
    labels_file_np = np.array(labels_file)
    label_column_number = 10
    labelled_by_column_name = 11
    
    INFO_DOCTOR = pd.DataFrame(columns = labels_file.columns)
    INFO_FOLLOW = pd.DataFrame(columns = labels_file.columns)
    INFO_BIOPSY = pd.DataFrame(columns = labels_file.columns)
    
    # loop over the input images
    for l,imagePath_ct in enumerate(imagePaths_ct): #load, resize, normalize, etc
        ass=1
        try:

            
            patient_id = extract_patient_id_from_image_path(imagePath_ct)
            matching_image_path = find_image_path_by_patient_id(imagePaths_pet, patient_id)
            
            if verbose:
                print("---- Preparing Entry for ID = {}".format(patient_id))            
                print("Found the CT image. The corresponding PET image is with this ID: {}".format(matching_image_path))

            matching_indices = np.where(labels_file_np[:, 0] == patient_id)[0]
            label = str(labels_file_np[matching_indices,label_column_number])
            labelled_by = int(labels_file_np[matching_indices,labelled_by_column_name])
            complete_np_row = labels_file_np[matching_indices,:]
            literal_list = ast.literal_eval(label)
            label = literal_list[0]

            image_ct = cv2.imread(imagePath_ct)
            image_ct = cv2.resize(image_ct, (width,height))
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #image = np.expand_dims(image, axis=-1)
            image_ct = normalize_from_pixels (image_ct)
            
            
            image_pet = cv2.imread(matching_image_path)
            image_pet = cv2.resize(image_pet, (width,height))
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #image = np.expand_dims(image, axis=-1)
            image_pet = normalize_from_pixels (image_pet)
            
            if labelled_by == 3:
                INFO_DOCTOR.loc[len(INFO_DOCTOR)] = complete_np_row[0,:]
                data_ct_doctor.append(image_ct)
                data_pet_doctor.append(image_pet)
                labels_doctor.append(label)
            
            if labelled_by == 2:
                INFO_FOLLOW.loc[len(INFO_FOLLOW)] = complete_np_row[0,:]
                data_ct_follow.append(image_ct)
                data_pet_follow.append(image_pet)
                labels_follow.append(label)
                
            if labelled_by == 1:
                INFO_BIOPSY.loc[len(INFO_BIOPSY)] = complete_np_row[0,:]
                data_ct_biopsy.append(image_ct)
                data_pet_biopsy.append(image_pet)
                labels_biopsy.append(label)

        except Exception as e:
            print (e)
            continue
            
    data_ct_doctor = np.array(data_ct_doctor, dtype="float")
    data_ct_follow = np.array(data_ct_follow, dtype="float")
    data_ct_biopsy = np.array(data_ct_biopsy, dtype="float")
    
    data_pet_doctor = np.array(data_pet_doctor, dtype="float")
    data_pet_follow = np.array(data_pet_follow, dtype="float")
    data_pet_biopsy = np.array(data_pet_biopsy, dtype="float")    
    
    labels_doctor = np.array(labels_doctor)
    lb = LabelBinarizer()
    labels_doctor = lb.fit_transform(labels_doctor) 
    labels_doctor = tf.keras.utils.to_categorical(labels_doctor, num_classes=2)
    
    labels_follow = np.array(labels_follow)
    lb = LabelBinarizer()
    labels_follow = lb.fit_transform(labels_follow) 
    labels_follow = tf.keras.utils.to_categorical(labels_follow, num_classes=2)    

    labels_biopsy = np.array(labels_biopsy)
    lb = LabelBinarizer()
    labels_biopsy = lb.fit_transform(labels_biopsy) 
    labels_biopsy = tf.keras.utils.to_categorical(labels_biopsy, num_classes=2)    
    print("Data and labels loaded and returned")
    

    
    return data_ct_doctor,data_ct_follow,data_ct_biopsy,data_pet_doctor,data_pet_follow,data_pet_biopsy,labels_doctor,labels_follow,labels_biopsy,INFO_DOCTOR,INFO_FOLLOW,INFO_BIOPSY
