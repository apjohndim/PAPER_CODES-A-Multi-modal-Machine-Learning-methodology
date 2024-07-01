# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:52:08 2021

@author: John
"""


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
#in_shape = (300, 300, 3)
#classes = 2

from tensorflow.keras.layers import multiply
import numpy as np
import pydot


############################################################
########## MODEL SELECTOR #########################
############################################################

def selector (model_name,in_shape,tune,classes):
    
    # att_vgg19  lvgg    inception    vgg19_base    ffvgg19    att_ffvgg19  efficient 
    # ioapi_vit ioapi_swimtr ioapi_perceiver ioapi_involutional ioapi_convmixer
    #  ioapi_big_transfer ioapi_eanet ioapi_fnet ioapi_gmlp ioapi_mlpmixer
    # resnet mobile dense
    
    
    model = None
    if model_name == 'vgg19-final-spn':
        model = vgg19_releaseSPN(in_shape,tune,classes)
    if model_name == 'vgg19-base':
        model = furnish_base_vgg19(in_shape,tune,classes)
    if model_name == 'lvgg':
        model = furnish_ffvgg19(in_shape,tune,classes)
    if model_name == 'inception':
        model = make_inception(in_shape,tune,classes)
    if model_name == 'ffvgg19':
        model = furnish_ffvgg19(in_shape,tune,classes)
    if model_name == 'att_ffvgg19':
        model = furnish_attention_ffvgg19(in_shape,tune,classes)
    if model_name == 'efficient':
        model = make_eff(in_shape,tune,classes)       
    if model_name == 'resnet':
        model = make_resnet(in_shape,tune,classes)
    if model_name == 'mobile':
        model = make_mobile(in_shape,tune,classes)
    if model_name == 'dense':
        model = make_dense(in_shape,tune,classes)
        

    if model_name == 'ioapi_vit':
        model = ioapi_vit(in_shape,tune,classes)
    if model_name == 'ioapi_swimtr':
        model = ioapi_swimtr(in_shape,tune,classes)
    if model_name == 'ioapi_perceiver':
        model = ioapi_perceiver(in_shape,tune,classes)
    if model_name == 'ioapi_involutional':
        model = ioapi_involutional(in_shape,tune,classes)
    if model_name == 'ioapi_convmixer':
        model = ioapi_convmixer(in_shape,tune,classes)
    if model_name == 'ioapi_big_transfer':
        model = ioapi_big_transfer(in_shape,tune,classes)
    if model_name == 'ioapi_eanet':
        model = ioapi_eanet(in_shape,tune,classes)
    
    if model_name == 'ioapi_fnet':
        model = ioapi_fnet(in_shape,tune,classes)
    if model_name == 'ioapi_gmlp':
        model = ioapi_gmlp(in_shape,tune,classes)
    if model_name == 'ioapi_mlpmixer':
        model = ioapi_mlpmixer(in_shape,tune,classes)

    return model

############################################################
########## ADD THE NEW MODELS HERE #########################
############################################################







############################################################
######################### END ##############################
############################################################


def ioapi_vit (in_shape,tune,classes):
    import  fr_ioapi_vision_transformer
    model = fr_ioapi_vision_transformer.VisionTransformer(in_shape,tune,classes)
    
    return model

def ioapi_swimtr (in_shape,tune,classes):
    import  fr_ioapi_swim_transformer
    model = fr_ioapi_swim_transformer.ioapi_swim_transformer(in_shape,tune,classes)
    
    return model

def ioapi_perceiver (in_shape,tune,classes):
    import  fr_ioapi_perceiver
    model = fr_ioapi_perceiver.ioapi_perceiver(in_shape,tune,classes)
    
    return model

def ioapi_involutional (in_shape,tune,classes):
    import  fr_ioapi_involutional
    model = fr_ioapi_involutional.ioapi_involutional(in_shape,tune,classes)
    
    return model

def ioapi_convmixer (in_shape,tune,classes):
    import  fr_ioapi_convmixer
    model = fr_ioapi_convmixer.ioapi_convmixer(in_shape,tune,classes)
    
    return model

def ioapi_big_transfer (in_shape,tune,classes):
    import  fr_ioapi_big_transfer
    model = fr_ioapi_big_transfer.ioapi_big_transfer(in_shape,tune,classes)
    
    return model

def ioapi_eanet (in_shape,tune,classes):
    import  fr_ioapi_eanet
    model = fr_ioapi_eanet.ioapi_eanet(in_shape,tune,classes)
    
    return model


def ioapi_fnet (in_shape,tune,classes):
    import  fr_ioapi_fnet
    model = fr_ioapi_fnet.ioapi_fnet(in_shape,tune,classes)
    
    return model

def ioapi_gmlp (in_shape,tune,classes):
    import  fr_ioapi_gmlp
    model = fr_ioapi_gmlp.ioapi_gmlp(in_shape,tune,classes)
    
    return model

def ioapi_mlpmixer (in_shape,tune,classes):
    import  fr_ioapi_mlpmixer
    model = fr_ioapi_mlpmixer.ioapi_mlpmixer(in_shape,tune,classes)
    
    return model


def furnish_base_vgg19 (in_shape, tune='auto', classes=2): #tune = 0 is off the self, tune = 1 is scratch, auto is last conv trainable
    
    ###########################################################################
    #
    # The baseline VGG19 with option for fine-tune and classic Dense layers with Global Average Pooling
    #
    ###########################################################################
    

    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune == 20 or tune=='auto':   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[19:]:
            layer.trainable = True
    #base_model.summary()
    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    x = tf.keras.layers.Dense(400, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    
    print("[INFO] VGG19 Baseline Model Compiled!")
    return model


def vgg19_releaseSPN(in_shape,tune,classes):
    
    ###########################################################################
    #
    # The baseline Feature Fusion VGG19 - THE ONE THAT GIVES THE BEST IN EXCEL
    #
    ###########################################################################
    
    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])

    for layer in base_model.layers:
        layer.trainable = False
        

    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)

    x = tf.keras.layers.Dense(1200, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("[INFO] Model Compiled!")
    return model




def furnish_ffvgg19 (in_shape, tune, classes):
    
    ###########################################################################
    #
    # The baseline Feature Fusion VGG19
    #
    ###########################################################################
    
    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', 
                                                   input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[19:]:
        layer.trainable = False
    #base_model.summary()
    
    # early2 = layer_dict['block2_pool'].output 
    # #early2 = Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early2)
    # early2 = tf.keras.layers.BatchNormalization()(early2)
    # early2 = tf.keras.layers.Dropout(0.5)(early2)
    # early2= tf.keras.layers.GlobalAveragePooling2D()(early2)
        
    early3 = layer_dict['block3_pool'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3 = tf.keras.layers.BatchNormalization()(early3)
    early3 = tf.keras.layers.Dropout(0.5)(early3)
    early3= tf.keras.layers.GlobalAveragePooling2D()(early3)    
        
    early4 = layer_dict['block4_pool'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4 = tf.keras.layers.BatchNormalization()(early4)
    early4 = tf.keras.layers.Dropout(0.5)(early4)
    early4= tf.keras.layers.GlobalAveragePooling2D()(early4)     
    
    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1)
    x = tf.keras.layers.concatenate([x1, early4, early3], axis=-1)  
    
    x = tf.keras.layers.Dense(1200, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='ffvgg19-base.png')
    print("[INFO] Model Compiled!")
    return model



def furnish_ffvgg19_lstm(in_shape, tune, classes):
    
    ###########################################################################
    #
    # The baseline Feature Fusion VGG19 with LSTM units at the top
    #
    ###########################################################################
    
    '''FFVGG19'''
    def ffvgg19(in_shape, tune, classes):
        base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', 
                                                       input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
        layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
        #base_model.summary()
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[19:]:
            layer.trainable = False
      
        early3 = layer_dict['block3_pool'].output   
        early3 = tf.keras.layers.BatchNormalization()(early3)
        early3 = tf.keras.layers.Dropout(0.5)(early3)
        early3= tf.keras.layers.GlobalAveragePooling2D()(early3)    
            
        early4 = layer_dict['block4_pool'].output   
        early4 = tf.keras.layers.BatchNormalization()(early4)
        early4 = tf.keras.layers.Dropout(0.5)(early4)
        early4= tf.keras.layers.GlobalAveragePooling2D()(early4)     
        
        x1 = layer_dict['block5_conv3'].output 
        x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
        x = tf.keras.layers.concatenate([x1, early4, early3], axis=-1)
        exodus = tf.keras.layers.Dense(500, activation="relu")(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=exodus)
        
        return model
    
    
    cnn_net = ffvgg19(in_shape, tune, classes)
    input_layer = tf.keras.layers.Input(shape=(3, in_shape[0], in_shape[1], in_shape[2]))
    lstm_ip_layer = tf.keras.layers.TimeDistributed(cnn_net)(input_layer)
    
    x = tf.keras.layers.LSTM(500)(lstm_ip_layer)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.LSTM(200)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.LSTM(50)(x)
    output = tf.keras.layers.Dense(units=2,activation='softmax')(x)
    model = tf.keras.Model([input_layer],output)

    optimizer = 'Adam'
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def furnish_attention_ffvgg19 (in_shape, tune, classes):
    
    ###########################################################################
    #
    # The baseline Feature Fusion VGG19 with attention blocks and BD paths
    #
    ###########################################################################
    
    from tensorflow.keras.utils import plot_model
    def pay_attention(m_input,filters, name):
      # Based on https://www.kaggle.com/kmader/attention-on-pretrained-vgg16-for-bone-age#Attention-Model
      pt_depth = filters
      bn_features = BatchNormalization()(m_input)

      attn = Conv2D(64, kernel_size=(1,1), padding='same', activation='relu')(bn_features)
      attn = Conv2D(16, kernel_size=(1,1), padding='same', activation='relu')(attn)
      attn = Conv2D(1, 
                    kernel_size=(1,1), 
                    padding='valid', 
                    activation='sigmoid',
                    name=name)(attn)
      up_c2_w = np.ones((1, 1, 1, pt_depth))
      up_c2 = Conv2D(pt_depth,
                     kernel_size=(1,1),
                     padding='same', 
                     activation='linear',
                     use_bias=False,
                     weights=[up_c2_w])
      up_c2.trainable = False
      attn = up_c2(attn)

      mask_features = multiply([attn, bn_features])
      gap_features = GlobalAveragePooling2D()(mask_features)
      gap_mask = GlobalAveragePooling2D()(attn)
      attn_gap = Lambda(lambda x: x[0]/x[1])([gap_features, gap_mask])

      return attn_gap
  
    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', 
                                                   input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[19:]:
        layer.trainable = True
  
    f3 = layer_dict['block3_pool'].output   
    f3 = tf.keras.layers.BatchNormalization()(f3)
    f3 = tf.keras.layers.Dropout(0.5)(f3)
    f3= tf.keras.layers.GlobalAveragePooling2D()(f3)   
    
    f3b = layer_dict['block3_pool'].output
    f3b = pay_attention(f3b,256, 'Att_b3p')   
    
    
    f4 = layer_dict['block4_pool'].output   
    f4 = tf.keras.layers.BatchNormalization()(f4)
    f4 = tf.keras.layers.Dropout(0.5)(f4)
    f4= tf.keras.layers.GlobalAveragePooling2D()(f4)     
    
    f4b = layer_dict['block4_pool'].output
    f4b = pay_attention(f4b,512, 'Att_b4p')          
    
    
    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    x = tf.keras.layers.concatenate([x1, f3, f3b,f4, f4b], axis=-1)
          
    
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.5)(x)

    
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])  
    plot_model(model, to_file='ffvgg19-attent.png')
    return model
    

    
def furnish_attention_vgg19 (in_shape, tune, classes):
    
    ###########################################################################
    #
    # The baseline VGG19 with attention blocks
    #
    ###########################################################################
    
    from tensorflow.keras.utils import plot_model
    def pay_attention(m_input,filters, name):
      # Based on https://www.kaggle.com/kmader/attention-on-pretrained-vgg16-for-bone-age#Attention-Model
      pt_depth = filters
      bn_features = BatchNormalization()(m_input)

      attn = Conv2D(64, kernel_size=(1,1), padding='same', activation='relu')(bn_features)
      attn = Conv2D(16, kernel_size=(1,1), padding='same', activation='relu')(attn)
      attn = Conv2D(1, 
                    kernel_size=(1,1), 
                    padding='valid', 
                    activation='sigmoid',
                    name=name)(attn)
      up_c2_w = np.ones((1, 1, 1, pt_depth))
      up_c2 = Conv2D(pt_depth,
                     kernel_size=(1,1),
                     padding='same', 
                     activation='linear',
                     use_bias=False,
                     weights=[up_c2_w])
      up_c2.trainable = False
      attn = up_c2(attn)

      mask_features = multiply([attn, bn_features])
      gap_features = GlobalAveragePooling2D()(mask_features)
      gap_mask = GlobalAveragePooling2D()(attn)
      attn_gap = Lambda(lambda x: x[0]/x[1])([gap_features, gap_mask])

      return attn_gap
  

    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', 
                                                   input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[20:]:
        layer.trainable = True
  
    # f1 = layer_dict['block1_pool'].output
    # f1 = pay_attention(f1,64, 'Att_b1p')   
    
    
    # f2 = layer_dict['block2_pool'].output
    # f2 = pay_attention(f2,128, 'Att_b2p')         
  
  
    f3 = layer_dict['block3_pool'].output
    f3 = pay_attention(f3,256, 'Att_b3p')   
    
    
    f4 = layer_dict['block4_pool'].output
    f4 = pay_attention(f4,512, 'Att_b4p')          
    
    
    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    x = tf.keras.layers.concatenate([x1,f3,f4], axis=-1)
          
    
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])  
    plot_model(model, to_file='vgg19-attention.png')
    return model


def make_vgg (in_shape, tune, classes): #tune = 0 is off the self, tune = 1 is scratch, tune 
    
#import pydot
    
    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    

    
    for layer in base_model.layers:
        layer.trainable = False
        

    #base_model.summary()
  
    
    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(1200, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model

def make_lvgg (in_shape, tune, classes):
    
#import pydot
    
    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[19:]:
        layer.trainable = True
    #base_model.summary()
    
    # early2 = layer_dict['block2_pool'].output 
    # #early2 = Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early2)
    # early2 = tf.keras.layers.BatchNormalization()(early2)
    # early2 = tf.keras.layers.Dropout(0.5)(early2)
    # early2= tf.keras.layers.GlobalAveragePooling2D()(early2)
        
    early3 = layer_dict['block3_conv4'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3 = tf.keras.layers.BatchNormalization()(early3)
    #early3 = tf.keras.layers.Dropout(0.2)(early3)
    early3= tf.keras.layers.GlobalAveragePooling2D()(early3)    
        
    early4 = layer_dict['block4_conv4'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4 = tf.keras.layers.BatchNormalization()(early4)
    #early4 = tf.keras.layers.Dropout(0.2)(early4)
    early4= tf.keras.layers.GlobalAveragePooling2D()(early4)     
    
    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1)
    x = tf.keras.layers.concatenate([x1, early4, early3], axis=-1)  
    
    #x = Flatten()(x)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(1400, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    #model.summary()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model

def make_xception (in_shape, tune, classes):
    
    base_model = tf.keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    

    for layer in base_model.layers:
        layer.trainable = False
        
    x1 = layer_dict['block14_sepconv2'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    x = tf.keras.layers.Dense(1200, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model 

def make_inception (in_shape, tune, classes):
    
    base_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    #for layer in base_model.layers:
        #print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])


    for layer in base_model.layers:
        layer.trainable = False
    


    
    x1 = base_model.output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    x = tf.keras.layers.Dense(1500, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    #print("[INFO] Model Compiled!")
    return model 

def make_resnet (in_shape, tune, classes):
    
    base_model = tf.keras.applications.ResNet152V2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    for layer in base_model.layers:
        print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    

    for layer in base_model.layers:
        layer.trainable = False
        

    
    x1 = layer_dict['post_relu'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(1200, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model 


def make_mobile (in_shape, tune, classes):
    
    base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    # for layer in base_model.layers:
    #     print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    

    

    for layer in base_model.layers:
        layer.trainable = False

    
    x1 = layer_dict['out_relu'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    x = tf.keras.layers.Dense(1200, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("[INFO] Model Compiled!")
    return model 


def make_dense (in_shape, tune, classes):
    
    base_model = tf.keras.applications.DenseNet201(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    for layer in base_model.layers:
        print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])

    
    for layer in base_model.layers:
        layer.trainable = False
        
    
    x1 = layer_dict['relu'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    x = tf.keras.layers.Dense(1200, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("[INFO] Model Compiled!")
    return model 

def make_eff (in_shape, tune, classes):
    
    base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    for layer in base_model.layers:
        print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    
    for layer in base_model.layers:
        layer.trainable = False
    
    x1 = layer_dict['relu'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(1200, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model
