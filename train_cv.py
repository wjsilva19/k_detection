#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 19:20:07 2018

@author: wilson
"""

import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import matplotlib.pyplot as plt

import _pickle as cPickle
import numpy as np 
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,Conv2DTranspose, multiply, concatenate,\
Dense, Flatten, Dropout, Lambda
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split, KFold
from keras import losses
from train_generator import Generator 
import cv2
from keras import backend as K
#import pickle



""" Load all files """
with open("/data/wjsilva19/keypoints/Paper_train_test_divisions/X_train_221.pickle",'rb') as fp: 
        X_train = cPickle.load(fp)  

with open("/data/wjsilva19/keypoints/Paper_train_test_divisions/X_test_221.pickle",'rb') as fp: 
        X_test = cPickle.load(fp)
        
X = np.concatenate((X_train, X_test))        
        
with open("/data/wjsilva19/keypoints/Paper_train_test_divisions/y_train_221.pickle",'rb') as fp: 
        y_train = cPickle.load(fp)  

with open("/data/wjsilva19/keypoints/Paper_train_test_divisions/y_test_221.pickle",'rb') as fp: 
        y_test = cPickle.load(fp)  

y = np.concatenate((y_train, y_test))

with open("/data/wjsilva19/keypoints/Paper_train_test_divisions/heatmaps_train_221.pickle",'rb') as fp: 
        map_probs_train = cPickle.load(fp)  


with open("/data/wjsilva19/keypoints/Paper_train_test_divisions/heatmaps_test_221.pickle",'rb') as fp: 
        map_probs_test = cPickle.load(fp)  

map_probs = np.concatenate((map_probs_train, map_probs_test))


print(np.shape(X), 'shape of X')

""" Convert lists to numpy arrays """
X = np.array(X, dtype='float')
y = np.array(y)
map_probs = np.array(map_probs)


""" Normalize the keypoints by the width of the image """
y /= 512

""" Reshape y """
map_probs = np.reshape(map_probs,(-1,384,512,1))


""" Prepare the image for the VGG model """
X_train_processed = np.array(X)
from keras.applications.vgg16 import preprocess_input
X_train_processed = preprocess_input(X_train_processed)

print(np.max(X_train_processed),'maximum of X_train')
print(np.min(X_train_processed),'minimum of X_train')


#Indices to realize data augmentation by horizontal flips
flip_indices = [(0,34),(1,35),(2,36),(3,37),(4,38),(5,39),(6,40),(7,41),\
(8,42),(9,43),(10,44),(11,45),(12,46),(13,47),(14,48),(15,49),(16,50),(17,51),\
(18,52),(19,53),(20,54),(21,55),(22,56),(23,57),(24,58),(25,59),(26,60),(27,61),(28,62),(29,63),(30,64),(31,65),\
(32,66),(33,67),(68,68),(69,69),(70,72),(71,73)]

skf = KFold(n_splits=5, shuffle=True, random_state=42)

total_preds = [] 

for train_indices, test_indices in skf.split(X_train_processed): 
    K.clear_session()
    conv_base = VGG16(weights = 'imagenet',
                  include_top = False,
                  input_shape = (384,512,3))


    def u_net(inputs): 
        u_net_input = Lambda(lambda inputs:(inputs/255.0))(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(u_net_input)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        
        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
        
        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
        
        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
        
        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
        
        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    
    
        return conv10
    
    def feature_extraction(inputs):
        conv1 = conv_base(inputs)
        conv1 = Conv2D(512, (3, 3), activation='relu', padding='valid')(conv1)
        conv1 = Conv2D(512, (3, 3), activation='relu', padding='valid')(conv1)
        conv1 = Conv2D(512, (3, 3), activation='relu', padding='valid')(conv1)
        conv1 = Conv2D(512, (1, 1), activation='relu', padding='valid')(conv1)
        flat = Flatten()(conv1)
        dense1 = Dense(256,activation='relu')(flat)
        dense1 = Dropout(0.2)(dense1)
        dense1 = Dense(128,activation='relu')(dense1)
        reg = Dense(74,activation='sigmoid', name = 'keypoints')(dense1)
        
        return reg
    
    rows = np.shape(X)[1]
    cols = np.shape(X)[2]
    
    """Input is an image with 3 colour channels"""
    inputs = Input((rows, cols, 3))
    
    """ First step is to obtain probability maps"""
    stage1 = u_net(inputs)
    
    """Concatenate probability maps in order to have an image with the same number of channels as input image"""
    stage1_concat = concatenate([stage1,stage1,stage1])
    
    """Multiplication between prob maps and input image, to select region of interest"""
    stage2_in = multiply([stage1_concat,inputs])
    
    stage2 = u_net(stage2_in)
    
    stage2_concat = concatenate([stage2,stage2,stage2])
    
    stage3_in = multiply([stage2_concat,inputs])
    
    stage3 = u_net(stage3_in)
    
    stage3_concat = concatenate([stage3, stage3, stage3])
    
    stage4_in = multiply([stage3_concat, inputs])
    
    """Perform regression"""
    stage4 = feature_extraction(stage4_in)
    
    conv_base.trainable=True 
    
    set_trainable = False
    for layer in conv_base.layers: 
        if layer.name=='block5_conv1' or layer.name=='block5_conv2' or\
        layer.name=='block5_conv3' or layer.name=='block4_conv1' or\
        layer.name=='block4_conv2' or layer.name=='block4_conv3':
            set_trainable = True
        if set_trainable == True: 
            layer.trainable = True
        else:
            layer.trainable = False
    
    """Model has one input: image; and two outputs: probability maps and keypoints"""
    model = Model(inputs=[inputs], outputs=[stage1, stage2, stage3, stage4])

    model.compile(optimizer='adadelta', loss = [losses.mean_squared_error, losses.mean_squared_error, \
                                            losses.mean_squared_error, losses.mean_squared_error], \
    loss_weights= [1, 2, 4, 10])
    
    X_train, X_test = X_train_processed[train_indices], X_train_processed[test_indices]
    map_probs_train, map_probs_test = map_probs[train_indices], map_probs[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    

    """Apply generator"""
    my_generator = Generator(X_train,
                         map_probs_train,
                         y_train,
                         batchsize=2,
                         flip_ratio=0.5,
                         translation_ratio=0.5,
                         rotate_ratio = 0.5,
                         flip_indices=flip_indices
                         )


    model.fit_generator(my_generator.generate(),steps_per_epoch = my_generator.size_train/2, epochs=250, verbose=0)     

    preds = model.predict(X_test)
    
    keypoints = preds[3]
        
    total_preds.append(keypoints)
    

with open("total_preds_221.pickle","wb") as output_file:
    cPickle.dump(total_preds, output_file)


print(total_preds)    





      
    

