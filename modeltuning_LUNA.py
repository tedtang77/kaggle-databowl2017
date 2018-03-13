import os, sys, math
import numpy as np
import matplotlib.pyplot as plt

import csv
from glob import glob
import pandas as pd
from tqdm import tqdm

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_image_data_format("channels_last")

SMOOTH = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet(img_size=(512, 512)):
    #rows, cols = img_size
    inputs = Input(img_size+(1,)) #Channels Last
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    
    conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    
    conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
    
    conv4 = Conv2D(256, (3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)
    
    conv5 = Conv2D(512, (3,3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3,3), activation='relu', padding='same')(conv5)
   
    up6 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(256, (3,3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3,3), activation='relu', padding='same')(conv6)
    
    up7 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3,3), activation='relu', padding='same')(conv7)
    
    up8 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3,3), activation='relu', padding='same')(conv8)
    
    up9 = Concatenate(axis=-1)([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3,3), activation='relu', padding='same')(conv9)
    
    conv10 = Conv2D(1, (1,1), activation='sigmoid')(conv9)
    
    model = Model(inputs, conv10)
    
    model.compile(Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    
    return model

