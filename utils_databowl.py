import os, sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import scipy.ndimage
import matplotlib.pyplot as plt



MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    '''
        Normalization
        Our values currently range from -1024 to around 2000. Anything above 400 is not interesting to us, as these are simply bones with different radiodensity.  A commonly used set of thresholds in the LUNA16 competition to normalize between are -1000 and 400.
        
        Tip: To save storage space, don't do normalization and zero centering beforehand, but do this online (during training, just after loading). If you don't do this yet, your image are int16's, which are smaller than float32s and easier to compress as well.
        
        More Details: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    
    '''
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


PIXEL_MEAN = 0.25

def zero_center(image):
    '''
        To zero center your data so that your mean value is 0.
        To determine this mean you simply average all images in the whole dataset.  If that sounds like a lot of work, we found this to be around 0.25 in the LUNA16 competition. 
        Warning: Do not zero center with the mean per image (like is done in some kernels on here). The CT scanners are calibrated to return accurate HU measurements. There is no such thing as an image with lower contrast or brightness like in normal pictures.
         
        Tip: To save storage space, don't do normalization and zero centering beforehand, but do this online (during training, just after loading). If you don't do this yet, your image are int16's, which are smaller than float32s and easier to compress as well.
        
        More Details: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    '''
    image = image - PIXEL_MEAN
    return image