import os, sys
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import scipy

from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize

import scipy.ndimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def make_mask(center, diam, z, width, height, spacing, origin):
    '''
        the mask coordinates have to match the ordering of the array coordinates. 
        The x and y ordering is flipped (in (y, x) ordering)
        Args:
            Center : centers of circles px -- list of coordinates x,y,z
            diam : diameters of circles px -- diameter
            widthXheight : pixel dim of image
            spacing = mm/px conversion rate np array x,y,z
            origin = x,y,z mm np.array
            z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5]) 
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    return(mask)


def resample(image, spacing, new_spacing=[1,1,1]):
    '''
        resampling the full dataset to a certain isotropic resolution, 1mm*1mm*1mm pixels. So we can use 3D convnets without worrying about learning zoom/slice thickness invariance.
        
        A scan may have a pixel spacing of [2.5, 0.5, 0.5], which means that the distance between slices is 2.5 millimeters. For a different scan this may be [1.5, 0.725, 0.725], this can be problematic for automatic analysis (e.g. using ConvNets)! 
        
        Note: When you apply this, to save the new spacing! Due to rounding this may be slightly off from the desired spacing (above script picks the best possible spacing with rounding).
        
        More Details: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
        
        Args:
            image- indexes are z,y,x (notice the ordering)
            spacing- indexes are z,y,x (notice the ordering)
            new_spacing- indexes are z,y,x (notice the ordering)
    '''
    
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing


def matrix2int16(image):
    ''' 
        More Details: https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
        Args:
            matrix must be a numpy array NXN
        Returns: 
            int16 version matrix
    '''
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    return image

def matrix2uint16(matrix):
    ''' 
        Args:
            matrix must be a numpy array NXN
        Returns: 
            uint16 version matrix
    '''
    
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    matrix[matrix == -2000] = 0
    
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))


def plot_pixels(image, slice_idx=80):
    '''
       plot patient image pixels (before resampling)
       
       Args:
           pixels: patient image pixels
           slice_idx: the slice to show (middle slice is preferred)
       
    '''
    plt.hist(image.flatten(), bins=80, color='c')
    plt.xlabel("Hounsfield Units (HU)")
    plt.ylabel("Frequency")
    plt.show()

    # Show some slice in the middle
    plt.imshow(image[slice_idx], cmap=plt.cm.gray)
    plt.show()


MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image