import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import bcolz


def get_filename(case, file_list):
    '''
        Helper function to get rows in data frame associated with each file
    '''
    for f in file_list:
        if case in f:
            return(f)
    
    
def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()
    
    
def load_array(fname):
    return np.array(bcolz.open(rootdir=fname))