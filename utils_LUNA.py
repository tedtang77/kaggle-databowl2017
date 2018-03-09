import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_filename(case, file_list):
    '''
        Helper function to get rows in data frame associated with each file
    '''
    for f in file_list:
        if case in f:
            return(f)
        
