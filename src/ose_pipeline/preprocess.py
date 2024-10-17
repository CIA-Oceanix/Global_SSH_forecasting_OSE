import pytorch_lightning
from omegaconf import OmegaConf
import hydra
import xarray as xr
import numpy as np
import pyinterp
import os
import logging
import netCDF4
import scipy.signal
from scipy import interpolate
import matplotlib.pylab as plt
from glob import glob

MODEL_TYPES = ['4DVARNET']

def check_model_type(model_type):
    if model_type not in MODEL_TYPES:
        raise Exception('model type Not Implemented')

def get_leadtimes(model_type, rec_path):

    name_sep = ''
    check_model_type(model_type)
    if model_type == '4DVARNET':
        name_sep = 'test_data_'

    leadtime_filepaths = glob(os.path.join(rec_path, name_sep+'*'))
    
    leadtimes = {}
    for leadtime_filepath in leadtime_filepaths:
        leadtimes[int(leadtime_filepath.split(name_sep)[1].split('.')[0])] = leadtime_filepath
    indices = list(leadtimes.keys())
    indices.sort()

    prev_index = None
    for index in indices:
        if prev_index is not None:
            if prev_index+1 != index:
                raise Exception('leadtimes are not properly ordered')
        prev_index = index

    return leadtimes

def get_preprocessed_rec_4dvarnet(rec_filepath):
    print(rec_filepath)
    return xr.open_dataset(rec_filepath)

def get_preprocessed_rec(rec_filepath, model_type):

    check_model_type(model_type)
    if model_type == '4DVARNET':
        return get_preprocessed_rec_4dvarnet(rec_filepath)



