import xarray as xr
import os
from glob import glob
import datetime
import pandas as pd

MODEL_TYPES = ['4DVARNET', '4DVARNET_FAIR', 'MERCATOR_FORECAST', 'DUACS', 'GLORYS12_REANALYSIS']

def check_model_type(model_type):
    if model_type not in MODEL_TYPES:
        raise Exception('model type Not Implemented')

def get_leadtimes(model_type, rec_path):

    check_model_type(model_type)
    if model_type == '4DVARNET':
        return get_leadtimes_4dvarnet(rec_path)
    if model_type == '4DVARNET_FAIR':
        return get_leadtimes_4dvarnet(rec_path)
    if model_type == 'MERCATOR_FORECAST':
        return get_leadtimes_mercator_forecast(rec_path)
    if model_type == 'GLORYS12_REANALYSIS':
        return get_leadtimes_glorys12_reanalysis(rec_path)
    if model_type == 'DUACS':
        return get_leadtimes_duacs(rec_path)



def get_preprocessed_rec(rec_filepath, model_type, **kwargs):

    check_model_type(model_type)
    if model_type == '4DVARNET':
        return get_preprocessed_rec_4dvarnet(rec_filepath)
    if model_type == '4DVARNET_FAIR':
        return get_preprocessed_rec_4dvarnet_fair(rec_filepath)
    if model_type == 'MERCATOR_FORECAST':
        return get_preprocessed_rec_mercator_forecast(rec_filepath, **kwargs)
    if model_type == 'GLORYS12_REANALYSIS':
        return get_preprocessed_glorys12_reanalysis(rec_filepath, **kwargs)
    if model_type == 'DUACS':
        return get_preprocessed_duacs(rec_filepath, **kwargs)

# 4DVAR
def get_preprocessed_rec_4dvarnet(rec_filepath):
    return xr.open_dataset(rec_filepath)

# 4DVAR_FAIR
def get_preprocessed_rec_4dvarnet_fair(rec_filepath):
    data_4dvar_fair = xr.open_dataset(rec_filepath)
    data_4dvar_fair = data_4dvar_fair.sel(time=data_4dvar_fair.time.dt.weekday == 2)

    data_4dvar_fair["time"] = data_4dvar_fair["time"] + pd.to_timedelta(12, unit="h")
    return data_4dvar_fair

def get_leadtimes_4dvarnet(rec_path):

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

import numpy as np
# MERCATOR_FORECAST
def get_preprocessed_rec_mercator_forecast(folders_glob_pattern, leadtime_index=0):
    forecast_folders = glob(folders_glob_pattern)

    lat = np.linspace(-80, 90, 2041)
    lon = np.linspace(-180, 180, 4320)

    data_arrays = []

    for forecast_folder in forecast_folders:
        # folder name of pattern R%year%month%day
        R_folder_name = forecast_folder.split('/')[-2]
        R_year = R_folder_name[1:5]
        R_month = R_folder_name[5:7]
        R_day = R_folder_name[7:9]
        date = datetime.datetime.strptime(R_year+R_month+R_day, '%Y%m%d')
        date_plus_x = date + datetime.timedelta(days=leadtime_index, hours=12)

        R_date_plus_x = datetime.datetime.strftime(date_plus_x, '%Y%m%d')
        
        ds = xr.open_dataset(os.path.join(forecast_folder, 'glo12_rg_1d-m_'+R_date_plus_x+'-'+R_date_plus_x+'_fcst_'+R_folder_name+'.nc'))
        da = ds.isel(time=0)['zos']

        da = da.assign_coords({'latitude': lat, 'longitude': lon, 'time':pd.to_datetime(date_plus_x.strftime('%Y%m%d'), format='%Y%m%d')}).expand_dims('time')
        data_arrays.append(da)

    concat_da = xr.concat(data_arrays, dim='time').sortby('time')
    concat_ds = concat_da.to_dataset(name='zos')
    leadtime = concat_ds.rename({'zos':'out', 'latitude':'lat', 'longitude':'lon'})
    return leadtime

def get_leadtimes_mercator_forecast(rec_path):
    leadtimes = {}
    for i in range(7):
        leadtimes[i] = os.path.join(rec_path, 'R*/')
    return leadtimes

def get_leadtimes_glorys12_reanalysis(rec_path):
    return {i: rec_path for i in range(14,21)}

def get_preprocessed_glorys12_reanalysis(rec_path, **kwargs):
    time_min = kwargs.pop('time_min')
    time_max = kwargs.pop('time_max')
    date_range = pd.date_range(start=time_min, end=time_max, freq='D')
    wednesdays = date_range[date_range.weekday == 2]

    ds = xr.open_dataset(rec_path)

    monthly_avg_ds = ds.groupby("time.month").mean(dim="time")

    w_data = None

    for month in range(1, 13):
        w_for_month = wednesdays[wednesdays.month == month]
        for i in range(len(w_for_month)):
            if w_data is None:
                w_data = np.expand_dims(monthly_avg_ds["zos"].sel(month=month).values, axis=0)
            else:
                w_data = np.concatenate([w_data, np.expand_dims(monthly_avg_ds["zos"].sel(month=month).values, axis=0)], axis=0)

    ds = xr.Dataset(
        data_vars={'out': (('time', 'lat', 'lon'), w_data)},
        coords={'time': wednesdays,
                'lat': ds.latitude.values,
                'lon': ds.longitude.values}
    )
    ds["time"] = ds["time"] + pd.to_timedelta(12, unit="h")

    return ds

    return ds

# DUACS
def get_leadtimes_duacs(rec_path):
    return {i: rec_path for i in range(14,21)}

def get_preprocessed_duacs(rec_path, **kwargs):
    time_min = kwargs.pop('time_min')
    time_max = kwargs.pop('time_max')

    date_range = pd.date_range(start=time_min, end=time_max, freq='D')
    wednesdays = date_range[date_range.weekday == 2]

    rec_path_20y, rec_path_for_mdt = rec_path
    mdt_ds = xr.open_dataset(rec_path_for_mdt)
    sla_ds = xr.open_dataset(rec_path_20y)

    mdt_array = mdt_ds.isel(time=1)['adt'] - mdt_ds.isel(time=1)['sla']
    monthly_avg_ds = sla_ds.groupby("time.month").mean(dim="time")

    w_data = None

    for month in range(1, 13):
        w_for_month = wednesdays[wednesdays.month == month]
        for i in range(len(w_for_month)):
            if w_data is None:
                w_data = np.expand_dims(monthly_avg_ds["sla"].sel(month=month).values + mdt_array.values, axis=0)
            else:
                w_data = np.concatenate([w_data, np.expand_dims(monthly_avg_ds["sla"].sel(month=month).values + mdt_array.values, axis=0)], axis=0)

    ds = xr.Dataset(
        data_vars={'out': (('time', 'lat', 'lon'), w_data)},
        coords={'time': wednesdays,
                'lat': mdt_array.latitude.values,
                'lon': mdt_array.longitude.values}
    )
    ds["time"] = ds["time"] + pd.to_timedelta(12, unit="h")

    return ds