import os
import sys
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.mod_stat import *
from src.mod_velocities_geos import *


def main():
    time_min = '2023-01-15'
    time_max = '2023-12-15'
    output_dir = '../results'
    os.makedirs(output_dir, exist_ok=True)
    lambda_min = 65.
    lambda_max = 500.

    test = 'UNet_sla_real_data_training_Nadir_filt_DUACS_losses_2017-2022'

    # Mapping from test name to (method_name, dataset_path_template)
    # For tests with custom stat/psd filenames, we handle them separately
    test_config = {
        'fine_tuning': {
            'method_name': 'stat_uv_4dvar_ssh_fine_tunning',
            'stat_output': f'{output_dir}/stat_ssh_stat_uv_4dvar_ssh_fine_tunning_test_global.nc',
            'psd_output': f'{output_dir}/psd_ssh_stat_uv_4dvar_ssh_fine_tunning_test_global.nc',
            'data_path': '/Odyssey/public/glorys/rec/glorys4_global_1patch_fine_tunning_real_data_V2/processed_2023_global_4/test_data_{}.nc',
        },
        'sla': {
            'method_name': 'stat_uv_4dvar_sla',
            'stat_output': f'{output_dir}/stat_ssh_stat_uv_4dvar_sla_test_V2_global.nc',
            'psd_output': f'{output_dir}/psd_ssh_stat_uv_4dvar_sla_test_V2_global.nc',
            'data_path': '/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_not_SSH_V2/processed_2023_global_4/test_data_{}.nc',
        },
        'bathy': {
            'method_name': 'bathymetrie',
            'data_path': '/Odyssey/public/glorys/rec/glorys4_global_multivar_1patch_IN_ssh_bathy0_OUT_ssh/nrt_2023_global_4/test_data_{}_dim0.nc',
        },
        'ssh': {
            'method_name': 'stat_uv_4dvar_ssh_not_bathym',
            'stat_output': f'{output_dir}/stat_ssh_stat_uv_4dvar_ssh_not_bathym_V2_test_global.nc',
            'psd_output': f'{output_dir}/psd_ssh_stat_uv_4dvar_ssh_not_bathym_V2_test_global.nc',
            'data_path': '/Odyssey/public/glorys/rec/glorys4_global_1patch10y/nrt_2023_global_4/test_data_{}.nc',
        },
        'fine_tunning_SLA': {
            'method_name': 'stat_uv_4dvar_sla_fine_tunning',
            'stat_output': f'{output_dir}/stat_ssh_stat_uv_4dvar_sla_fine_tunning_test_global.nc',
            'psd_output': f'{output_dir}/psd_ssh_stat_uv_4dvar_sla_fine_tunning_test_global.nc',
            'data_path': '/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_fine_tunning/processed_2023_global_4/test_data_{}.nc',
        },
        'latent_dim_0': {
            'method_name': 'latent_dim_0',
            'stat_output': f'{output_dir}/stat_ssh_stat_uv_4dvar_sla_latent_dim_0_global.nc',
            'psd_output': f'{output_dir}/psd_ssh_stat_uv_4dvar_sla_latent_dim_0_global.nc',
            'data_path': '/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_one_leadtime_corrected/processed_2023_global_4/test_data_{}.nc',
        },
        'bathy_v2': {
            'method_name': 'bathy_corrected_ssh',
            'data_path': '/Odyssey/public/glorys/rec/glorys4_global_multivar_1patch_IN_ssh_bathy0_OUT_ssh/nrt_2023_global_4/test_data_{}_dim0.nc',
        },
        'UNet': {
            'method_name': 'UNet',
            'data_path': '/Odyssey/public/glorys/rec/unet_test/nrt_2023_global_4/test_data_{}.nc',
        },
        'UNet_Daria': {
            'method_name': 'UNet_Daria_MDT_MERCATOR',
            'data_path': '/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet/processed_2023_global_4/test_data_{}.nc',
        },
        'latent_dim_0_V2': {
            'method_name': 'latent_dim_0_V2_MDT_MERCATOR',
            'data_path': '/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_one_leadtime_corrected_21/processed_2023_global_4/test_data_{}.nc',
        },
        'UNet_latent_21': {
            'method_name': 'UNet_latent_dim_0_V2',
            'data_path': '/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_21/processed_2023_global_4/test_data_{}.nc',
        },
        'UNet_more_complex': {
            'method_name': 'UNet_more_complex',
            'data_path': '/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_complex/processed_2023_global_4/test_data_{}.nc',
        },
        'UNet_NRT': {
            'method_name': 'UNet_NRT_test_plot',
            'data_path': '/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet/nrt_sla/test_data_{}.nc',
        },
        '4dvarnet_sla_21_NRT': {
            'method_name': '4dvarnet_sla_21_NRT_test',
            'data_path': '/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_one_leadtime_corrected_21_NRT/nrt_sla/test_data_{}.nc',
        },
        'UNet_sla_real_data_training_Nadir_filt_DUACS_losses_2017-2022': {
            'method_name': 'UNet_sla_real_data_training_Nadir_filt_DUACS_losses_2017-2022',
            'data_path': '/Odyssey/public/glorys/rec/glorys4_global_1patch_SLA_UNet_Filtered_OSE_DUACS_losses_2017-2022/nrt_sla/test_data_{}.nc',
        },
    }

    config = test_config[test]
    method_name = config['method_name']

    if 'stat_output' in config:
        stat_output_filename = config['stat_output']
        psd_output_filename = config['psd_output']
    else:
        stat_output_filename = f'{output_dir}/stat_ssh_stat_uv_4dvar_{method_name}_global.nc'
        psd_output_filename = f'{output_dir}/psd_ssh_stat_uv_4dvar_{method_name}_global.nc'

    segment_lenght = 1000.

    for leadtime in [0]:
        one_day = False

        ds_maps = xr.open_dataset(config['data_path'].format(14 + leadtime))

        ds_maps_leadtime_i = ds_maps
        ds_maps_leadtime_i = ds_maps_leadtime_i.sel(
            time=slice(time_min, time_max)
        ).rename({'out': 'ssh', 'lat': 'latitude', 'lon': 'longitude'})

        ds_maps_leadtime_i["longitude"] = (
            (ds_maps_leadtime_i["longitude"] % 360)
            .where(ds_maps_leadtime_i["longitude"] != 360, 0)
        )
        lon_unique, index = np.unique(ds_maps_leadtime_i.coords["longitude"], return_index=True)
        ds_maps_leadtime_i = ds_maps_leadtime_i.isel(longitude=index)
        ds_maps_leadtime_i.latitude.attrs['units'] = 'degrees_north'
        ds_maps_leadtime_i.longitude.attrs['units'] = 'degrees_east'
        ds_maps_leadtime_i = ds_maps_leadtime_i.sortby(['time', 'longitude', 'latitude'])

        ds_maps_leadtime_i = retreive_geos_velocities(ds_maps_leadtime_i, "sla", "4dvar")
        ds_maps_leadtime_i.to_netcdf(
            '/Odyssey/public/glorys/rec/evaluation/'
            + method_name
            + 'all_days_GEOS_velocities_leadtime_'
            + str(leadtime)
            + '.nc'
        )
        print(f'============== Leadtime {leadtime} saved ==============')


if __name__ == '__main__':
    main()
