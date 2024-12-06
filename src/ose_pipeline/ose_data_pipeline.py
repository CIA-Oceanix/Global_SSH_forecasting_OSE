import os

from src.ose_pipeline.data_utils import download_copernicus_data_for_sat, filt_daily_ssh_data, grid_input


def data_exists(dir, overwrite=False):
    if os.path.exists(dir):
        print('{} already exists'.format(dir), end=', ')
        if overwrite:
            print('overwriting.')
            return False
        else:
            print('skipping.')
            return True
    return False


def download_data(**kwargs):
    download_copernicus_data_for_sat(**kwargs)

def concatenate_data(**kwargs):
    filt_daily_ssh_data(**kwargs)

def grid_data(**kwargs):
    grid_input(**kwargs)

# DATA PIPELINE
def execute_data_pipeline(
        min_time,
        max_time,
        min_lon,
        max_lon,
        min_lat,
        max_lat,
        copernicus_dataset_id,
        ref_satellites,
        dl_sat_ref_dir,
        concat_ref_path,
        overwrite,
):
    print('-'*60+'\n'+'-'*60+'\nDATA PIPELINE START:\n')
            
    print('-'*60+'\n'+'downloading ref')
    for ref_sat in ref_satellites:
        if not data_exists(dl_sat_ref_dir.format(ref_sat), overwrite):
            download_data(
                    sat=ref_sat, 
                    download_dir=dl_sat_ref_dir.format(ref_sat),
                    min_time=min_time, 
                    max_time=max_time, 
                    copernicus_dataset_id=copernicus_dataset_id
                )

    print('-'*60+'\n'+'concatenate ref')
    if not data_exists(concat_ref_path, overwrite):
        concatenate_data(
            input_dir=dl_sat_ref_dir.format(''),
            output_path=concat_ref_path,
            min_time=min_time,
            max_time=max_time,
            min_lon=min_lon,
            max_lon=max_lon,
            min_lat=min_lat,
            max_lat=max_lat
        )

    print('DATA PIPELINE END:\n'+'-'*60+'\n'+'-'*60)