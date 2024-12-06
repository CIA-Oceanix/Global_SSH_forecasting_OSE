import datetime
import os

from src.ose_pipeline.ose_data_pipeline import execute_data_pipeline
from src.ose_pipeline.ose_metrics_pipeline import execute_metrics_pipeline

def setup_config(
        min_time,
        max_time,
        time_day_crop,
        ose_data_path,
        rec_path,
        metrics_path,
        xp_name,
        data_name,
):

    # time
    min_time_date = datetime.datetime.strptime(min_time, '%Y-%m-%d')
    max_time_date = datetime.datetime.strptime(max_time, '%Y-%m-%d')
    time_offset = datetime.timedelta(days=time_day_crop)
    min_time_offset = min_time_date + time_offset
    max_time_offset = max_time_date - time_offset
    min_time_offseted = min_time_offset.strftime('%Y-%m-%d')
    max_time_offseted = max_time_offset.strftime('%Y-%m-%d')
    
    # data path
    dl_sat_ref_dir = os.path.join(ose_data_path, data_name, 'dl', 'ref', '{}')
    concat_ref_path = os.path.join(ose_data_path, data_name, 'concat', 'concatenated_ref.nc')

    # metrics path
    metrics_paths = os.path.join(metrics_path, xp_name, data_name, '{}.pickle')

    return (
        dl_sat_ref_dir,
        concat_ref_path,
        metrics_paths,
        min_time_offseted,
        max_time_offseted
    )

    
def ose_results():
    pass

def execute_full_pipeline(
        min_time,
        max_time,
        min_lon,
        max_lon,
        min_lat,
        max_lat,
        metrics_spatial_domains,
        time_day_crop,
        copernicus_dataset_id,
        ref_satellites,
        ose_data_path,
        rec_path,
        metrics_path,
        xp_name,
        data_name,
        model_type,
        skip,
        overwrite,
        overrides={}
):
    (
        dl_sat_ref_dir,
        concat_ref_path,
        metrics_paths,
        min_time_offseted,
        max_time_offseted
    ) = setup_config(
        min_time,
        max_time,
        time_day_crop,
        ose_data_path,
        rec_path,
        metrics_path,
        xp_name,
        data_name
    )

    if not skip['data']:
        execute_data_pipeline(
            min_time = min_time,
            max_time = max_time,
            min_lon = min_lon,
            max_lon = max_lon,
            min_lat = min_lat,
            max_lat = max_lat,

            copernicus_dataset_id = copernicus_dataset_id,
            ref_satellites = ref_satellites,

            dl_sat_ref_dir = dl_sat_ref_dir,
            concat_ref_path = concat_ref_path,

            overwrite = overwrite['data']
        )
    else:
        print('-'*60+'\nDATA PIPELINE SKIPPED\n'+'-'*60)

    if not skip['metrics']:
        execute_metrics_pipeline(
            concat_ref_path = concat_ref_path,
            rec_path = rec_path,
            metrics_paths = metrics_paths,

            model_type=model_type,

            min_time_offseted = min_time_offseted,
            max_time_offseted = max_time_offseted,

            spatial_domains = metrics_spatial_domains,

            overwrite = overwrite['metrics']
        )
    else:
        print('-'*60+'\nMETRICS PIPELINE SKIPPED\n'+'-'*60)

