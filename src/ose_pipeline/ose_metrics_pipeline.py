from tqdm import tqdm
import pickle
import os
from pathlib import Path

from src.ose_pipeline.metrics_utils import eval_ose, EmptyDomainException
from src.ose_pipeline.preprocess import get_leadtimes, get_preprocessed_rec

def file_exists(dir, overwrite=False):
    if os.path.exists(dir):
        print('{} already exists'.format(dir), end=', ')
        if overwrite:
            print('overwriting.')
            return False
        else:
            print('skipping.')
            return True
    return False

def domain_metrics(
        concat_ref_path,
        rec_path,
        model_type,
        metrics_paths,
        min_time_offseted,
        max_time_offseted,
        spatial_domain,
        domain_name,
        leadtimes
):
    lon_min = spatial_domain.lon.start
    lon_max = spatial_domain.lon.stop
    lat_min = spatial_domain.lat.start
    lat_max = spatial_domain.lat.stop

    is_circle = False
    if domain_name == 'GLOBE':
        is_circle=True

    centered = False
    if model_type == 'MERCATOR_FORECAST' or model_type == 'GLORYS12_REANALYSIS':
        centered = True

    RMSE_dict = dict()

    for leadtime_index, leadtime_filepath in tqdm(leadtimes.items()):
        a,b = eval_ose(
            path_alongtrack = concat_ref_path,
            rec_ds = get_preprocessed_rec(leadtime_filepath, model_type=model_type, leadtime_index=leadtime_index, time_min=min_time_offseted, time_max=max_time_offseted),
            time_min = min_time_offseted,
            time_max = max_time_offseted,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            is_circle=is_circle,
            centered=centered
        )

        tqdm.write('leadtime {} - RMSE: {:.5f} | PSD: {:.4f}'.format(leadtime_index, a, b))
        RMSE_dict[leadtime_index] = a

    with open(metrics_paths.format(domain_name+'_metrics'), mode='wb') as f:
        pickle.dump(RMSE_dict, file=f)

def execute_metrics_pipeline(
        concat_ref_path,
        rec_path,
        metrics_paths,
        model_type,
        min_time_offseted,
        max_time_offseted,
        spatial_domains,
        overwrite,

):
    print('-'*60+'\n'+'-'*60+'\nMETRICS PIPELINE START:\n')

    Path(os.path.dirname(metrics_paths)).mkdir(parents=True, exist_ok=True)

    leadtimes = get_leadtimes(model_type=model_type, rec_path=rec_path)

    for domain_name, spatial_domain in spatial_domains.items():
        if not file_exists(metrics_paths.format(domain_name+'_metrics'), overwrite):
            try:
                print('evaluating on {}'.format(domain_name))
                domain_metrics(
                    concat_ref_path,
                    rec_path,
                    model_type,
                    metrics_paths,
                    min_time_offseted,
                    max_time_offseted,
                    spatial_domain,
                    domain_name,
                    leadtimes=leadtimes
                )
                print('-'*60)
            except EmptyDomainException:
                print('domain has empty ref obs, skipping...')
    

    print('METRICS PIPELINE END:\n'+'-'*60+'\n'+'-'*60)