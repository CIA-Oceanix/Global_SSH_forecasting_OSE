import xarray as xr
import numpy as np
import os
from datetime import timedelta
import pandas as pd
from parcels import FieldSet, ParticleSet, JITParticle, Variable, AdvectionRK4


import os
import numpy as np
import pandas as pd
from datetime import timedelta
from parcels import FieldSet, ParticleSet, JITParticle, Variable, AdvectionRK4

def advection(ds, model_name):
    '''
    Perform advection only if output file does not already exist.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing drifter trajectories.
    model_name : str
        Name of the velocity field model (e.g., '4dvarnet_sla_21_NRT_test').
    '''
    n_traj = ds.dims['trajectory']
    n_time = ds.dims['TIME']

    lons = ds['LONGITUDE'].values
    lats = ds['LATITUDE'].values
    times = ds['TIME'].values

    orig_ids = np.arange(n_traj)
    print(orig_ids)

    init_lons = np.full(n_traj, np.nan)
    init_lats = np.full(n_traj, np.nan)
    init_times = np.full(n_traj, np.nan)

    for traj_idx in range(n_traj):
        valid_indices = np.where(~np.isnan(lons[traj_idx, :]) & ~np.isnan(lats[traj_idx, :]))[0]
        if valid_indices.size > 0:
            first_idx = valid_indices[0]
            init_lons[traj_idx] = lons[traj_idx, first_idx]
            init_lats[traj_idx] = lats[traj_idx, first_idx]
            init_times[traj_idx] = times[first_idx]

    init_times_py = [pd.to_datetime(t).to_pydatetime() if not pd.isna(t) else None for t in init_times]
    valid = ~np.isnan(init_lons) & ~np.isnan(init_lats) & np.array([t is not None for t in init_times_py])

    init_lons = init_lons[valid]
    init_lats = init_lats[valid]
    init_times_py = [t for v, t in zip(valid, init_times_py) if v]
    orig_ids = orig_ids[valid]

    print(f"Number of valid trajectories: {len(init_lons)}")

    for leadtime in [0, 3, 5]:
        output_dir = f'./output_lagrangian/{model_name}/'

        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        #output_file = os.path.join(output_dir, f'drifter_trajectories_with_ids_leadtime_{leadtime}.zarr')

        output_file = f'./output_lagrangian/' + model_name + '/drifter_trajectories_with_ids_leadtime_{leadtime}.zarr'

        if os.path.exists(output_file):
            print(f"Output already exists for leadtime {leadtime}, skipping...")
            continue

        data_folder = f'/Odyssey/public/glorys/rec/evaluation/{model_name}/daily_leadtime_/{leadtime}/'
        file_list = sorted([os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.nc')])

        variables = {'U': 'ugos', 'V': 'vgos'}
        dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
        fieldset = FieldSet.from_netcdf(file_list, variables, dimensions, allow_time_extrapolation=True)

        class DrifterParticle(JITParticle):
            id_traj_drifter = Variable('id_traj_drifter', dtype=np.int32)

        pset = ParticleSet(fieldset=fieldset,
                           pclass=DrifterParticle,
                           lon=init_lons,
                           lat=init_lats,
                           time=init_times_py,
                           id_traj_drifter=orig_ids)

        output = pset.ParticleFile(name=output_file, outputdt=timedelta(hours=6))

        print(f'=============== Leadtime {leadtime} ===============')
        pset.execute(AdvectionRK4,
                     runtime=timedelta(days=10),
                     dt=timedelta(minutes=30),
                     output_file=output)

import os
import pandas as pd
from collections import defaultdict  # Import defaultdict

def gather_and_sort_files(base_path, model_name):
    """
    Gathers and sorts all NetCDF forecast files across leadtimes.
    """
    all_files = []  # List to store all files from all leadtimes

    # Loop through all leadtime folders (0, 1, 2, ..., 6)
    for leadtime in range(7):
        folder = os.path.join(base_path, model_name, 'daily_leadtime_', str(leadtime))
        if not os.path.exists(folder):
            print(f"⚠️ Missing folder: {folder}")
            continue
        
        for fname in os.listdir(folder):
            if fname.endswith('.nc'):  # Only consider .nc files
                date_str = fname.split('_')[-1].replace('.nc', '')  # Extract date part from filename
                try:
                    base_date = pd.to_datetime(date_str)  # Convert to datetime object
                except ValueError:
                    print(f"⚠️ Skipping file with invalid date format: {fname}")
                    continue
                
                file_path = os.path.join(folder, fname)
                all_files.append((base_date, file_path))  # Store date and file path as tuple

    # Sort files by date (ascending order)
    all_files.sort(key=lambda x: x[0])  # Sort by base_date (first element of tuple)

    # Extract sorted file paths
    sorted_files = [file[1] for file in all_files]
    
    return sorted_files

def advection_7_leadtimes(ds, model_name):
    '''
    Perform 7-day advection, launching a new forecast every 7 days
    for each valid trajectory in the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing drifter trajectories with dimensions:
        - 'trajectory', 'TIME'
    model_name : str
        Name of the velocity field model (e.g., '4dvarnet_sla_21_NRT_test')
    '''

    n_traj = ds.dims['trajectory']
    n_time = ds.dims['TIME']

    
    lons = ds['LONGITUDE'].values
    lats = ds['LATITUDE'].values
    times = ds['TIME'].values
    trajs = ds['trajectory'].values

    init_lons = []
    init_lats = []
    init_times_py = []
    orig_ids = []

    for traj_idx in range(n_traj):
        lons_traj = lons[traj_idx]
        lats_traj = lats[traj_idx]
        traj_traj = trajs[traj_idx]
        
        # Estimate step interval (~7 days)
        time_diffs = np.diff(times).astype('timedelta64[s]').astype(float)
        if len(time_diffs) == 0 or np.median(time_diffs) == 0:
            continue

        dt_seconds = np.median(time_diffs)
        step_interval = int(np.round((7 * 86400) / dt_seconds))

        #valid_mask = ~np.isnan(lons_traj) & ~np.isnan(lats_traj)
        #valid_indices = np.where(valid_mask)[0]

        #if valid_indices.size < 1:
        #    continue

        # Loop over initial times spaced by ~7 days
        for k, start_idx in enumerate(np.arange(len(lons_traj))[::24*7]):  #(valid_indices):
            if start_idx + step_interval >= len(times):
                continue  # Not enough future time for 7-day advection
            if(not np.isnan(lons_traj[start_idx])):
                init_lons.append(lons_traj[start_idx])
                init_lats.append(lats_traj[start_idx])
                init_times_py.append(pd.to_datetime(times[start_idx]).to_pydatetime())
                #orig_ids.append(tra
                # j_idx)
                orig_ids.append(str(traj_traj) + '_' + str(start_idx))  # assign unique ID per launch

    print(f"Number of initial points: {len(init_lons)}")
    
    # Sample list of string-based identifiers
    trajectory_strings = orig_ids

    # Create a dictionary to map each unique string to a unique integer
    string_to_int_map = {s: i for i, s in enumerate(set(trajectory_strings))}

    # Convert the original list to a list of integer identifiers
    int_ids = [string_to_int_map[s] for s in trajectory_strings]

    # Step 2: Integer → String
    int_to_string_map = {v: k for k, v in string_to_int_map.items()}
    recovered_strings = [int_to_string_map[i] for i in int_ids]


    if len(init_lons) == 0:
        print("No valid initial points found, exiting.")
        return

    output_dir = f'./output_lagrangian/{model_name}/'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'drifter_trajectories_leadtime7.zarr')
    if os.path.exists(output_file):
        print("Output already exists, skipping...")
        return

    # Load velocity field from model data
    file_list = gather_and_sort_files('/Odyssey/public/glorys/rec/evaluation', model_name)

    variables = {'U': 'ugos', 'V': 'vgos'}
    dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    fieldset = FieldSet.from_netcdf(file_list, variables, dimensions, allow_time_extrapolation=True)

    class DrifterParticle(JITParticle):
        id_traj_drifter = Variable('id_traj_drifter')

    particle_uids = list(range(len(init_lons)))  # one unique ID per launch
    
    print('init_times_py')
    print(init_times_py)
    
    init_lons = np.array(init_lons)
    print("any nan in init_lons ? ")
    print(np.isnan(init_lons))
    init_lats = np.array(init_lats)
    init_times_py = np.array(init_times_py)

    valid_mask = ~np.isnan(init_lons) & ~np.isnan(init_lats)

    print(f"Total launches prepared: {len(init_lons)}")
    
    launch_times = np.array(init_times_py)
    launch_min = launch_times.min()
    launch_max = launch_times.max()

    print("Launch time range:")
    print("Start:", launch_min)
    print("End:", launch_max)
    
    print('orig_ids[valid_mask]')
    print(np.array(orig_ids)[valid_mask])
    pset = ParticleSet(fieldset=fieldset,
                       pclass=DrifterParticle,
                       lon=init_lons,
                        lat=init_lats,
                        time=init_times_py,
                        id_traj_drifter=np.array(int_ids) #np.arange(np.sum(valid_mask))  # unique IDs
                        )
    
    print("Number of particles to simulate:", len(pset))

    output = pset.ParticleFile(name=output_file, outputdt=timedelta(hours=1))

    print("============== Launching 7-day advection ==============")
    pset.execute(AdvectionRK4,
                 runtime=timedelta(days=7),
                 dt=timedelta(minutes=30),
                 output_file=output)
    
    return recovered_strings, int_to_string_map


import os
import numpy as np
import pandas as pd
from datetime import timedelta
from parcels import FieldSet, ParticleSet, JITParticle, Variable, AdvectionRK4
import xarray as xr

def advection_7_leadtimes_save_orig_traj(ds, model_name):
    '''
    Perform 7-day advection, launching a new forecast every 7 days
    for each valid trajectory in the dataset, and store the original and simulated trajectories together.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing drifter trajectories with dimensions:
        - 'trajectory', 'TIME'
    model_name : str
        Name of the velocity field model (e.g., '4dvarnet_sla_21_NRT_test')
    '''

    n_traj = ds.dims['trajectory']
    n_time = ds.dims['TIME']

    lons = ds['LONGITUDE'].values
    lats = ds['LATITUDE'].values
    times = ds['TIME'].values
    trajs = ds['trajectory'].values

    init_lons = []
    init_lats = []
    init_times_py = []
    orig_ids = []

    # Lists to store original and simulated trajectories
    orig_lons = []
    orig_lats = []
    sim_lons = []
    sim_lats = []
    
    idx_kept_by_parcels_list = []
    idx_kept_by_parcels = 0
    
    for traj_idx in range(n_traj):
        lons_traj = lons[traj_idx]
        lats_traj = lats[traj_idx]
        traj_traj = trajs[traj_idx]

        # Estimate time step interval (~7 days)
        time_diffs = np.diff(times).astype('timedelta64[s]').astype(float)
        if len(time_diffs) == 0 or np.median(time_diffs) == 0:
            continue

        dt_seconds = np.median(time_diffs)
        step_interval = int(np.round((7 * 86400) / dt_seconds))

        # Loop over initial times spaced by ~7 days
        for k, start_idx in enumerate(np.arange(len(lons_traj))[::24*7]):
            if start_idx + step_interval >= len(times):
                continue  # Not enough future time for 7-day advection
            if not np.isnan(lons_traj[start_idx]):
                init_lons.append(lons_traj[start_idx])
                init_lats.append(lats_traj[start_idx])
                init_times_py.append(pd.to_datetime(times[start_idx]).to_pydatetime())
                orig_ids.append(f"{traj_traj}_{start_idx}")  # Unique ID per launch
                #print('lons_traj[start_idx]')
                #print(lons_traj[start_idx])
                # Store the original trajectory
                orig_lons.append(lons_traj[start_idx:start_idx+24*7])
                orig_lats.append(lats_traj[start_idx:start_idx+24*7])
                idx_kept_by_parcels_list.append(idx_kept_by_parcels)
                idx_kept_by_parcels = idx_kept_by_parcels + 1


    print(f"Number of initial points: {len(init_lons)}")
    
    
    # Map each unique string-based trajectory ID to an integer
    string_to_int_map = {s: i for i, s in enumerate(set(orig_ids))}
    int_ids = [string_to_int_map[s] for s in orig_ids]
    int_to_string_map = {v: k for k, v in string_to_int_map.items()}
    recovered_strings = [int_to_string_map[i] for i in int_ids]

    if len(init_lons) == 0:
        print("No valid initial points found, exiting.")
        return

    output_dir = f'./output_lagrangian/{model_name}/'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'drifter_trajectories_leadtime7.zarr')
    #if os.path.exists(output_file):
    #    print("Output already exists, skipping...")
    #    return

    # Load velocity field from model data
    file_list = gather_and_sort_files('/Odyssey/public/glorys/rec/evaluation', model_name)
    print(file_list)
    print(xr.open_dataset(file_list[0]))

    variables = {'U': 'ugos', 'V': 'vgos'}
    dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    fieldset = FieldSet.from_netcdf(file_list, variables, dimensions, allow_time_extrapolation=True)

    class DrifterParticle(JITParticle):
        id_traj_drifter = Variable('id_traj_drifter')
        #orig_lon = Variable('orig_lon')
        #orig_lat = Variable('orig_lat')
        idx_kept_by_parcels_list = Variable('idx_kept_by_parcels_list')


    particle_uids = list(range(len(init_lons)))  # One unique ID per launch
    
    init_lons = np.array(init_lons)
    init_lats = np.array(init_lats)
    init_times_py = np.array(init_times_py)
    
    orig_lons = np.array(orig_lons)
    orig_lats = np.array(orig_lats)

    valid_mask = ~np.isnan(init_lons) & ~np.isnan(init_lats)

    print(f"Total launches prepared: {len(init_lons)}")

    launch_times = np.array(init_times_py)
    launch_min = launch_times.min()
    launch_max = launch_times.max()

    print("Launch time range:", launch_min, launch_max)
    
    pset = ParticleSet(fieldset=fieldset,
                       pclass=DrifterParticle,
                       lon=init_lons,
                       lat=init_lats,
                       time=init_times_py,
                       id_traj_drifter=np.array(int_ids),
                       idx_kept_by_parcels_list = np.array(idx_kept_by_parcels_list))
                       #orig_lon = orig_lons,
                       #orig_lat = orig_lats)

    print("Number of particles to simulate:", len(pset))

    output = pset.ParticleFile(name=output_file, outputdt=timedelta(hours=1))

    print("============== Launching 7-day advection ==============")
    pset.execute(AdvectionRK4,
                 runtime=timedelta(days=7),
                 dt=timedelta(minutes=5),
                 output_file=output)
    
    data = xr.open_dataset(output_file)
    #print(data.idx_kept_by_parcels_list.values[:,0].astype(int))
    # Add to dataset
    data = data.assign(
        orig_lon=(("trajectory", "obs"), orig_lons[data.idx_kept_by_parcels_list.values[:,0].astype(int)]),
        orig_lat=(("trajectory", "obs"), orig_lats[data.idx_kept_by_parcels_list.values[:,0].astype(int)])
    )

    # Save updated dataset
    data.to_zarr(output_file, mode="a")  # append mode
    
    # After the simulation, we now have the simulated trajectories
    #simulated_lons = pset.lon
    #simulated_lats = pset.lat

    # Store original and simulated trajectories together in a dataset
    '''
    traj_ds = xr.Dataset(
        {
            'orig_lon': (['trajectory', 'time'], np.array(orig_lons)),
            'orig_lat': (['trajectory', 'time'], np.array(orig_lats)),
            'sim_lon': (['trajectory', 'time'], simulated_lons),
            'sim_lat': (['trajectory', 'time'], simulated_lats),
        },
        coords={
            'trajectory': np.arange(len(orig_lons)),
            'time': np.arange(n_time),
        }
    )

    # Save the dataset to disk
    traj_ds.to_netcdf(os.path.join(output_dir, 'combined_trajectories.nc'))
    '''

    print("Simulated and original trajectories stored successfully.")

    return recovered_strings, int_to_string_map



import numpy as np
import pandas as pd
import xarray as xr
from geopy.distance import geodesic

def compute_separation_distances(ds_real, ds_sim, int_to_string_map):
    """
    For each matching segment between real and simulated drifters, compute the
    separation distance at each time step.

    Parameters
    ----------
    ds_real : xarray.Dataset
        Dataset of real drifters
    ds_sim : xarray.Dataset
        Dataset of simulated trajectories
    int_to_string_map : dict
        Mapping from integer ID to string-based identifiers of the form 'trajectoryID_index'
    
    Returns
    -------
    separation_dict : dict
        A dictionary with int_id as keys and list of distances (meters) as values.
    """
    separation_dict = {}
    
    # Reverse real trajectory map for quick lookup
    real_traj_map = {}  # key: (trajectory_str, index) → (lat list, lon list, time list)

    real_lons = ds_real['LONGITUDE'].values
    real_lats = ds_real['LATITUDE'].values
    real_times = ds_real['TIME'].values
    real_trajs = ds_real['trajectory'].values
    print(real_lats)

    # Build real trajectory map
    for traj_idx, traj_str in enumerate(real_trajs):
        lons = real_lons[traj_idx]
        lats = real_lats[traj_idx]
        
        print("INITIAL lats is")
        print(lats)

        # Estimate time step in seconds
        time_diffs = np.diff(real_times).astype('timedelta64[s]').astype(float)
        if len(time_diffs) == 0 or np.median(time_diffs) == 0:
            continue
        dt_seconds = np.median(time_diffs)
        step_interval = int(np.round((7 * 86400) / dt_seconds))

        for start_idx in np.arange(len(real_times))[::24*7]:
            if start_idx + step_interval >= len(real_times):
                continue
            if(not np.isnan(lats[start_idx])):
                key = f"{str(traj_str)}_{start_idx}"
            
                real_traj_map[key] = (
                    lats[start_idx:start_idx + 24*7],
                    lons[start_idx:start_idx + 24*7],
                    real_times[start_idx:start_idx + 24*7]
                )
                

    for int_id, traj_key in int_to_string_map.items():
        if traj_key not in real_traj_map:
            continue

        real_lat, real_lon, real_time = real_traj_map[traj_key]

        # Get simulated index
        sim_idx = int_id
        if sim_idx >= ds_sim.dims['trajectory']:
            continue
        
        print("sim_idx, traj_key")
        print(sim_idx)
        print(traj_key)
        

        sim_lat = ds_sim['lat'][sim_idx].values
        sim_lon = ds_sim['lon'][sim_idx].values
        
        print("real_lat, sim_lat")
        print(real_lat)
        print(sim_lat)

        # Ensure same length for distance computation
        min_len = min(len(sim_lat), len(real_lat))
        sim_lat = sim_lat[:min_len]
        sim_lon = sim_lon[:min_len]
        real_lat = real_lat[:min_len]
        real_lon = real_lon[:min_len]

        # Compute geodesic distance per time step
        distances = []
        for i in range(min_len):
            if np.isnan(sim_lat[i]) or np.isnan(real_lat[i]):
                distances.append(np.nan)
            else:
                sim_point = (sim_lat[i], sim_lon[i])
                real_point = (real_lat[i], real_lon[i])
                distance = geodesic(sim_point, real_point).meters
                distances.append(distance)

        separation_dict[int_id] = distances

    return separation_dict
