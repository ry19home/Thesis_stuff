import sys
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion, RectangleSkyRegion
from gammapy.maps import WcsGeom, MapAxis, Map, WcsNDMap
from gammapy.makers import MapDatasetMaker, SafeMaskMaker, FoVBackgroundMaker
from gammapy.data import DataStore
from gammapy.datasets import Datasets, FluxPointsDataset, MapDataset
from gammapy.modeling import Fit
from gammapy.modeling.models import SkyModel, LogParabolaSpectralModel, PointSpatialModel, PowerLawSpectralModel, FoVBackgroundModel, Models
from gammapy.estimators import FluxPoints,FluxPointsEstimator, ExcessMapEstimator
import sys, os
import numpy as np
import itertools
import re
import datetime
import gammapy
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
from pathlib import Path
from astropy import units as u
from astropy.coordinates import SkyCoord
from gammapy.stats import CashCountsStatistic, cash, cash_sum_cython
from gammapy.maps import WcsGeom, MapAxis, Map
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.data import DataStore, Observations
from gammapy.datasets import Datasets, FluxPointsDataset, MapDatasetOnOff
from gammapy.modeling import Fit
from gammapy.modeling.models import SkyModel, LogParabolaSpectralModel, PointSpatialModel, PowerLawSpectralModel, DiskSpatialModel, TemplateSpatialModel, PowerLawNormSpectralModel
from gammapy.datasets import Datasets, MapDataset
from regions import CircleSkyRegion, RectangleSkyRegion, EllipseSkyRegion
from gammapy.estimators import (
    FluxPoints,
    FluxPointsEstimator,
    EnergyDependentMorphologyEstimator)
from gammapy.modeling.models import FoVBackgroundModel, GaussianSpatialModel, ShellSpatialModel, DiskSpatialModel, PiecewiseNormSpectralModel
from gammapy.makers import FoVBackgroundMaker
from gammapy.estimators import ExcessMapEstimator, TSMapEstimator, FluxMaps
from gammapy.maps import WcsNDMap
from gammapy.visualization import plot_npred_signal, plot_distribution
import ipywidgets
from gammapy.modeling.models import Models
from astropy.coordinates import Angle
from collections import OrderedDict
import glob
import matplotlib.pyplot as plt
import datetime
from scipy.stats import chi2, norm
from astropy.table import Table
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
DB_general = pd.read_csv('/home/wecapstor1/caph/caph101h/projects/test_projects/Run_matching_tutorial/db_data.csv', header=0)
from scipy.optimize import curve_fit
import common_utils
from common_utils import get_excluded_regions
from matplotlib.lines import Line2D
from PIL import Image

import numpy as np
import pandas as pd
from importlib import reload
from scipy.optimize import least_squares, minimize
import sys
import runmatching_utilities as util
sys.path.insert(1, '/home/wecapstor1/caph/caph101h/projects/test_projects/Run_matching_tutorial')
import my_utility_functions as muf
from importlib import reload
reload(util)
reload(muf)



def run_match_process(on_runs, off_run_database, randomness, duration, Fits_Era , nsb, muon_eff, transparency, trig_rate, radio):
    for i in range(randomness):
        np.random.shuffle(on_runs)
        np.random.shuffle(off_run_database)
        
    RM = util.run_matching(OnRuns=on_runs, OffRuns=off_run_database, FitsEra=Fits_Era, duration=duration, nsb=nsb, muon_eff=muon_eff, transparency=transparency, trig_rate=trig_rate, radio=radio)
    runmatching_list = RM.matching_operation()

    surviving_on = [item[0] for item in runmatching_list]
    surviving_off = [item[1] for item in runmatching_list]
    surviving_deviation = [item[2] for item in runmatching_list]

    remaining_on_runs = muf.find_missing_runs(on_runs, surviving_on)

    print(f'Finished run matching, {len(surviving_on)} survived On runs')
    print(f'Finished run matching, {len(surviving_off)} survived Off runs')
    
    return surviving_on, surviving_off, surviving_deviation, remaining_on_runs


def datastore_application(on_runs, off_runs, deviation, affected_runs, change_necessary=False, surviving_on=None, individual_off=None, individual_deviation=None):    
    print(f'{len(on_runs)} initial On runs')
    print(f'{len(off_runs)} initial Off runs')
    print(f'Mean Deviation of initial On & Off runs: {np.mean(deviation)}')
    
    if change_necessary:
        indices_to_change = np.where(np.isin(on_runs, surviving_on))
        off_runs[indices_to_change] = individual_off
        deviation[indices_to_change] = individual_deviation


    basedir_off_runs = '/home/wecapstor1/caph/shared/hess/fits/fits_flashcam_stereo/OFFruns/out'
    basedir_on_runs = '/home/wecapstor1/caph/shared/hess/fits/fits_flashcam_stereo/Monogem/out/'

    ds_off = DataStore.from_dir(basedir_off_runs,
                                'hdu-index-bg-3d-v07b-fov-radec.fits.gz',
                                'obs-index-bg-3d-v07b-fov-radec.fits.gz')
    
    ds_on = DataStore.from_dir(basedir_on_runs,
                               'hdu-index-bg-3d-v07b-fov-radec.fits.gz',
                               'obs-index-bg-3d-v07b-fov-radec.fits.gz')

    affected_off_runs = muf.compare_arrays(off_runs, affected_runs)
    
    indices_affected = []
    for af in affected_off_runs:
        indices_affected.append(np.where(off_runs == af))
    
    on_runs_unaffected = np.delete(on_runs, [runs for runs in indices_affected])
    off_runs_unaffected = np.delete(off_runs, [runs for runs in indices_affected])
    deviation_unaffected = np.delete(deviation, [runs for runs in indices_affected])

    on_run_deleted_due_to_off_run = muf.find_missing_runs(on_runs_unaffected, on_runs)
    
    # print('\n')
    # print(f'{len(affected_off_runs)} affected OFF runs: {affected_off_runs}')
    # print(f'{len(on_run_deleted_due_to_off_run)} runs deleted, as OFF run was affected')
    
    on_runs_final = on_runs_unaffected
    off_runs_final = off_runs_unaffected
    deviation_final = deviation_unaffected
    
    print('\n')
    print(f'{len(on_runs_final)} remaining ON runs')
    print(f'{len(off_runs_final)} remaining OFF runs')
    print(f'Mean Deviation of remaining ON & OFF runs: {np.mean(deviation_final)}')
    
    obs_list_on = ds_on.get_observations(on_runs_final)
    obs_list_off = ds_off.get_observations(off_runs_final)

    return obs_list_on, obs_list_off, deviation_final, ds_on, ds_off, on_run_deleted_due_to_off_run

def analysis(on_obs_list, off_obs_list, deviation_array, ds_on, ds_off, mu_lower, mu_upper, std_lower, std_upper, tilt_lower, tilt_upper, norm_lower, norm_upper , run_difference_param):
    ra_obj = 105.12 
    dec_obj = 14.32
    target = SkyCoord(ra_obj, dec_obj, frame='icrs', unit='deg')
    
    e_reco = np.logspace(-1, 2, 25) * u.TeV 
    e_true = np.logspace(-1, 2, 50) * u.TeV 
    
    energy_axis = MapAxis.from_edges(e_reco, unit='TeV', name='energy', interp='log')
    energy_axis_true = MapAxis.from_edges(e_true, unit='TeV', name="energy_true", interp='log')
    
    FoV_width = (8, 8)
    geom = WcsGeom.create(skydir=(ra_obj, dec_obj), binsz=0.02, width=FoV_width, frame="icrs", proj="CAR", axes=[energy_axis])
    offset_max = 2 * u.deg
    maker = MapDatasetMaker()
    maker_safe_mask2 = SafeMaskMaker(methods=["offset-max", 'aeff-default', 'aeff-max', 'edisp-bias', 'bkg-peak'], offset_max=offset_max, bias_percent=10)

    ########################################################################################################################################################################
    Information_table = {'On_run': [], 'Off_run': [], 'Deviation': [] ,
                         'Mean_off_run': [], 'Std_off_run': [], 'Excess_off_run': [],
                         'Mean_on_run': [], 'Std_on_run': [], 'Excess_on_run': [],
                         'Norm': [], 'Norm_error': [], 'Tilt': [], 'Tilt_error': []}
    
    alright_runs = []
    dropped_runs = []

    stacked_name = 'Monogem_stacked_Dataset'
    stacked = MapDataset.create(geom=geom, name=stacked_name, energy_axis_true=energy_axis_true)
    
    stacked_name_good_runs = 'Monogem_stacked_Dataset_good_runs'
    stacked_good_runs = MapDataset.create(geom=geom, name=stacked_name_good_runs, energy_axis_true=energy_axis_true)

    iteration = 0
    
    while len(alright_runs) == 0 or len(dropped_runs) == len(on_obs_list):
        print(f'Iteration {iteration}: Mu low {round(mu_lower, 3)}, Mu up {round(mu_upper, 3)}; Std low {round(std_lower, 3)}, Std up {round(std_upper, 3)}; Tilt low {round(tilt_lower, 3)}, Tilt up {round(tilt_upper, 3)}; Norm low {round(norm_lower, 2)}, Norm up {round(norm_upper, 2)}; Run diff. {round(run_difference_param,3)}')
        alright_runs.clear()
        dropped_runs.clear()
        for m in range(len(on_obs_list)):
            obs = on_obs_list[m]
            off_run = off_obs_list[m]
            deviation = deviation_array[m]

            # ON run dataset geometry 
            cutout = stacked.cutout(obs.pointing.fixed_icrs, width=2 * offset_max, name=f"obs-{obs.obs_id}")
            dataset = maker.run(cutout, obs)
            dataset = maker_safe_mask2.run(dataset, obs)   

            number_on = muf.find_energy_threshold(dataset)

            # OFF run dataset geometry:
            geom_off = WcsGeom.create(skydir=off_run.pointing.fixed_icrs, binsz=0.02, width=FoV_width, frame="icrs", proj="CAR", axes=[energy_axis])
        
            stacked_off = MapDataset.create(geom=geom_off, name=f"obs-{off_run.obs_id}", energy_axis_true= energy_axis_true)
            cutout_off = stacked_off.cutout(off_run.pointing.fixed_icrs, width=2 * offset_max, name=f"obs-{off_run.obs_id}")
        
            # Define exclusion region and safe mask for the background fit
            maker_fov_off, ex = muf.get_exclusion_mask(off_run, geom_off, 10, radius_source_mask=.5, offset_max=offset_max)

            # Fit the background to the OFF run
            dataset_off = maker.run(cutout_off, off_run)
            dataset_off = maker_safe_mask2.run(dataset_off, off_run)
        
            number_off = muf.find_energy_threshold(dataset_off)

            # Find the highest energy threshold between ON and OFF run
            number = max(number_on, number_off)

            # Apply the energy threshold to both ON and OFF run
            dataset.background = muf.adjust_energy_threshold(dataset.background, number)
            dataset.counts = muf.adjust_energy_threshold(dataset.counts, number)
            dataset_off.background = muf.adjust_energy_threshold(dataset_off.background, number)
            dataset_off.counts = muf.adjust_energy_threshold(dataset_off.counts, number)

            # The background fit
            bkg_model_off = FoVBackgroundModel(dataset_name=dataset_off.name)
            dataset_off.models = bkg_model_off
            dataset_off.background_model.spectral_model.tilt.frozen = False
            dataset_off = maker_fov_off.run(dataset_off)

            bkg_model_on = FoVBackgroundModel(dataset_name=dataset.name)
            dataset.models = bkg_model_on

            dataset.background_model.spectral_model.norm.value = dataset_off.background_model.spectral_model.norm.value
            dataset.background_model.spectral_model.tilt.value = dataset_off.background_model.spectral_model.tilt.value
            dataset.background_model.spectral_model.reference.value = dataset_off.background_model.spectral_model.reference.value
        
            dataset.background_model.spectral_model.norm.frozen = True
            dataset.background_model.spectral_model.tilt.frozen = True
        
            # Apply the corrections
            dataset = muf.livetime_corr(obs, off_run, dataset)
            dataset = muf.zenith_corr(obs, off_run, dataset, ds_on, ds_off)

            total_counts = dataset.info_dict()['counts']
            predicted_total_counts = dataset.info_dict()['npred']

            difference_between_runs = np.abs(predicted_total_counts - total_counts)/predicted_total_counts *100

            counts_on_ppx = np.mean(dataset.counts.data / np.sqrt(dataset.counts.geom.bin_volume())) * u.TeV**.5 * u.sr**.5
            counts_off_ppx = np.mean(dataset_off.counts.data / np.sqrt(dataset_off.counts.geom.bin_volume())) * u.TeV**.5 * u.sr**.5
    
            pixel_diff = np.abs(counts_on_ppx - counts_off_ppx)

            # print(difference_between_runs)
            
            obs_id = obs.obs_id
            obs_id_off = off_run.obs_id
            excess_vals_on = dataset.info_dict()['excess']
            excess_vals_off = dataset_off.info_dict()['excess']
            norm_val = dataset_off.background_model.spectral_model.norm.value
            norm_err = dataset_off.background_model.spectral_model.norm.error
            tilt = dataset_off.background_model.spectral_model.tilt.value
            tilt_error = dataset_off.background_model.spectral_model.tilt.error
        
            # Significance of the model:
            estimator_001 = ExcessMapEstimator(correlation_radius="0.15 deg")
            lima_maps_001 = estimator_001.run(dataset_off)
     
            significance_map = lima_maps_001["sqrt_ts"] 
            significance_off = significance_map.data[np.isfinite(significance_map.data)]
            mu, std = norm.fit(significance_off)
    
            estimator_002 = ExcessMapEstimator(correlation_radius="0.15 deg")
            lima_maps_002 = estimator_002.run(dataset)
     
            significance_map_on = lima_maps_002["sqrt_ts"] 
            significance_map_on_excl = significance_map_on.data
            significance_on = significance_map_on_excl[np.isfinite(significance_map_on_excl)]
            mu_on, std_on = norm.fit(significance_on)
            
            if mu_lower < mu < mu_upper and std_lower < std < std_upper:
                if tilt_lower < tilt < tilt_upper and norm_lower < norm_val < norm_upper:
                    alright_runs.append((obs_id, obs_id_off, deviation))
                else:
                    dropped_runs.append((obs_id, obs_id_off, deviation))  
            else:
                dropped_runs.append((obs_id, obs_id_off, deviation)) 

        if len(alright_runs) == 0 or len(dropped_runs) == len(on_obs_list):
            print('Adjusting the values and re-run the analysis')
            mu_lower *= 1.05
            mu_upper *= 1.05
            std_lower *= 0.95
            std_upper *= 1.05
            tilt_lower *= 1.05
            tilt_upper *= 1.05
            norm_lower *= 0.95
            norm_upper *= 1.05
            #run_difference_param *= 1.1
            iteration +=1
            if iteration == 10:
                break

    print(f'{len(alright_runs)} sufficed run analysis')
    print(f'{len(dropped_runs)} did not sufficed run analysis')

    for obs_id, obs_id_off, deviation in alright_runs:
        Information_table['On_run'].append(obs_id)   
        Information_table['Off_run'].append(obs_id_off)
        Information_table['Deviation'].append(deviation)  
        Information_table['Mean_off_run'].append(mu) 
        Information_table['Std_off_run'].append(std)  
        Information_table['Excess_off_run'].append(excess_vals_off)   
        Information_table['Mean_on_run'].append(mu_on)
        Information_table['Std_on_run'].append(std_on) 
        Information_table['Excess_on_run'].append(excess_vals_on)
        Information_table['Norm'].append(norm_val)
        Information_table['Norm_error'].append(norm_err)
        Information_table['Tilt'].append(tilt)
        Information_table['Tilt_error'].append(tilt_error)
    
    return alright_runs, dropped_runs, Information_table



def load_run_base(base_dirs, generation_prefix_range=None, iteration_range=None):
    # Initialize list to store concatenated arrays
    all_data = []
    
    # Iterate over all base directories
    for base_dir in base_dirs:
        # Iterate over all generation directories
        for gen_dir in os.listdir(base_dir):
            # Check if generation directory is within the specified numerical prefix range
            if generation_prefix_range:
                try:
                    # Extract numerical suffix from the generation directory
                    gen_number = int(gen_dir.split('_')[-1])  # Assuming naming format 'generation_<number>'
                    
                    # Check if the current generation number falls within the range
                    if not (generation_prefix_range[0] <= gen_number <= generation_prefix_range[1]):
                        continue
                except (ValueError, IndexError):
                    # Handle cases where the generation directory name doesn't contain a valid integer
                    print(f"Invalid generation directory name: {gen_dir}")
                    continue
            
            gen_path = os.path.join(base_dir, gen_dir)
            
            # Iterate over all iteration directories in the current generation directory
            for iteration_dir in os.listdir(gen_path):
                try:
                    # Extract iteration number (assuming iteration_dir is in the format 'iteration_<number>')
                    iteration_number = int(iteration_dir.split('_')[-1])
                    
                    # Check if the iteration number is within the specified range
                    if iteration_range and not (iteration_range[0] <= iteration_number <= iteration_range[1]):
                        continue
                    
                    iteration_path = os.path.join(gen_path, iteration_dir)
                    
                    # Define the paths for on, off, and deviation files
                    passed_on_path = os.path.join(iteration_path, f'passed_on_runs_{iteration_dir}.txt')
                    passed_off_path = os.path.join(iteration_path, f'passed_off_runs_{iteration_dir}.txt')
                    passed_deviation_path = os.path.join(iteration_path, f'passed_deviation_{iteration_dir}.txt')
                    
                    # Check if all required files exist
                    if os.path.exists(passed_on_path) and os.path.exists(passed_off_path) and os.path.exists(passed_deviation_path):
                        try:
                            passed_on_array = np.loadtxt(passed_on_path)
                            passed_off_array = np.loadtxt(passed_off_path)
                            passed_deviation_array = np.loadtxt(passed_deviation_path)
                            
                            # Check if arrays are valid and convert 0-D arrays to 1-D
                            passed_on_array = np.atleast_1d(passed_on_array)
                            passed_off_array = np.atleast_1d(passed_off_array)
                            passed_deviation_array = np.atleast_1d(passed_deviation_array)
        
                            # Ensure all arrays have the same length
                            min_length = min(len(passed_on_array), len(passed_off_array), len(passed_deviation_array))
                            passed_on_array = passed_on_array[:min_length]
                            passed_off_array = passed_off_array[:min_length]
                            passed_deviation_array = passed_deviation_array[:min_length]
                            
                            # Stack arrays horizontally to form triplets
                            combined_array = np.column_stack((passed_on_array, passed_off_array, passed_deviation_array))
                            
                            # Append combined array to list
                            all_data.append(combined_array)
                        except Exception as e:
                            print(f"Error loading files for iteration {iteration_dir} in {gen_dir}: {e}")
                except ValueError:
                    # Handle cases where the iteration directory name doesn't contain a valid integer
                    print(f"Invalid iteration directory name: {iteration_dir}")
    
    # Concatenate all combined arrays vertically
    if all_data:
        final_array = np.vstack(all_data)
    else:
        final_array = np.array([])
    # Create a DataFrame from the final array
    df = pd.DataFrame(final_array, columns=['on', 'off', 'deviation'])
    
    # Sort by 'on' and 'deviation' to ensure smallest deviations come first
    df = df.sort_values(by=['on', 'deviation'])
    
    # Initialize structures to hold the unique off runs and results
    used_off_runs = set()
    final_selection = []
    # Iterate through each unique 'on' run
    for on_run in df['on'].unique():
        # Get all rows corresponding to the current 'on' run
        on_run_df = df[df['on'] == on_run]
        
        # Get the unique 'off' runs for this 'on' run
        unique_off_runs = on_run_df['off'].unique()
    
        # If there is only one unique 'off' run for this 'on' run, add it
        if len(unique_off_runs) == 1:
            selected_row = on_run_df.iloc[0]  # Take the row with the smallest deviation
            # Check if the row is the one to be excluded
            if not (selected_row['on'] == 165394 and selected_row['off'] == 166897):
                final_selection.append(selected_row)
                used_off_runs.add(selected_row['off'])  # Mark this 'off' run as used
        else:
            # If there are multiple unique 'off' runs
            for idx, row in on_run_df.iterrows():
                # Skip the pair On 165394, Off 166897
                if row['on'] == 165394 and row['off'] == 166897:
                    continue
                if row['off'] not in used_off_runs:
                    final_selection.append(row)  # Add the row with the smallest deviation
                    used_off_runs.add(row['off'])  # Mark this 'off' run as used
                    break  # Stop after adding the first unused 'off' run
                    
    # Convert final_selection to a DataFrame if needed
    final_selection_df = pd.DataFrame(final_selection)
    
    # Convert the DataFrame to a final array
    final_min_deviation_array = final_selection_df[['on', 'off', 'deviation']].to_numpy()
    
    # Print final result for validation
    print(f'{len(final_min_deviation_array)} runs contained')
    
    on_runs = final_min_deviation_array[:, 0]
    off_runs = final_min_deviation_array[:, 1]
    deviation = final_min_deviation_array[:, 2]
    
    return on_runs, off_runs, deviation

















