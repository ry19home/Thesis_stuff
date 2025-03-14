from collections import Counter
import os
from gammapy.datasets import MapDataset, MapDatasetOnOff
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm

import runmatching_utilities as util
import numpy as np
import pandas as pd
from importlib import reload
from scipy.optimize import least_squares, minimize
import sys

sys.path.insert(1, '/home/wecapstor1/caph/caph101h/projects/test_projects/Run_matching_tutorial')
import my_utility_functions as muf
from importlib import reload
reload(util)
reload(muf)

from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion, RectangleSkyRegion
from gammapy.maps import WcsGeom, MapAxis, Map, WcsNDMap
from gammapy.makers import MapDatasetMaker, SafeMaskMaker, FoVBackgroundMaker
from gammapy.data import DataStore
from gammapy.datasets import Datasets, FluxPointsDataset, MapDataset
from gammapy.modeling import Fit
from gammapy.modeling.models import SkyModel, LogParabolaSpectralModel, PointSpatialModel, PowerLawSpectralModel, FoVBackgroundModel, Models
from gammapy.estimators import FluxPoints,FluxPointsEstimator, ExcessMapEstimator

import os
import itertools
import re
import datetime
import gammapy
import time
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
from gammapy.estimators import FluxPoints, FluxPointsEstimator, EnergyDependentMorphologyEstimator
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
from collections import Counter

def find_duplicates(arr):
    # Count the occurrences of each element in the array
    counts = Counter(arr)
    
    # Find elements that appear exactly twice
    duplicates = [item for item, count in counts.items() if count == 2]
    
    return duplicates
    
def find_missing_runs(array_total, array_to_compare):
    non_matching_values = np.setdiff1d(array_total, array_to_compare)

    return non_matching_values

def find_indices_of_duplicates(arr):
    # Convert the array to a list if it's a numpy array
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()

    value_indices = {}
    duplicates = []

    for index, value in enumerate(arr):
        if value in value_indices:
            if value_indices[value] is not None:
                duplicates.append(value_indices[value])
                duplicates.append(index)
                value_indices[value] = None  # Mark as processed
        else:
            value_indices[value] = index

    return duplicates

def find_indices_of_repeated_values(arr):
    # Convert the array to a list if it's a numpy array
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()

    from collections import defaultdict

    value_indices = defaultdict(list)
    repeated_indices = []

    for index, value in enumerate(arr):
        value_indices[value].append(index)

    for indices in value_indices.values():
        if len(indices) > 1:
            repeated_indices.extend(indices)

    return repeated_indices


def remove_shared_values(larger_array, smaller_array):
    """
    Remove elements from larger_array that are also in smaller_array.

    Parameters:
    larger_array (list): The larger array from which elements will be removed.
    smaller_array (list): The smaller array containing elements to be removed from the larger array.

    Returns:
    list: A new list with elements from larger_array that are not in smaller_array.
    """
    # Convert the smaller array to a set for faster look-up times
    smaller_set = set(smaller_array)
    
    # Use list comprehension to filter out elements present in the smaller set
    result_array = [element for element in larger_array if element not in smaller_set]
    
    return result_array

def sigma_to_ts(sigma, df):
    """
    Convert sigma to delta ts.
    Parameters:
    - sigma (float): The sigma you want to have as a Test Significance (TS).
    - df (int): Degree('s) of freedom of your fit.

    Returns:
    TS value.
    
    """
    p_value = 2 * norm.sf(sigma)
    return chi2.isf(p_value, df=df)

def ts_to_sigma(ts, df):
    """
    Convert delta ts to sigma.
    
    Parameters:
    - ts (float): The Test Significance (TS) you want in sigmas.
    - df (int): Degree('s) of freedom of your fit.

    Returns:
    Significance sigma.
    
    """
    p_value = chi2.sf(ts, df=df)
    return norm.isf(0.5 * p_value)

def model_significance(dataset_model, dataset_without_model, print_result=True):
    """
    Calculate the Test Statistic (TS) and Significance of model.

    Parameters:
    - dataset_model (MapDataset): The MapDataset with model applied.
    - dataset_without_model (MapDataset): The MapDataset without model applied.
    - print_result (bool): Whether or not to print the result, default at True.

    Returns:
    The TS and significance of the model compared to no model MapDataset.  
    """
    delta_ts = np.abs(dataset_without_model.stat_sum() - dataset_model.stat_sum())

    DF = len(dataset_model.models.parameters.free_parameters)
    sqrt_delta_ts = ts_to_sigma(ts=delta_ts, df=DF)

    if print_result == True:
        print(f'Test Statistic: {delta_ts}')
        print(f'Significance: {sqrt_delta_ts}')

    return (delta_ts, sqrt_delta_ts)

def compare_arrays(array1, array2):
    counter1 = Counter(array1)
    counter2 = Counter(array2)

    if counter1 == counter2:
        return list((counter1 & counter2).elements())
    else:
        return list((counter1 & counter2).elements())

def find_non_matching_values(array1, array2):
    # Convert arrays to sets
    set1 = set(array1)
    set2 = set(array2)
    
    # Find values that are in set1 but not in set2 and vice versa
    non_matching_values = list((set1 - set2) | (set2 - set1))
    
    return non_matching_values


def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def load_and_process_file(file_path, type_):
    """
    Load a dataset from a FITS file and process it.

    Parameters:
    - file_path (str): Path to the FITS file.
    - type (gammapy.datasets): Currently supported: MapDataset, MapDatasetOnOff

    Returns:
    tuple: Tuple containing (dataset_name, selected_dataset).
    """
    #print('Taking files from:', file_path)
    if type_ == MapDataset:    
        dataset = type_.read(file_path)
        dataset_name = os.path.splitext(os.path.splitext(os.path.basename(file_path))[0])[0]

    elif type_ == MapDatasetOnOff:
        dataset = type_.read(file_path)
        dataset_name = os.path.splitext(os.path.splitext(os.path.basename(file_path))[0])[0]
        
    return dataset_name, dataset
    

def load_fits_data(output_directory, type_):
    """
    Load FITS datasets from the specified directory using ThreadPoolExecutor.

    Parameters:
    - output_directory (str): The directory where the FITS files are stored.

    Returns:
    dict: A dictionary where the keys are dataset names and the values are MapDataset objects.
    """
    print('Begin loading, depending on file size this may take a while.')
    
    # Get the list of FITS files
    fits_files = [os.path.join(output_directory, file_name) for file_name in os.listdir(output_directory) if file_name.endswith('.fits.gz')]
    num_files = len(fits_files)
    print(f'Total number of files {num_files}')
    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_and_process_file, file_path, type_) for file_path in fits_files]
        
        loaded_datasets_dict = {}
        
        # Process each future as it completes
        for i, future in enumerate(as_completed(futures), start=1):
            dataset_name, dataset = future.result()
            loaded_datasets_dict[dataset_name] = dataset
    
    print('Finished loading the data!')
    
    return loaded_datasets_dict

def filter_and_categorize(region, dataset_base, mask_value):
    """
    Filters dataset names based on a given regex pattern and categorizes the datasets into two lists:
    one for names that match the pattern but do not end in 'mask_value', and another for the rest.

    Parameters:
    - region (string): Mask region you want to select, currently only up, middle, down
    - dataset_base (dict): Dictionary containing dataset names as keys and their associated data as values.
    - mask_value (string) : Selection for which mask value you want to extract for the region.

    Returns:
    - matches (dict): Dictionary of the matches in accordance with set parameters.
    - no_matches (dict): Dictionary containing all datasets that did not match the criteria.
    """
    # Compile the regular expression pattern
    compiled_pattern = re.compile(region + r'_(\d+\.\d+)')
    
    # Initialize dictionaries
    matches = {}
    no_matches = {}
    
    # Categorize the datasets
    for dataset_name in dataset_base.keys():
        match = compiled_pattern.search(dataset_name)
        if match and match.group(1) == mask_value:
            matches[dataset_name] = dataset_base[dataset_name]
        else:
            no_matches[dataset_name] = dataset_base[dataset_name]

    print(f'Database contains {len(matches)} matched cases.')
    
    return matches, no_matches

def find_coordinates_above_threshold(map_, threshold, frame):
    """
    Find coordinates where the values in significance_map.data are above a given threshold.

    Parameters:
    - significance_map (WcsNDMap): The WCS map containing the data.
    - threshold (float): The threshold value for selecting coordinates.
    - frame (string): The WCS you want to use, e.g., 'icrs' or 'galactic'

    Returns:
    - sky_coords (SkyCoord): An Astropy SkyCoord object containing the coordinates.
    """
    # Step 1: Find the indices where the condition holds
    indices = np.where(~np.isnan(map_.data) & (map_.data >= threshold))
    
    # Step 2: Extract the pixel coordinates from the indices
    pixel_coords = tuple(indices[::-1]) 
    # Step 3: Convert pixel coordinates to coordinates
    world_coords = map_.geom.pix_to_coord(pixel_coords)
    # Convert to SkyCoord
    sky_coords = SkyCoord(world_coords[0], world_coords[1], unit='deg', frame=frame)

    return sky_coords

def save_figures(plot_directory, name):
    """
    Saves the current matplotlib figure in both PNG and PDF formats.

    Parameters:
    plot_directory (str): The directory where the figure will be saved.
    name (str): The base name of the saved figure files.

    This function will save the current figure as 'name.png' and 'name.pdf'
    in the specified plot_directory.
    """
    save_types = ['png', 'pdf']
    for ext in save_types:
        plt.savefig(f'{plot_directory}/{name}.{ext}', bbox_inches = 'tight')

def extract_value_from_string(input_string):
    # Define the regular expression pattern to match the number before '.fits.gz'
    pattern = r'_rad_([0-9]*\.?[0-9]+)'

    # Search for the pattern in the input string
    match = re.search(pattern, input_string)

    # Check if a match is found and return the value, otherwise return None
    if match:
        return float(match.group(1))
    else:
        return None



def find_energy_threshold(dataset):
    number = 0
    for k in range(0,dataset.counts.data.shape[0]): 
        for l in range(0,dataset.counts.data.shape[1]):
            for n in range(0,dataset.counts.data.shape[2]):
                if dataset.mask_safe.data[k][l][n] == True:
                    number = k 
                    break
        if number != 0:
            break
    return number

def get_exclusion_mask(off_run, geom_off, radius, radius_source_mask, offset_max):
    """ Define the exclusion mask that will be used for the background fit """ 
    
    hap_exclusion_regions = get_excluded_regions(geom_off.center_coord[0].value, geom_off.center_coord[1].value, radius)
    excl_regions = []
    for source in hap_exclusion_regions:
        center = SkyCoord(source.ra, source.dec, unit='deg', frame='icrs')
        region = CircleSkyRegion(center=center, radius=source.radius*u.deg)
        excl_regions.append(region)

    tel_pointing = off_run.pointing.fixed_icrs
    source_pos = DB_general[DB_general['Run']==off_run.obs_id]
    ra = tel_pointing.ra.value #- source_pos['Offset_x'].iloc[0]
    dec = tel_pointing.dec.value #- source_pos['Offset_y'].iloc[0]
    excl_regions.append(CircleSkyRegion(center=SkyCoord(ra*u.deg, dec*u.deg), radius=radius_source_mask*u.deg))
    
    data2 = geom_off.region_mask(regions=excl_regions, inside=False)
    maker_fov_off = FoVBackgroundMaker(method="fit", exclusion_mask=data2)
    ex = maker_fov_off.exclusion_mask.cutout(off_run.pointing.fixed_icrs, width=2 * offset_max)
    return maker_fov_off, ex

def adjust_energy_threshold(dataset, number):
    """ Apply a predifined save energy to the counts and background of a dataset """
    
    bkg_array = np.zeros_like(dataset.data)
    for a in range(number, bkg_array.shape[0]):
        for b in range(0,dataset.data.shape[1]):
            for c in range(0,dataset.data.shape[2]):
                bkg_array[a][b][c] = dataset.data[a][b][c]
    
    masked_data = WcsNDMap(geom=dataset.geom, data=np.array(bkg_array))
    return masked_data

def livetime_corr(obs, off_run, dataset):
    """ Correct for the differences in deadtime corrected observation time """
    
    livetime_dev = off_run.observation_live_time_duration.value - obs.observation_live_time_duration.value
    counts_per_sec = dataset.background.data/off_run.observation_live_time_duration.value
    factors = counts_per_sec*livetime_dev
    bkg = [x + y for x, y in zip(dataset.background.data, factors)]

    bkg = WcsNDMap(geom=dataset.counts.geom, data=np.array(bkg))
    dataset.background = bkg
    return dataset

    
def zenith_corr(obs, off_run, dataset, ds_on, ds_off):
    """ Correct for the differences in zenith angle """
    muon_phases2 = [20000, 40093, 57532, 60838, 63567, 68545, 80000, 85000, 95003,
                    100800, 110340, 127700, 128600, 132350, 154814, 190000]
    
    DB_general = pd.read_csv(r'/home/wecapstor1/caph/shared/hess/fits/database_image/data_1223.csv', header=0)
    correction_factor = pd.read_csv(r'/home/wecapstor1/caph/caph101h/projects/test_projects/Run_matching_tutorial/zenith_correction_factors.csv', sep='\t')
    
    for phase in range(0,len(muon_phases2)-1):
       # print(obs.obs_id,  muon_phases2[phase], muon_phases2[phase+1])
        if obs.obs_id > muon_phases2[phase] and obs.obs_id < muon_phases2[phase+1]:
            factor = correction_factor.x_2.iloc[phase]

    bkg = dataset.background.data
    zenith_off = np.deg2rad(ds_off.obs_table[ds_off.obs_table['OBS_ID']==off_run.obs_id]["ZEN_PNT"])
    zenith_on = np.deg2rad(ds_on.obs_table[ds_on.obs_table['OBS_ID']==obs.obs_id]["ZEN_PNT"])
    bkg = np.array(bkg) * np.cos(zenith_on - zenith_off)**factor
    
    bkg = WcsNDMap(geom=dataset.counts.geom, data=np.array(bkg))
    dataset.background = bkg
    return dataset

def systematics(obs, off_run, dataset, sys):
    """ Get the upper or lower limit of the background rate from the systematic error on the background template and run matching """
    run_dev = self.deviation[self.obs_list.index(obs)]
    bkg = dataset.background.data
    binsz = (self.systematic_shift['run dev'].iloc[1] - self.systematic_shift['run dev'].iloc[0])/2
    for i in range(0,len(self.systematic_shift)):
        if run_dev > self.systematic_shift['run dev'].iloc[i]-binsz and run_dev < self.systematic_shift['run dev'].iloc[i]+binsz:
            index = i
            sys_factor = self.systematic_shift['sys dev error'].iloc[i]
            if sys_factor == 0:
                closest_filled = index - min((index - self.systematic_shift[self.systematic_shift['sys dev'] != 0].index), key=abs)
                sys_factor = self.systematic_shift['sys dev error'].iloc[closest_filled]
        
        elif run_dev < self.systematic_shift['run dev'].iloc[0]:
            sys_factor = self.systematic_shift['sys dev error'].iloc[0]

        elif run_dev > self.systematic_shift['run dev'].iloc[-1]:
            sys_factor = self.systematic_shift['sys dev error'].iloc[-1]
    
    if sys == 'low':
        bkg = [x - x*sys_factor for x in bkg] 
    elif sys == 'high':
        bkg = [x + x*sys_factor for x in bkg]   
    
    bkg = WcsNDMap(geom=dataset.counts.geom, data=np.array(bkg))
    dataset.background = bkg
    return dataset


def load_data_to_dataframe(directories):
    # Initialize a list to store the data
    data = []
    
    # Loop over each directory
    for directory in directories:
        # Define the file pattern
        file_pattern = os.path.join(directory, 'Off_run_list_for_on_run_Id_Obs_id_*.txt')
        
        # Use glob to find all files matching the pattern
        matching_files = glob.glob(file_pattern)
        
        # Loop over each file and extract the required information
        for file_path in matching_files:
            # Extract the On run Id from the file name
            filename = os.path.basename(file_path)
            on_run_id = filename.split('_')[9].replace('.txt', '')  # Assuming the pattern doesn't change
            
            # Load the file content into a DataFrame
            file_df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['Off run Id', 'Deviation'])
            
            # Add the On run Id to each row
            file_df['On run Id'] = on_run_id
            
            # Append the data to the list
            data.append(file_df)
    
    # Concatenate all data into a single DataFrame
    final_df = pd.concat(data, ignore_index=True)
    
    # Reorder the columns
    final_df = final_df[['On run Id', 'Off run Id', 'Deviation']]
    
    return final_df





