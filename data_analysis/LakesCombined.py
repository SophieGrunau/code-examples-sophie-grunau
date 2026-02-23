#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lake-Filling Event Analysis: Rainfall Drivers and Satellite Observations

Author:      Lea Sophie Grunau  
Created on:  2025-08-04
Last updated: 2025-12-02

Description:
    Analyzes lake-filling events in Australian ephemeral lakes using satellite observations
    (Digital Earth Australia) and gridded rainfall data (AGCD). Identifies filling events,
    calculates rainfall triggers, and explores relationships with climate drivers (ENSO, IPO).

This script:
    • Loads and processes satellite lake observations, gridded rainfall, and station data
    • Identifies lake-filling events from DEA time series using a 10% threshold
    • Calculates rolling sum rainfall and cumulative rainfall for each event
    • Classifies rainfall periods by threshold exceedance and event association
    • Generates comprehensive figures showing:
        - Catchment maps with station locations
        - Lake size time series with detected events
        - Cumulative rainfall patterns for events
        - Composite maps of event vs non-event rainfall
        - Rolling sum analysis with variable windows
        - Climate driver (ENSO/IPO) relationships with lake filling

Dependencies:
    - Python 3.x
    - Core libraries: numpy, pandas, xarray, scipy
    - Visualization: matplotlib, cartopy
    - Data files:
        • {Lake}_mask_r005.nc (catchment mask)
        • agcd_v1_precip_total_r005_daily_{Lake}_1900to2024.nc (gridded rainfall)
        • {Lake}_daily_station_rainfall.nc (rainfall stations)
        • {Lake}_daily_station_runoff.nc (runoff stations)
        • agcd_v1_precip_total_r005_daily_NAus_1900to2024_masked.nc (regional rainfall)
        • {Lake}_dea.nc (satellite lake observations)
        • ClimaticDrivers/enso_*.nc (ENSO indices)
        • ClimaticDrivers/tpi.timeseries*.nc (IPO indices)

Configuration:
    - Lake: Set via 'Lake' variable (e.g., 'LW' for Lake Woods)
    - Analysis period: 1900-2024 (configurable via ds_start/ds_end)
    - Rolling window: 112 days (configurable via window_size_daily)
    - Thresholds: 600mm and 800mm (configurable via lower/higher_threshold_daily)

Note: 
    File paths in the CONFIGURATION section must be adapted to your system.

Output:
    Generates Figures 1-8 as PNG files in the output directory, showing various aspects
    of lake-filling event analysis and rainfall patterns.

Usage:
    1. If you don't have the data files, run the download script first:
        python download_data.py
    2. Update the 'Lake' variable and file paths in the CONFIGURATION section
    3. Run the entire script or individual figure sections as needed
    4. Figures are saved automatically to the output directory
    
    Example for different lake:
        Lake = 'LE'  # Change to Lake Eyre
        # Update file paths accordingly
        # Run script: python LakesCombined.py

References:
    - Digital Earth Australia: https://www.ga.gov.au/dea
    - AGCD rainfall data: http://www.bom.gov.au/climate/data/
"""

# ========== IMPORTS ==========
from pathlib import Path
import glob
import pandas as pd
import xarray as xr
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib import colormaps
import matplotlib.lines as mlines
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
import cartopy.feature as cfeature


# ========== CONFIGURATION ==========
# Lake identifier
Lake = 'LW'

# Analysis period
ds_start = '1900'
ds_end = '2024'

# Rolling sum parameters
window_size_daily = 112          # Rolling window size (days)
lower_threshold_daily = 600      # Lower rainfall threshold (mm)
higher_threshold_daily = 800     # Higher rainfall threshold (mm)

# Computation settings
dask_chunks = 3650


# ========== FILE PATHS ==========
# Directories
try:
    base_dir = Path(__file__).parent
except NameError:
    base_dir = Path.cwd()
input_dir = base_dir / "data"
output_dir = base_dir / "output"
# Create directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Input files
Lake_mask_r005_file = f'{input_dir}/{Lake}_mask_r005.nc'
agcd_daily_file = f'{input_dir}/agcd_v1_precip_total_r005_daily_{Lake}_1900to2024.nc'
runoff_stations_file = f'{input_dir}/{Lake}_daily_station_runoff.nc'
rainfall_stations_file = f'{input_dir}/{Lake}_daily_station_rainfall.nc'
agcd_NT_daily_file = f'{input_dir}/agcd_v1_precip_total_r005_daily_NAus_1900to2024.nc'
NT_mask_r005_file = f'{input_dir}/NT_mask_r005.nc'
dea_file = f'{input_dir}/{Lake}_dea.nc'
enso_file = f'{input_dir}/ClimaticDrivers/enso_*.nc'
ipo_file = f'{input_dir}/ClimaticDrivers/tpi.timeseries*.nc'


# ========== METADATA ATTRIBUTES ==========
ds_attrs = {
    'agcd': {
        'source': 'AGCD',
        'units': 'mm',
        'long_name': {
            'gridded_catchment_rainfall': 'Daily gridded catchment rainfall',
            'mean_catchment_rainfall': 'Daily mean catchment rainfall',
            'cumulative_rainfall_per_event': 'Daily cumulative rainfall per event',
            'rolling_sum_of_mean_catchment_rainfall': f'Daily rolling sum of mean catchment rainfall ({window_size_daily}-days)',
            'cumulative_rainfall_up_to_peaks': f'Daily cumulative rainfall up to peaks of {window_size_daily}-day windows'
        },
        'description': {
            'gridded_catchment_rainfall': f'Daily gridded catchment rainfall over {Lake}.',
            'mean_catchment_rainfall': f'Daily mean catchment rainfall over {Lake}.',
            'cumulative_rainfall_per_event': (
                f'Daily cumulative rainfall per event, based on mean catchment rainfall over {Lake}. '
                f'Starting from the first day of each identified event up to the day of peak lake size. '
                f'Coordinates identify event number, peak day, and offset from peak.'
            ),
            'rolling_sum_of_mean_catchment_rainfall': (
                f'Daily rolling sum of mean catchment rainfall ({window_size_daily}-days) over {Lake}. '
                f'Coordinates identify whether peaks exceed thresholds ({lower_threshold_daily}/{higher_threshold_daily} mm).'
            ),
            'cumulative_rainfall_up_to_peaks': (
                f'Daily cumulative rainfall up to peaks of {window_size_daily}-day rolling sum windows over {Lake}. '
                f'Coordinates identify whether segments occur during events and/or exceed {lower_threshold_daily} mm threshold.'
            )
        }
    },
    
    'dea': {
        'source': 'Geoscience Australia Landsat Waterbodies Collection 3',
        'units': 'km²',
        'long_name': 'Daily lake extent and surface area',
        'description': f'Daily lake extent (% wet pixels) and surface area (km²) for {Lake}'
    },
    
    'station': {
        'units': {
            'rainfall': 'mm',
            'runoff': 'km³'
        },
        'source': {
            'rainfall': 'BOM IDCJAC0009 product',
            'runoff': 'Water Data Online'
        },
        'long_name': {
            'rainfall': 'Daily rainfall at catchment stations',
            'runoff': 'Daily runoff at catchment stations'
        },
        'description': {
            'rainfall': f'Daily rainfall observations from gauge stations within the {Lake} catchment area.',
            'runoff': f'Daily runoff observations from gauge stations within the {Lake} catchment area.'
        }
    }
}

#-----------------------------------------------------------------------------------------------------------------------------------
#### ========= Functions: Prepare data =========
def normalise_latlon(ds, lat_names=('lat', 'latitude'), lon_names=('lon', 'longitude')):
    """
    Standardize latitude and longitude coordinate names to 'lat' and 'lon'.
    
    Searches for common variations of latitude and longitude coordinate names
    and renames them to standard 'lat' and 'lon' for consistency across datasets.
    
    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset or DataArray with latitude/longitude coordinates.
    lat_names : tuple of str, optional
        Possible names for latitude coordinate. Default is ('lat', 'latitude').
    lon_names : tuple of str, optional
        Possible names for longitude coordinate. Default is ('lon', 'longitude').
    
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Dataset with standardized coordinate names 'lat' and 'lon'.    
    """
    # Build mapping of found coordinate names to standard names
    rename_map = {}
    
    # Check for latitude variations
    for name in lat_names:
        if name in ds.coords or name in ds.dims:
            rename_map[name] = 'lat'
            break
    
    # Check for longitude variations
    for name in lon_names:
        if name in ds.coords or name in ds.dims:
            rename_map[name] = 'lon'
            break
    
    # Apply renaming if any coordinates were found
    if rename_map:
        ds = ds.rename(rename_map)
    
    return ds


def ensure_time_coord(ds, time_name='time'):
    """
    Convert time coordinate to datetime format and normalize to midnight.
    
    Ensures the time coordinate is in datetime64 format and sets the time-of-day
    to 00:00:00, which is useful for daily data to avoid time-of-day inconsistencies.
    
    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset or DataArray with a time coordinate.
    time_name : str, optional
        Name of the time coordinate to normalize. Default is 'time'.
    
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Dataset with normalized time coordinate (unchanged if time_name not found).
    """
    if time_name in ds.coords:
        ds = ds.copy()
        # Convert to datetime and set time-of-day to midnight for daily data consistency
        ds[time_name] = pd.to_datetime(ds[time_name].values).normalize()
    return ds


def assign_mask_with_check(ds, mask_da, mask_name='catchment_mask', verbose=False):
    """
    Assign a spatial mask to a dataset with coordinate alignment validation.
    
    Checks if the mask's lat/lon coordinates match the dataset's coordinates.
    If coordinates are slightly misaligned (within tolerance), reindexes the mask
    to match. Raises an error if coordinates are incompatible.
    
    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset with 'lat' and 'lon' coordinates.
    mask_da : xarray.DataArray
        Mask DataArray with 'lat' and 'lon' coordinates to assign to ds.
    mask_name : str, optional
        Name for the mask coordinate in the output. Default is 'catchment_mask'.
    verbose : bool, optional
        If True, prints information about coordinate reindexing. Default is False.
    
    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Dataset with mask assigned as a coordinate.
    
    Raises
    ------
    ValueError
        If lat/lon lengths don't match or if coordinate values differ beyond tolerance.
    
    Notes
    -----
    Tolerance for coordinate matching is set to 10% of the median coordinate spacing.
    This allows for small numerical differences while catching genuine misalignments.
    """
    # Check that lat/lon dimensions have matching lengths
    if len(ds['lat']) != len(mask_da['lat']) or len(ds['lon']) != len(mask_da['lon']):
        raise ValueError('Latitude or longitude lengths do not match')
    
    # Check for exact coordinate match
    lat_exact = np.array_equal(ds['lat'].values, mask_da['lat'].values)
    lon_exact = np.array_equal(ds['lon'].values, mask_da['lon'].values)
    
    if lat_exact and lon_exact:
        # Perfect match - use mask as-is
        mask_checked = mask_da
    else:
        # Check if coordinates are close (within tolerance)
        lat_tolerance = np.median(np.diff(ds.lat)) * 0.1
        lon_tolerance = np.median(np.diff(ds.lon)) * 0.1
        lat_close = np.allclose(ds['lat'].values, mask_da['lat'].values, atol=lat_tolerance)
        lon_close = np.allclose(ds['lon'].values, mask_da['lon'].values, atol=lon_tolerance)
    
        if lat_close and lon_close:
            # Slight mismatch - reindex to align coordinates
            mask_checked = mask_da.reindex_like(ds, method='nearest')
            if verbose:
                print('Mask reindexed to match dataset coordinates')
        else:
            # Coordinates are incompatible
            raise ValueError('Coordinates are incompatible: same length but values differ beyond tolerance')
    
    # Assign mask as coordinate
    return ds.assign_coords({mask_name: (('lat', 'lon'), mask_checked.values)})


def add_attrs(dtype, ds, var_name, attrs):
    """
    Add standardized metadata attributes to a dataset variable.
    
    Extracts and applies appropriate metadata (source, units, long_name, description)
    from an attributes dictionary based on the data type. Handles different attribute
    structures for AGCD gridded data, DEA satellite data, and station observations.
    
    Parameters
    ----------
    dtype : str
        Data type: 'agcd' (gridded), 'dea' (satellite), or 'station' (observations).
    ds : xarray.Dataset
        Dataset containing the variable to add attributes to.
    var_name : str
        Name of the variable to add attributes to.
    attrs : dict
        Nested dictionary containing metadata for each data type.
        Expected structure varies by dtype (see Notes).
    
    Returns
    -------
    xarray.Dataset
        Dataset with updated variable attributes.
    
    Raises
    ------
    KeyError
        If var_name is not found in dataset (for AGCD data, also checks for 'precip').
    
    Notes
    -----
    For 'agcd' data, if var_name is not found but 'precip' exists, the function
    automatically renames 'precip' to var_name before adding attributes.
    
    Expected attrs structure:
    - agcd/station: attrs[dtype]['units'][var_name], attrs[dtype]['long_name'][var_name]
    - dea: attrs[dtype]['units'], attrs[dtype]['long_name'] (no var_name indexing)
    """
    # Extract source and units based on data type structure
    if dtype in ('agcd', 'dea'):
        source = attrs[dtype]['source']
        units = attrs[dtype]['units']
    elif dtype == 'station':
        source = attrs[dtype]['source'][var_name]
        units = attrs[dtype]['units'][var_name]
    
    # Extract long_name and description based on data type structure
    if dtype in ('agcd', 'station'):
        long_name = attrs[dtype]['long_name'][var_name]
        description = attrs[dtype]['description'][var_name]
    elif dtype == 'dea':
        long_name = attrs[dtype]['long_name']
        description = attrs[dtype]['description']
    
    # For AGCD data, handle 'precip' rename if needed
    if dtype == 'agcd':
        if var_name not in ds.data_vars:
            if 'precip' in ds.data_vars:
                ds = ds.rename({'precip': var_name})
            else:
                raise KeyError(f'Neither {var_name} nor "precip" found in dataset')
    
    # Apply attributes to variable
    da = ds[var_name]
    da.attrs.update({
        'source': source,
        'units': units,
        'long_name': long_name,
        'description': description
    })
    
    return ds


def filter_out_empty_stations(ds, station_type):
    """
    Remove stations that contain only NaN values across all time steps.
    
    Filters out stations with no valid data for a given station type (e.g., rainfall
    or runoff). Removes the stations from all associated variables and coordinates.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing station-based variables.
    station_type : str
        Type of station: 'rainfall' or 'runoff'. The function looks for a station
        dimension named '{station_type}_station' and a reference variable named
        'station_{station_type}' to determine which stations have data.
    
    Returns
    -------
    xarray.Dataset
        Dataset with empty stations (all-NaN over time) removed from all relevant
        variables and coordinates. Returns unchanged if no matching variables found.
    
    Raises
    ------
    KeyError
        If the expected reference variable 'station_{station_type}' is not found.
    ValueError
        If the reference variable does not contain a 'time' dimension.
    """
    dim_name = f'{station_type}_station'
    reference_var = f'station_{station_type}'
    
    # Validate reference variable exists
    if reference_var not in ds:
        raise KeyError(f'Reference variable "{reference_var}" not found in dataset')
    
    # Ensure reference variable has time dimension
    if 'time' not in ds[reference_var].dims:
        raise ValueError(f'Reference variable "{reference_var}" must have a time dimension')
    
    # Identify stations with at least one non-NaN value
    stations_with_data = ~ds[reference_var].isnull().all(dim='time')
    
    # Find all variables and coordinates that include the station dimension
    variables_to_filter = [var for var in ds.data_vars if dim_name in ds[var].dims]
    coords_to_filter = [coord for coord in ds.coords if dim_name in ds[coord].dims or coord == dim_name]
    
    # Return unchanged if nothing to filter
    if not variables_to_filter and not coords_to_filter:
        return ds
    
    # Filter all relevant variables and coordinates
    filtered_vars = {
        var: ds[var].sel({dim_name: stations_with_data})
        for var in variables_to_filter + coords_to_filter
    }
    
    # Drop original variables and reassign filtered versions
    ds_filtered = ds.drop_vars(variables_to_filter + coords_to_filter)
    ds_filtered = ds_filtered.assign(**filtered_vars)
    
    return ds_filtered


def read_driver_files(file_dir, driver):
    """
    Read ENSO/ IPO time series files from a directory and combine into a single DataArray.
    
    Reads multiple NetCDF files containing ENSO/ IPO indices (e.g., Niño 3.4, ONI/ ERSSTV5, COBE ) and
    combines them along a new 'enso_indices' or 'ipo_indices' dimension for easy comparison.
    
    Parameters
    ----------
    file_dir : str
        Glob pattern matching ENSO/IPO  NetCDF files (e.g., 'path/to/enso_*.nc').
    driver : str
        Climate driver type, either 'enso' or 'ipo'.
    
    Returns
    -------
    xarray.DataArray
        DataArray with dimensions (time, enso_indices/ ipo_indices). The enso_indices/ ipo_indices
        coordinate contains labels from each file's variable name. Includes metadata attributes
        (units, source, url, long_name, description).
    
    Raises
    ------
    FileNotFoundError
        If no files matching the pattern are found.
    
    Notes
    -----
    Files are read with dask chunking along the time dimension for memory efficiency.
    If an enso file's variable has no name, it is assigned a default name 'da{i}'.
    """
    # Find all matching files
    file_paths = glob.glob(file_dir)
    
    if not file_paths:
        raise FileNotFoundError(f'No {driver} files found matching pattern: {file_dir}')
    	
    # Read each file and extract the main variable
    indices_list = []
    indices_names = []
    for i, path in enumerate(file_paths):
        ds = xr.open_dataset(path, chunks={'time': dask_chunks})
        var_name = list(ds.data_vars)[0]
        da_var = ds[var_name]    
        indices_list.append(da_var)
        if driver == 'ipo':       
            index_name = Path(path).name.replace('tpi.timeseries.', '').replace('.nc', '')
        else:
            index_name = da_var.name if da_var.name is not None else f'da{i+1}'
        indices_names.append(index_name)
    
    # Combine into single DataArray with enso_indices dimension
    dim_name = f'{driver}_indices'
    da_combined = xr.concat(indices_list, dim=dim_name, join='outer')
    da_combined = da_combined.assign_coords(**{dim_name: np.array(indices_names, dtype=object)})
    da_combined = da_combined.rename(driver)
    
    # Add metadata attributes
    da_combined.attrs.update({
        'long_name': f'Monthly {driver} indices',
        'source': 'NOAA',
        'units': '°C'})
    
    if driver == 'enso':
        da_combined.attrs.update({
            'description': (
                'Combined ENSO dataset containing multiple sea surface temperature anomaly indices. '
                'Includes Niño 3.4 (ERSSTv5), Niño 3.4 (HadISST), and Oceanic Niño Index (ONI) from NOAA CPC.'),
            'url': 'https://psl.noaa.gov/data/timeseries/month/'
        })
    else:
        da_combined.attrs.update({
            'description': (
                'Combined IPO dataset containing multiple sea surface temperature anomaly indices. '
                'Includes Tripole Index (TPI) from ERSSTv5, HadISST, and COBE datasets.')
        })
        
    return da_combined







def resample_drivers_monthly_to_daily(da, dim='time', method='ffill'):
    """
    Resample monthly climate driver data to daily resolution.
    
    Converts monthly time series (e.g., ENSO, IPO indices) to daily by forward-filling
    or back-filling values across each month. Adds an extra month at the end to ensure
    proper resampling of the final month, then removes the extra days.
    
    Parameters
    ----------
    da : xarray.DataArray
        Monthly DataArray to resample.
    dim : str, optional
        Name of the time dimension. Default is 'time'.
    method : str or None, optional
        Resampling method: 'ffill' (forward fill), 'bfill' (backward fill), or None
        (leaves values at start of month only). Default is 'ffill'.
    
    Returns
    -------
    xarray.DataArray
        Daily resolution DataArray covering the same time period as the input.
    
    Raises
    ------
    ValueError
        If method is not 'ffill', 'bfill', or None.
    
    Notes
    -----
    The function temporarily extends the time series by one month to ensure the last
    month resamples correctly, then removes the extra days. Forward fill ('ffill') is
    typically used for climate indices where each month's value applies to all days
    in that month.
    """
    # Extend time series by one month to ensure proper resampling
    extra_time = pd.Timestamp(da[dim].max().values) + pd.DateOffset(months=1)
    da_extended = da.reindex({dim: list(da[dim].values) + [np.datetime64(extra_time)]})
    
    # Resample to daily resolution using specified method
    if method == 'ffill':
        da_daily = da_extended.resample({dim: '1D'}).ffill()
    elif method == 'bfill':
        da_daily = da_extended.resample({dim: '1D'}).bfill()
    elif method is None:
        da_daily = da_extended.resample({dim: '1D'})
    else:
        raise ValueError(f'Unsupported resample method: "{method}". Use "ffill", "bfill", or None')
    
    # Remove the extra days added for resampling
    da_daily = da_daily.isel({dim: slice(0, -1)})
    
    return da_daily



def find_dea_event_max_coord(ds):
    """
    Create a boolean coordinate indicating the maximum lake size point of each event.
    
    For each detected lake event, identifies the time step where lake size reaches
    its maximum and returns a boolean array marking these peak times.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing:
            - 'dea': lake variables with 'lake_variable' dimension including 'Size'
            - 'dea_events': event ID labels for each time step
    
    Returns
    -------
    xarray.DataArray
        Boolean coordinate along the time dimension: True at the maximum of each
        event, False otherwise.
    
    Raises
    ------
    KeyError
        If 'dea' or 'dea_events' are missing from dataset, or if 'dea' does not
        contain 'Size' in the lake_variable dimension.
    """
    # Validate required variables
    required_vars = ['dea', 'dea_events']
    missing = [v for v in required_vars if v not in ds]
    if missing:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing)}')
    
    if 'Size' not in ds['dea'].lake_variable:
        raise KeyError('"dea" exists but does not contain "Size" in lake_variable dimension')
    
    # Extract lake size time series
    lake_size = ds['dea'].sel(lake_variable='Size')
    
    # Find the time of maximum lake size for each event
    max_dates = lake_size.groupby('dea_events').apply(
        lambda x: x.time.isel(time=x.compute().argmax(dim='time'))
    )
    
    # Create boolean array marking event peak times
    event_max = ds.time.isin(max_dates)
    
    return event_max


def calculate_event_offset_coord(ds, time_unit):
    """
    Calculate the time offset of each point relative to its event's peak.
    
    For each lake event, computes how many days (or months) each time step is from
    the event's peak lake size. Negative values indicate times before the peak,
    positive values indicate times after the peak.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing:
            - 'dea': lake variables with 'lake_variable' dimension including 'Size'
            - 'dea_events': event ID labels for each time step
    time_unit : str
        Unit for offset calculation: 'daily' or 'monthly'.
    
    Returns
    -------
    xarray.DataArray
        Offset (in days or months) of each time point relative to its event's peak.
        NaN for times not part of any event. Named 'dea_offset'.
    
    Raises
    ------
    KeyError
        If 'dea' or 'dea_events' are missing, or if 'Size' is not in lake_variable.
    ValueError
        If time_unit is not 'daily' or 'monthly'.    
    """
    # Validate required variables
    required_vars = ['dea', 'dea_events']
    missing = [v for v in required_vars if v not in ds]
    if missing:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing)}')
    
    if 'Size' not in ds['dea'].lake_variable:
        raise KeyError('"dea" exists but does not contain "Size" in lake_variable dimension')
    
    if time_unit not in ['daily', 'monthly']:
        raise ValueError('time_unit must be "daily" or "monthly"')
    
    # Extract lake size time series
    lake_size = ds['dea'].sel(lake_variable='Size')
    
    # Compute offset for each event relative to its peak
    def compute_offset(group, time_unit=time_unit):
        # Find the timestamp at the group's maximum lake size
        peak_date = group.time.isel(time=group.compute().argmax(dim='time'))
        
        # Calculate offset in requested units
        if time_unit == 'daily':
            return (group.time - peak_date) / np.timedelta64(1, 'D')
        else:  # monthly
            return (group.time.astype('datetime64[M]') - peak_date.astype('datetime64[M]')) / np.timedelta64(1, 'M')
    
    # Apply offset calculation to each event
    offset_da = lake_size.groupby('dea_events').map(compute_offset)
    offset_da.name = 'dea_offset'
    
    # Reindex to full time dimension (fills NaN for non-event times)
    offset_da = offset_da.reindex(time=ds.time)
    offset_da = offset_da.reset_coords(drop=True)
    
    return offset_da
    

def calculate_cumulative_rainfall_per_event(ds, variable):
    """
    Calculate cumulative rainfall for each lake event from start to peak.
    
    For each event, computes the cumulative sum of rainfall from the event start
    up to and including the peak lake size (where dea_offset = 0). Times after
    the peak are excluded.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing:
            - 'dea_events': event ID labels for each time step
            - 'dea_offset': time offset relative to event peak (≤0 before peak)
            - rainfall variable (e.g., 'mean_catchment_rainfall')
    variable : str
        Name of the rainfall variable in ds to use for cumulative calculation.
    
    Returns
    -------
    xarray.DataArray
        Cumulative rainfall for each event up to its peak, sorted chronologically
        by time. Only includes times where dea_offset ≤ 0.
    
    Raises
    ------
    KeyError
        If 'dea_events', 'dea_offset', or the specified variable is missing.
    """
    # Validate required variables
    required_vars = ['dea_events', 'dea_offset', variable]
    missing = [v for v in required_vars if v not in ds]
    if missing:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing)}')
    
    # Extract rainfall variable
    rainfall = ds[variable]
    
    # Mask to include only times up to event peak (offset ≤ 0)
    event_offset_rise_mask = ds.dea_offset <= 0
    
    # Calculate cumulative sum for each event separately
    rainfall_cumsum = (
        rainfall
        .where(event_offset_rise_mask, drop=True)
        .groupby('dea_events')
        .cumsum(dim='time')
    )
    
    return rainfall_cumsum.sortby('time')


def filter_cumulative_rainfall(ds):
    """
    Filter cumulative rainfall to remove low-intensity periods at event edges.
    
    For each lake event, filters the cumulative rainfall to include only periods
    where daily rainfall exceeds 1% of the maximum daily rainfall. Trims low-intensity
    periods from the start and end of each event while preserving any interior periods
    (even if below threshold). Also creates an adjusted offset coordinate for the
    filtered periods.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing:
            - 'cumulative_rainfall_per_event': cumulative rainfall per event
            - 'mean_catchment_rainfall': daily rainfall
            - 'dea_events': event ID labels
            - 'dea_offset': time offset relative to event peak
    
    Returns
    -------
    xarray.DataArray
        Filtered cumulative rainfall with a new 'dea_offset_filtered' coordinate
        adjusted so the peak of each filtered period is at zero. Sorted chronologically.
    
    Raises
    ------
    KeyError
        If required variables are missing from dataset.
    
    Notes
    -----
    The 1% threshold helps focus on significant rainfall periods. The function trims
    edge periods below threshold but preserves interior gaps, maintaining the overall
    event structure while removing insignificant lead-in and tail-off periods.
    """
    # Validate required variables
    required_vars = [
        'cumulative_rainfall_per_event',
        'mean_catchment_rainfall',
        'dea_events',
        'dea_offset'
    ]
    missing = [v for v in required_vars if v not in ds]
    if missing:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing)}')
    
    # Extract variables
    precip_cum = ds['cumulative_rainfall_per_event']
    precip = ds['mean_catchment_rainfall']
    
    # Create mask for valid rainfall periods (above threshold and within rise period)
    threshold_value = precip.max() / 100  # 1% of maximum daily rainfall
    mask_valid = (precip > threshold_value) & (ds.dea_offset <= 0)
    
    # Apply mask (invalid periods become NaN)
    precip_cum_masked = precip_cum.where(mask_valid).compute()
    
    def trim_edges(group):
        """Trim NaN values from edges while preserving interior NaNs."""
        non_nan_indices = np.where(~np.isnan(group.values))[0]
        
        if len(non_nan_indices) == 0:
            return group.isel(time=slice(0, 0))  # Return empty if all NaN
        
        # Find first and last valid values
        first_valid = non_nan_indices[0]
        last_valid = non_nan_indices[-1]
        
        # Trim from both ends
        return group.isel(time=slice(first_valid, last_valid + 1))
    
    # Trim edge NaNs for each event
    precip_filtered = precip_cum_masked.groupby('dea_events').map(trim_edges)
    
    def adjust_offset(group):
        """Adjust offset so peak of filtered period is at zero."""
        return group.assign_coords(
            dea_offset_filtered=('time', (group.dea_offset - group.dea_offset.max()).values)
        )
    
    # Adjust offset coordinates for filtered periods
    precip_filtered = precip_filtered.groupby('dea_events').map(adjust_offset)
    
    return precip_filtered.sortby('time')


def identify_peaks_and_troughs(series_values, distance_val):
    """
    Identify and clean local peaks and troughs in a time series.
    
    Finds local maxima (peaks) and minima (troughs) in a 1D time series, then
    cleans the results to ensure proper alternation: trough-peak-trough-peak.
    Removes consecutive peaks or troughs by keeping the more extreme value.
    Forces the series to start with a trough.
    
    Parameters
    ----------
    series_values : xarray.DataArray
        1D time series with a 'time' dimension to analyze.
    distance_val : int
        Minimum distance (in time steps) between consecutive peaks or troughs.
    
    Returns
    -------
    dict
        Dictionary with keys:
            - 'peaks': np.ndarray of indices for cleaned local maxima
            - 'troughs': np.ndarray of indices for cleaned local minima
        Returns empty arrays if no peaks or troughs are found.
    
    Raises
    ------
    KeyError
        If series_values does not have a 'time' dimension.
    ValueError
        If distance_val is not positive.
    
    Notes
    -----
    Prominence threshold is set to the 25th percentile of the series values,
    helping to identify significant peaks while filtering out noise.
    """
    # Validate inputs
    if 'time' not in series_values.dims:
        raise KeyError('series_values must have a "time" dimension')
    
    if distance_val <= 0:
        raise ValueError('distance_val must be a positive integer')
    
    # Find initial peaks and troughs using prominence threshold
    prominence_val = np.percentile(series_values.dropna(dim='time'), 25)
    
    peaks, _ = find_peaks(series_values.values, prominence=prominence_val, distance=distance_val)
    troughs, _ = find_peaks(-series_values.values, prominence=prominence_val, distance=distance_val)
    
    # Return empty if no peaks or troughs found
    if len(peaks) == 0 or len(troughs) == 0:
        return {'troughs': np.array([]), 'peaks': np.array([])}
    
    # Ensure series starts with a trough (remove first peak if needed)
    if peaks[0] < troughs[0]:
        peaks = peaks[1:]
    
    # Clean consecutive troughs (keep the lower one)
    cleaned_troughs = [troughs[0]]
    for i in range(len(troughs) - 1):
        t0 = troughs[i]
        t1 = troughs[i + 1]
        
        # Check if there's a peak between these two troughs
        in_between_peaks = peaks[(peaks > t0) & (peaks < t1)]
        
        if in_between_peaks.any():
            # Peak exists between troughs - add next trough
            cleaned_troughs.append(t1)
        else:
            # No peak between - keep only the lower trough
            if series_values[t0] >= series_values[t1]:
                # Next trough is lower - replace current with next
                del cleaned_troughs[-1]
                cleaned_troughs.append(t1)
            # Otherwise keep current trough (do nothing)
    
    # Clean consecutive peaks (keep the higher one)
    cleaned_peaks = [peaks[0]]
    for i in range(len(peaks) - 1):
        p0 = peaks[i]
        p1 = peaks[i + 1]
        
        # Check if there's a trough between these two peaks
        in_between_troughs = troughs[(troughs > p0) & (troughs < p1)]
        
        if in_between_troughs.any():
            # Trough exists between peaks - add next peak
            cleaned_peaks.append(p1)
        else:
            # No trough between - keep only the higher peak
            if series_values[p0] <= series_values[p1]:
                # Next peak is higher - replace current with next
                del cleaned_peaks[-1]
                cleaned_peaks.append(p1)
            # Otherwise keep current peak (do nothing)
    
    return {
        'troughs': np.array(cleaned_troughs),
        'peaks': np.array(cleaned_peaks)
    }


def classify_peak_segments(rolling_sum_da, peak_and_trough_ids, threshold1, threshold2):
    """
    Classify rainfall peak segments based on whether they exceed thresholds.
    
    Divides a rolling sum time series into segments between troughs, then creates
    boolean coordinates indicating whether each segment's peak exceeds the given
    thresholds. Each time point in a segment receives the same boolean value based
    on that segment's peak magnitude.
    
    Parameters
    ----------
    rolling_sum_da : xarray.DataArray
        Rolling sum time series with 'time' dimension.
    peak_and_trough_ids : dict
        Dictionary with keys 'peaks' and 'troughs' containing arrays of indices.
    threshold1 : float
        Lower rainfall threshold (mm).
    threshold2 : float
        Higher rainfall threshold (mm).
    
    Returns
    -------
    xarray.DataArray
        Rolling sum DataArray with two new boolean coordinates:
            - 'rolling_peak_above_{threshold1}': True where segment peak > threshold1
            - 'rolling_peak_above_{threshold2}': True where segment peak > threshold2
    
    Notes
    -----
    Segments are defined as the period between consecutive troughs. Times before
    the first trough and after the last trough are assigned False for both thresholds.
    """
    # Extract peak and trough indices
    peak_ids = peak_and_trough_ids['peaks']
    trough_ids = peak_and_trough_ids['troughs']
    
    # Prepare rolling sum for segmentation
    rolling_sum_segmented = rolling_sum_da.reset_coords(drop=True)
    
    # Create segment boundaries using troughs
    trough_ids_plus_start = np.concatenate(([0], trough_ids))
    trough_ids_plus_end = np.concatenate((trough_ids, [len(rolling_sum_segmented)]))
    
    # Calculate length of each segment
    segment_lengths = trough_ids_plus_end - trough_ids_plus_start
    
    # Create segment IDs by repeating each ID for its segment length
    segment_id_values = np.repeat(np.arange(len(segment_lengths)), segment_lengths)
    
    # Assign segment IDs as coordinate
    segment_ids = xr.DataArray(
        segment_id_values,
        coords={'time': rolling_sum_segmented.time},
        dims=['time']
    )
    
    # Validate segment assignment
    assert segment_ids.time.size == rolling_sum_segmented.time.size, \
        f"Size mismatch! segment_ids: {segment_ids.time.size}, " \
        f"rolling_sum: {rolling_sum_segmented.time.size}"
    
    rolling_sum_segmented = rolling_sum_segmented.assign_coords(segment_id=segment_ids)
    
    # Check if peaks exceed thresholds
    above_thresh1 = rolling_sum_da[peak_ids] > threshold1
    above_thresh2 = rolling_sum_da[peak_ids] > threshold2
    
    # Add False for times before first and after last trough
    above_thresh1_extended = np.concatenate(([False], above_thresh1, [False]))
    above_thresh2_extended = np.concatenate(([False], above_thresh2, [False]))
    
    # Create DataArrays with segment_id dimension
    above_thresh1_da = xr.DataArray(
        above_thresh1_extended,
        dims=['segment_id'],
        coords={'segment_id': np.arange(len(above_thresh1_extended), dtype=float)}
    )
    
    above_thresh2_da = xr.DataArray(
        above_thresh2_extended,
        dims=['segment_id'],
        coords={'segment_id': np.arange(len(above_thresh2_extended), dtype=float)}
    )
    
    # Map segment boolean values to all times in each segment
    peak_above_thresh1 = above_thresh1_da.sel(segment_id=rolling_sum_segmented.segment_id)
    peak_above_thresh2 = above_thresh2_da.sel(segment_id=rolling_sum_segmented.segment_id)
    
    # Assign threshold coordinates to original rolling sum
    return rolling_sum_da.assign_coords({
        f'rolling_peak_above_{threshold1}': peak_above_thresh1,
        f'rolling_peak_above_{threshold2}': peak_above_thresh2
    })


def calculate_cumulative_rainfall_for_window(ds, peak_ids, window, time_unit):
    """
    Calculate cumulative rainfall for windows leading up to rolling sum peaks.
    
    For each peak in the rolling sum, extracts a window of rainfall data leading
    up to the peak and computes the cumulative sum. Also identifies whether each
    window is associated with a lake-filling event by checking for event peaks
    within or shortly after the rainfall window.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing:
            - 'mean_catchment_rainfall': daily or monthly rainfall
            - 'dea_events': event ID labels
            - 'dea_event_max': boolean marking event peaks
    peak_ids : array-like
        Indices of peaks in the rolling sum time series.
    window : int
        Size of window (in days or months) to extract before each peak.
    time_unit : str
        Unit of time: 'days' or 'months'.
    
    Returns
    -------
    xarray.DataArray
        Cumulative rainfall with coordinates:
            - 'cum_window_is_event': event ID if window associated with event, else NaN
            - 'cum_window_{window}_{time_unit}': position within window (1, 2, ..., n)
    
    Raises
    ------
    KeyError
        If required variables are missing from dataset.
    ValueError
        If time_unit is invalid, multiple event peaks found in one window, or
        if events are missing from final output.
    
    Notes
    -----
    The function checks for event peaks in two stages:
    1. Within the rainfall window (start to peak)
    2. Extended window (start to peak + window) to catch delayed events
    
    This two-stage approach handles cases where lake-filling events occur slightly
    after the rainfall peak, which can happen with smaller events.
    """
    # Validate inputs
    required_vars = ['mean_catchment_rainfall', 'dea_event_max', 'dea_events']
    missing_vars = [v for v in required_vars if v not in ds]
    if missing_vars:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing_vars)}')
    
    if time_unit not in ['months', 'days']:
        raise ValueError('time_unit must be "months" or "days"')
    
    # Extract event IDs at peak times only
    event_max = ds.dea_events.where(ds.dea_event_max)
    
    # Process each peak
    cum_list = []
    for i, peak_idx in enumerate(peak_ids):
        
        # Extract rainfall window leading up to peak
        start_idx = peak_idx - (window - 1)
        window_data = ds['mean_catchment_rainfall'].isel(time=slice(start_idx, peak_idx + 1))
        
        # Compute cumulative sum
        cumsum_data = window_data.cumsum(dim='time')
        
        # Check for event peaks within the rainfall window
        event_window_pre = event_max.isel(time=slice(start_idx, peak_idx + 1))
        
        if event_window_pre.notnull().any():
            # Event peak found in rainfall window
            is_event = np.unique(event_window_pre.dropna(dim='time'))
            
            if is_event.size > 1:
                raise ValueError('Multiple event peaks found in a single window')
            
            event_coord = xr.full_like(cumsum_data, fill_value=is_event.item())
            cumsum_data = cumsum_data.assign_coords({'cum_window_is_event': event_coord})
        
        else:
            # No event in rainfall window - check extended window after peak
            event_window_post = event_max.isel(time=slice(start_idx, peak_idx + window + 1))
            
            if event_window_post.notnull().any():
                # Event found in extended window
                is_event = np.unique(event_window_post.dropna(dim='time'))
                
                if is_event.size > 1:
                    raise ValueError('Multiple event peaks found in extended window')
                
                event_coord = xr.full_like(cumsum_data, fill_value=is_event.item())
                cumsum_data = cumsum_data.assign_coords({'cum_window_is_event': event_coord})
            
            else:
                # No event found - mark as NaN
                cumsum_data = cumsum_data.assign_coords({'cum_window_is_event': event_window_pre})
        
        # Add position-within-window coordinate
        coord_name = f'cum_window_{window}_months' if time_unit == 'months' else f'cum_window_{window}_days'
        cum_coord = xr.DataArray(
            np.arange(1, len(cumsum_data) + 1),
            dims=['time'],
            coords={'time': cumsum_data.time}
        )
        cumsum_data = cumsum_data.assign_coords({coord_name: cum_coord})
        
        cum_list.append(cumsum_data)
    
    # Combine all windows and sort chronologically
    cum_da = xr.concat(cum_list, dim='time').sortby('time')
    
    # Validate that all events were captured
    window_events = np.unique(cum_da.cum_window_is_event.dropna(dim='time'))
    events = np.unique(ds.dea_events.dropna(dim='time'))
    missing_events = np.setdiff1d(events, window_events)
    
    if missing_events.size:
        raise ValueError(
            f'Events missing from output after window extension: {missing_events}'
        )
    
    return cum_da


def resample_lake_dataset_to_monthly(ds_daily):
    """
    Resample a daily lake dataset to monthly resolution.
    
    Aggregates daily data to monthly using appropriate methods for each variable type:
    rainfall and runoff are summed, lake variables (DEA) use maximum values, and
    static variables are preserved unchanged.
    
    Parameters
    ----------
    ds_daily : xarray.Dataset
        Daily dataset containing rainfall, runoff, and/or DEA lake variables.
    
    Returns
    -------
    xarray.Dataset
        Monthly resampled dataset with:
            - Rainfall and runoff variables summed
            - DEA-related variables (lake size, events) maximized
            - Event coordinates (dea_events, dea_event_max) preserved
            - Static variables unchanged
            - Attributes updated to reflect monthly resolution
    
    Raises
    ------
    ValueError
        If dataset contains no rainfall, runoff, or DEA variables to resample.
    
    Notes
    -----
    Time resampling uses '1MS' (month start) frequency. DEA variables use maximum
    to capture peak lake conditions within each month.
    """
    ds_monthly = xr.Dataset()
    
    # Identify variables with time dimension
    vars_with_time = [var for var in ds_daily.data_vars if 'time' in ds_daily[var].dims]
    
    # Categorize variables by aggregation method
    vars_to_sum = [var for var in vars_with_time 
                   if any(key in var for key in ['rainfall', 'runoff'])]
    vars_to_max = [var for var in vars_with_time if 'dea' in var]
    
    # Validate that there are variables to resample
    if not vars_to_sum and not vars_to_max:
        raise ValueError(
            'Dataset contains no rainfall, runoff, or dea variables to resample'
        )
    
    # Sum rainfall and runoff variables monthly
    for var in vars_to_sum:
        ds_monthly[var] = ds_daily[var].resample(time='1MS').sum()
    
    # Take monthly maximum for DEA variables
    for var in vars_to_max:
        ds_monthly[var] = ds_daily[var].resample(time='1MS').max()
        
        # Preserve event coordinates if present
        if 'dea_events' in ds_daily[var].coords:
            ds_events_monthly = ds_daily[var].coords['dea_events'].resample(time='1MS').max()
            ds_monthly[var] = ds_monthly[var].assign_coords(dea_events=ds_events_monthly)
        
        if 'dea_event_max' in ds_daily[var].coords:
            ds_event_max_monthly = ds_daily[var].coords['dea_event_max'].resample(time='1MS').max()
            ds_monthly[var] = ds_monthly[var].assign_coords(dea_event_max=ds_event_max_monthly)
    
    # Copy static variables unchanged
    static_vars = [var for var in ds_daily.data_vars if 'time' not in ds_daily[var].dims]
    for var in static_vars:
        ds_monthly[var] = ds_daily[var]
    
    # Update attributes to reflect monthly resolution
    for var in ds_monthly.data_vars:
        if 'long_name' in ds_monthly[var].attrs:
            ds_monthly[var].attrs['long_name'] = (
                ds_monthly[var].attrs['long_name'].replace('Daily', 'Monthly')
            )
        if 'description' in ds_monthly[var].attrs:
            ds_monthly[var].attrs['description'] = (
                ds_monthly[var].attrs['description']
                .replace('daily', 'monthly')
                .replace('Daily', 'Monthly')
            )
    
    return ds_monthly
    

def find_rainfall_peaks_near_lake_peaks(ds, percentile=0.8):
    """
    Find the most significant monthly rainfall peaks closest to each lake peak.
    
    Identifies local peaks in monthly rainfall, filters for major peaks above a
    percentile threshold, then finds the closest rainfall peak to each lake event peak.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing:
            - 'dea_event_max': boolean marking lake event peaks
            - 'mean_catchment_rainfall': daily rainfall to aggregate monthly
    percentile : float, optional
        Percentile threshold for selecting significant rainfall peaks (0-1).
        Default is 0.8, keeping the top 20% of peaks.
    
    Returns
    -------
    xarray.DataArray
        Monthly rainfall values at the significant rainfall peaks closest to
        each lake event peak.
    
    Raises
    ------
    KeyError
        If required variables are missing from dataset.
    
    Notes
    -----
    Local peaks are identified where rainfall increases then decreases (derivative
    changes from positive to negative). This avoids spurious peaks from noise.
    """
    # Validate inputs
    required_vars = ['dea_event_max', 'mean_catchment_rainfall']
    missing = [v for v in required_vars if v not in ds]
    if missing:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing)}')
    
    # Aggregate rainfall to monthly
    rainfall_monthly = ds['mean_catchment_rainfall'].resample(time='MS').sum()
    
    # Extract dates of lake event peaks
    dea_peak_dates = ds.where(ds.dea_event_max, drop=True).time
    
    # Identify local peaks in monthly rainfall using derivative method
    rainfall_diff = rainfall_monthly.diff(dim='time')
    peak_mask = (rainfall_diff > 0) & (rainfall_diff.shift(time=-1) < 0)
    rainfall_peaks = rainfall_monthly.where(peak_mask)
    
    # Filter for significant peaks above percentile threshold
    threshold = rainfall_peaks.quantile(percentile)
    significant_peaks = rainfall_peaks.where(rainfall_peaks > threshold).dropna(dim='time')
    
    # Find the nearest significant rainfall peak to each lake peak
    sig_peak_dates = significant_peaks.time
    closest_dates = sig_peak_dates.sel(time=dea_peak_dates, method='nearest')
    
    return rainfall_monthly.sel(time=closest_dates)
    

def calculate_sum_over_given_windows(ds, da, window_size):
    """
    Sum values over pre-defined fixed-length windows while preserving event information.
    
    Groups a DataArray into consecutive windows based on window coordinates from ds,
    then sums values within each window. Preserves event classification and threshold
    information by aggregating these coordinates appropriately.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing window coordinates and classification flags:
            - 'cum_window_{window_size}_days': window position coordinate
            - 'cum_window_is_event': event classification for each window
            - 'rolling_peak_above_{threshold}': threshold flags
    da : xarray.DataArray
        Input DataArray to sum over windows. Must have 'time' dimension aligned with ds.
    window_size : int
        Number of time steps per window (must be positive).
    
    Returns
    -------
    xarray.DataArray
        Windowed sums with coordinates:
            - 'time': first timestamp of each window
            - 'above_threshold': sum of threshold flags in window
            - 'dea_events': mean event ID in window (NaN if no event)
        Includes description attribute noting window size.
    
    Raises
    ------
    KeyError
        If required coordinates are missing from ds.
    ValueError
        If da lacks 'time' dimension or window_size is not positive.
    
    Notes
    -----
    Windows are defined by the 'cum_window' coordinate from ds. The function drops
    times where cum_window is NaN, then groups remaining times into windows of size
    window_size for aggregation.
    """
    # Validate required coordinates
    required_coords = [
        f'cum_window_{window_size}_days',
        'cum_window_is_event',
        f'rolling_peak_above_{lower_threshold_daily}'
    ]
    missing = [coord for coord in required_coords if coord not in ds]
    if missing:
        raise KeyError(f'Required coordinates missing from ds: {", ".join(missing)}')
    
    if 'time' not in da.dims:
        raise ValueError('Input DataArray must have a "time" dimension')
    
    if window_size <= 0:
        raise ValueError(f'window_size must be positive, got {window_size}')
    
    # Assign window coordinates from dataset to DataArray
    da = da.assign_coords({
        'cum_window': ds[f'cum_window_{window_size_daily}_days'].reset_coords(drop=True),
        'cum_window_is_event': ds['cum_window_is_event'].reset_coords(drop=True),
        'cum_window_above_threshold': ds[f'rolling_peak_above_{lower_threshold_daily}'].reset_coords(drop=True)
    })
    
    # Drop times outside defined windows
    da_dropped_na = da.where(~da.cum_window.isnull(), drop=True)
    
    # Create segment IDs for grouping into windows
    k = window_size
    n = da_dropped_na.sizes['time']
    segment_ids = xr.DataArray(np.arange(n) // k, dims='time')
    
    # Aggregate data and coordinates for each window
    windowed_sum = da_dropped_na.groupby(segment_ids).sum(dim='time')
    windowed_sum_coords_events = da_dropped_na.cum_window_is_event.groupby(segment_ids).mean(dim='time', skipna=True)
    windowed_sum_coords_above_threshold = da_dropped_na.cum_window_above_threshold.groupby(segment_ids).sum(dim='time')
    windowed_sum_coords_time = da_dropped_na.time.groupby(segment_ids).first()
    
    # Assign aggregated coordinates to result
    windowed_sum = windowed_sum.assign_coords({
        'time': windowed_sum_coords_time,
        'above_threshold': windowed_sum_coords_above_threshold,
        'dea_events': windowed_sum_coords_events
    })
    
    # Add metadata and swap to time dimension
    windowed_sum = windowed_sum.assign_attrs(
        description=f'Sum over windows of {window_size} time steps'
    )
    windowed_sum = windowed_sum.swap_dims({'group': 'time'})
    
    return windowed_sum


#### ========= Functions: Plot data =========
def plot_EA_map(FigNo, lake_mask_fine):
    """
    Plot map of Eastern Australia with catchment mask.
    
    Parameters
    ----------
    FigNo : str
        figure number (e.g., 'Fig1').
    lake_mask_fine : xarray.DataArray
        Fine resolution lake mask for boundary contour.
    
    Output
    ------
    Saves PNG: '{output_dir}/{FigNo}_{Lake}_EA_Map.png'
    """
           
    # ========== FILENAME CONSTRUCTION ==========
    filename_parts = [FigNo, Lake, 'EA_Map']
    file_name = '_'.join(filename_parts)
    
    
    # ========== PLOTTING CONFIGURATION ==========
    # Lake mask styling
    lake_contour_color = 'k'
    lake_contour_width = 1
    
    # Gridline styling
    gridline_config = {
        'linewidth': 0.5,
        'color': 'white',
        'alpha': 0.7,
        'linestyle': '--',
        'label_fontsize': 15
    }
    

    # ========== CREATE FIGURE AND PLOT ==========
    fig = plt.figure(figsize=(15, 17))
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Add coastlines
    ax1.coastlines()
    
    # Set longitude and latitude limits (xmin, xmax, ymin, ymax)
    ax1.set_extent([130, 155, -40, -10], crs=ccrs.PlateCarree())

    # Add land feature to the map
    land = cfeature.NaturalEarthFeature(category='physical', name='land', scale='50m', facecolor='white')
    ax1.add_feature(land, edgecolor='black', linewidth=2)

    # Plot lake masks as black contour lines
    ax1.contour(
        lake_mask_fine.lon, lake_mask_fine.lat, lake_mask_fine.values,
        colors=lake_contour_color, linewidths=lake_contour_width,
        transform=ccrs.PlateCarree(), zorder=2
    )
    
    
    # ========== FORMAT MAP ==========
    # Add gridlines
    gl = ax1.gridlines(
        draw_labels=True,
        crs=ccrs.PlateCarree(),
        linewidth=gridline_config['linewidth'],
        color=gridline_config['color'],
        alpha=gridline_config['alpha'],
        linestyle=gridline_config['linestyle']
    )
    
    gl.xlocator = LongitudeLocator()
    gl.ylocator = LatitudeLocator()
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Show only left and bottom labels
    gl.bottom_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': gridline_config['label_fontsize']}
    gl.ylabel_style = {'size': gridline_config['label_fontsize']}
    
       
    # ========== SAVE FIGURE ==========
    plt.savefig(f'{output_dir}/{file_name}.png', bbox_inches='tight')
    plt.close()
    
    
def plot_catchment_map(FigNo, ds, timeframe, lake_mask_coarse, lake_mask_fine):
    """
    Plot map showing rainfall and runoff station locations with lake boundary.
    
    Creates a map with stations active during the specified timeframe. Filters out
    stations with no data in the period and displays remaining stations with their
    IDs on the map and full names in a legend box.
    
    Parameters
    ----------
    FigNo : str
        figure number (e.g., 'Fig1').
    ds : xarray.Dataset
        Dataset containing:
            - 'rainfall_station', 'rainfall_station_lat', 'rainfall_station_lon', 'rainfall_station_name'
            - 'runoff_station', 'runoff_station_lat', 'runoff_station_lon', 'runoff_station_name'
    timeframe : dict
        Dictionary with 'start' and 'end' keys containing year strings for filtering.
    lake_mask_coarse : xarray.DataArray
        Coarse resolution lake mask for filling.
    lake_mask_fine : xarray.DataArray
        Fine resolution lake mask for boundary contour.
    
    Output
    ------
    Saves PNG: '{output_dir}/{FigNo}_{Lake}_Catchment_Map.png'
    
    Raises
    ------
    KeyError
        If required station variables are missing from dataset.
    """
    
    # ========== VALIDATE INPUTS ==========
    required_vars = [
        'rainfall_station', 'rainfall_station_lat', 'rainfall_station_lon', 'rainfall_station_name',
        'runoff_station', 'runoff_station_lat', 'runoff_station_lon', 'runoff_station_name'
    ]
    missing_vars = [v for v in required_vars if v not in ds]
    if missing_vars:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing_vars)}')
    
    
    # ========== DATA PREPARATION ==========
    # Subset to timeframe and remove stations with no data
    trimmed_ds = ds.sel(time=slice(timeframe['start'], timeframe['end']))
    trimmed_ds = filter_out_empty_stations(trimmed_ds, 'rainfall')
    trimmed_ds = filter_out_empty_stations(trimmed_ds, 'runoff')
    
    # Extract rainfall station data
    rainfall_lats = trimmed_ds['rainfall_station_lat'].values
    rainfall_lons = trimmed_ds['rainfall_station_lon'].values
    rainfall_ids = trimmed_ds['rainfall_station'].values
    rainfall_names = trimmed_ds['rainfall_station_name'].values
    
    # Extract runoff station data
    runoff_lats = trimmed_ds['runoff_station_lat'].values
    runoff_lons = trimmed_ds['runoff_station_lon'].values
    runoff_ids = trimmed_ds['runoff_station'].values
    runoff_names = trimmed_ds['runoff_station_name'].values
    
    
    # ========== FILENAME CONSTRUCTION ==========
    filename_parts = [FigNo, Lake, 'Catchment_Map']
    file_name = '_'.join(filename_parts)
    
    
    # ========== PLOTTING CONFIGURATION ==========
    # Station markers (shared)
    station_marker = 'X'
    station_size = 150
    
    # Station-specific colors and label offsets
    station_config = {
        'rainfall': {
            'color': '#336666',
            'label_offset_x': -0.16,
            'label_offset_y': 0.02
        },
        'runoff': {
            'color': '#99cc33',
            'label_offset_x': 0.02,
            'label_offset_y': 0.02
        }
    }
    
    # Lake mask styling
    lake_fill_cmap = ListedColormap(['white', 'white'])
    lake_contour_color = 'k'
    lake_contour_width = 1
    
    # Gridline styling
    gridline_config = {
        'linewidth': 0.5,
        'color': 'white',
        'alpha': 0.7,
        'linestyle': '--',
        'label_fontsize': 15
    }
    
    # Text box configuration
    textbox_config = {
        'position': (0.55, 0.2),
        'fontsize': 15,
        'box_props': dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
    }
    
    # Create station info text for legend box
    rainfall_info = [f'{sid} – {sname}' for sid, sname in zip(rainfall_ids, rainfall_names)]
    runoff_info = [f'{sid} – {sname}' for sid, sname in zip(runoff_ids, runoff_names)]
    
    textbox_content = 'Rainfall Stations:\n' + '\n'.join(rainfall_info)
    textbox_content += '\n\nRunoff Stations:\n' + '\n'.join(runoff_info)
    
    
    # ========== CREATE FIGURE AND PLOT ==========
    fig = plt.figure(figsize=(15, 17))
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Add coastlines
    ax1.coastlines()
    
    # Fill lake area
    ax1.pcolormesh(
        lake_mask_coarse.coords['lon'].values,
        lake_mask_coarse.coords['lat'].values,
        lake_mask_coarse.values,
        cmap=lake_fill_cmap,
        transform=ccrs.PlateCarree(),
        zorder=2
    )
    
    # Add lake boundary contour
    ax1.contour(
        lake_mask_fine.lon, lake_mask_fine.lat, lake_mask_fine.values,
        colors=lake_contour_color, linewidths=lake_contour_width,
        transform=ccrs.PlateCarree(), zorder=2
    )
    
    # Plot rainfall stations
    ax1.scatter(
        rainfall_lons, rainfall_lats,
        marker=station_marker, s=station_size,
        color=station_config['rainfall']['color'],
        transform=ccrs.PlateCarree(), zorder=3
    )
    
    # Add rainfall station ID labels
    for lon, lat, sid in zip(rainfall_lons, rainfall_lats, rainfall_ids):
        ax1.text(
            lon + station_config['rainfall']['label_offset_x'],
            lat + station_config['rainfall']['label_offset_y'],
            sid,
            fontsize=textbox_config['fontsize'],
            color=station_config['rainfall']['color'],
            transform=ccrs.PlateCarree(), zorder=4
        )
    
    # Plot runoff stations
    ax1.scatter(
        runoff_lons, runoff_lats,
        marker=station_marker, s=station_size,
        color=station_config['runoff']['color'],
        transform=ccrs.PlateCarree(), zorder=3
    )
    
    # Add runoff station ID labels
    for lon, lat, sid in zip(runoff_lons, runoff_lats, runoff_ids):
        ax1.text(
            lon + station_config['runoff']['label_offset_x'],
            lat + station_config['runoff']['label_offset_y'],
            sid,
            fontsize=textbox_config['fontsize'],
            color=station_config['runoff']['color'],
            transform=ccrs.PlateCarree(), zorder=4
        )
    
    
    # ========== FORMAT MAP ==========
    # Add gridlines
    gl = ax1.gridlines(
        draw_labels=True,
        crs=ccrs.PlateCarree(),
        linewidth=gridline_config['linewidth'],
        color=gridline_config['color'],
        alpha=gridline_config['alpha'],
        linestyle=gridline_config['linestyle']
    )
    
    # Show only left and bottom labels
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': gridline_config['label_fontsize']}
    gl.ylabel_style = {'size': gridline_config['label_fontsize']}
    
    # Add text box with station names
    ax1.text(
        textbox_config['position'][0],
        textbox_config['position'][1],
        textbox_content,
        transform=ax1.transAxes,
        fontsize=textbox_config['fontsize'],
        verticalalignment='top',
        horizontalalignment='left',
        bbox=textbox_config['box_props'],
        zorder=5
    )
       
    # ========== SAVE FIGURE ==========
    plt.savefig(f'{output_dir}/{file_name}.png', bbox_inches='tight')
    plt.close()
    

def plot_dea_timeseries(FigNo, ds, timeframe, time_unit_lake, hydro_type=None, time_unit_data=None, 
                        plot_mean=False, plot_station=False):
    """
    Plot DEA lake size time series with flood events and optional hydrological data.
    
    Creates a time series showing lake surface area from DEA with highlighted flood
    events. Can optionally overlay gridded mean or station rainfall/runoff data on
    a secondary y-axis.
    
    Parameters
    ----------
    FigNo : str
        figure number (e.g., 'Fig1').
    ds : xarray.Dataset
        Dataset containing 'dea' (lake size) and 'dea_events'.
    timeframe : dict
        Dictionary with 'start' and 'end' keys containing year strings.
    time_unit_lake : {'daily', 'monthly'}
        Time resolution for lake data.
    hydro_type : {'rainfall', 'runoff'}, optional
        Type of hydrological data to plot. Required if plot_mean or plot_station is True.
    time_unit_data : {'daily', 'monthly'}, optional
        Time resolution for hydrological data. Required if plot_mean or plot_station is True.
    plot_mean : bool, default False
        If True, overlay gridded mean catchment data.
    plot_station : bool, default False
        If True, overlay individual station observations.
    
    Output
    ------
    Saves PNG: '{output_dir}/{FigNo}_{Lake}_FloodEvents-{time_unit_lake}[_hydro_info].png'
    
    Raises
    ------
    KeyError
        If required variables are missing from dataset.
    ValueError
        If time units are invalid or peak requirements aren't met.
    
    Notes
    -----
    The function automatically resamples to monthly resolution when needed by either
    lake or hydrological data. Station data is filtered to remove stations with no
    data in the specified timeframe.
    """
    # ========== VALIDATE INPUTS ==========
    # Check required variables
    required_vars = ['dea', 'dea_events']
    if plot_mean:
        required_vars.append(f'mean_catchment_{hydro_type}')
    if plot_station:
        required_vars.append(f'station_{hydro_type}')
    
    missing_vars = [v for v in required_vars if v not in ds.data_vars and v not in ds.coords]
    if missing_vars:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing_vars)}')
    
    # Validate time units
    if time_unit_lake not in ['daily', 'monthly']:
        raise ValueError(f'time_unit_lake must be "daily" or "monthly", got "{time_unit_lake}"')
    
    if (plot_mean or plot_station) and time_unit_data not in ['daily', 'monthly']:
        raise ValueError(f'time_unit_data must be "daily" or "monthly", got "{time_unit_data}"')
    
    
    # ========== DATA PREPARATION ==========
    # Subset to timeframe and filter empty stations
    trimmed_ds = ds.sel(time=slice(timeframe['start'], timeframe['end']))
    if plot_station:
        trimmed_ds = filter_out_empty_stations(trimmed_ds, hydro_type)
    
    # Determine if monthly resampling is needed
    needs_monthly_lake = (time_unit_lake == 'monthly')
    needs_monthly_hydro = ((plot_mean or plot_station) and time_unit_data == 'monthly')
    needs_monthly = needs_monthly_lake or needs_monthly_hydro
    
    # Resample if needed
    monthly_ds = resample_lake_dataset_to_monthly(trimmed_ds) if needs_monthly else None
    
    # Extract lake size data
    if time_unit_lake == 'daily':
        lake_size = trimmed_ds['dea'].sel(lake_variable='Size')
    else:
        lake_size = monthly_ds['dea'].sel(lake_variable='Size')
    
    # Extract event IDs
    events = np.unique(trimmed_ds['dea_events'].dropna(dim='time').values)
    
    # Extract hydrological data if requested
    hydro_data = {}
    if plot_mean or plot_station:
        source_ds = monthly_ds if time_unit_data == 'monthly' else trimmed_ds
        
        if plot_mean:
            hydro_data['mean'] = source_ds[f'mean_catchment_{hydro_type}']
        
        if plot_station:
            hydro_data['station'] = source_ds[f'station_{hydro_type}']
    
    
    # ========== FILENAME CONSTRUCTION ==========
    filename_parts = [FigNo, Lake, f'FloodEvents-{time_unit_lake}']
    
    # Add hydrological data information
    if plot_mean or plot_station:
        filename_parts.append(f'{hydro_type.capitalize()}-{time_unit_data}')
        
        if plot_mean:
            filename_parts.append('mean')
        if plot_station:
            filename_parts.append('station')
    
    file_name = '_'.join(filename_parts)
    
    
    # ========== PLOTTING CONFIGURATION ==========
    # Colors for multiple stations
    blue_colors = ['#0000FF', '#00008B', '#4169E1', '#6495ED', '#87CEFA',
                   '#4682B4', '#5F9EA0', '#7B68EE', '#87CEEB', '#ADD8E6']
    mean_colour = 'maroon'
   
    # Determine plot style for hydrological data
    if plot_mean or plot_station:
        if time_unit_data == 'daily':
            plot_style_hydro = {'linestyle': '', 'marker': 'o', 'markersize': 6}
        else:  # monthly
            plot_style_hydro = {'linestyle': '-', 'marker': '', 'markersize': 0}
    
    # Configure labels
    labels = {}
    if plot_mean:
        labels['hydro_ylabel'] = f'{hydro_type.capitalize()} ({hydro_data["mean"].units})'
    elif plot_station:
        labels['hydro_ylabel'] = f'{hydro_type.capitalize()} ({hydro_data["station"].units})'
    
    labels['lake_ylabel'] = f'Lake Size {lake_size.units}'
    
    
    # ========== CREATE FIGURE AND AXES ==========
    fig = plt.figure(figsize=(40, 10))
    ax1 = fig.add_subplot(111)
    
    # Set up axes based on what we're plotting
    if plot_mean or plot_station:
        ax2 = ax1.twinx()
        axis_hydro = ax1  # Hydrological data on left
        axis_dea = ax2    # Lake data on right
    else:
        axis_dea = ax1    # Lake data only
    
    
    # ========== PLOT HYDROLOGICAL DATA ==========
    if plot_mean:
        # Plot mean catchment data
        axis_hydro.plot(
            hydro_data['mean'].time.values,
            hydro_data['mean'].values,
            label=f'{time_unit_data.capitalize()} mean {hydro_type} over catchment',
            color=mean_colour,
            zorder=3,
            **plot_style_hydro
        )
    
    if plot_station:
        # Plot individual station time series
        station_data = hydro_data['station']
        for i in range(station_data[f'{hydro_type}_station'].size):
            ts = station_data.isel({f'{hydro_type}_station': i})
            color = blue_colors[i % len(blue_colors)]
            station_id = ts[f'{hydro_type}_station'].item()
            station_name = ts[f'{hydro_type}_station_name'].values.item()
            
            axis_hydro.plot(
                ts.time.values, ts.values,
                label=f'{station_id}: {station_name}',
                color=color, alpha=0.8,
                zorder=2,
                **plot_style_hydro
            )
    
    
    # ========== PLOT LAKE DATA AND EVENTS ==========
    # Plot all lake observations as grey background
    axis_dea.plot(
        lake_size.time.values, lake_size.values,
        color='grey', linestyle='', marker='o', markersize=6,
        alpha=0.4, zorder=1
    )
    
    # Highlight each detected event
    for event in events:
        event_sizes = lake_size.where(lake_size.dea_events == event, drop=True).dropna(dim='time')
        axis_dea.plot(
            event_sizes.time, event_sizes,
            linestyle='--', marker='o', markersize=6,
            color='black',
            label=f'Event {int(event)}',
            zorder=5
        )
        
        # Add event label at peak
        label_size = event_sizes.where(event_sizes.dea_event_max, drop=True)
        axis_dea.text(
            label_size.time, label_size + 3,
            f'Event {int(event)}',
            fontsize=20, ha='center', va='bottom',
            color='black', zorder=6
        )
    
    
    # ========== FORMAT AXES ==========
    # Set y-axis labels
    if plot_mean or plot_station:
        axis_hydro.set_ylabel(labels['hydro_ylabel'], fontsize=20)
        axis_hydro.tick_params(axis='y', labelsize=20)
    
    axis_dea.set_ylabel(labels['lake_ylabel'], fontsize=20)
    axis_dea.tick_params(axis='y', labelsize=20)
    
    # Format x-axis with yearly ticks
    ax1.grid(True, which='major', axis='both')
    ax1.xaxis.set_major_locator(mdates.YearLocator(1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax1.get_xticklabels(), rotation=90, ha='center', fontsize=20)
    
    
    # ========== SAVE FIGURE ==========
    plt.savefig(f'{output_dir}/{file_name}.png', bbox_inches='tight')
    plt.close()    


def plot_dea_indices(FigNo, ds, timeframe, time_unit_lake, index):
    """
    Plot Rainfall Indices DEA lake size time series with flood events.
    
    Creates a time series showing rainfall indices and lake surface area from DEA
    with highlighted flood events.
    
    Parameters
    ----------
    FigNo : str
        figure number (e.g., 'Fig1').
    ds : xarray.Dataset
        Dataset containing 'dea' (lake size) and 'dea_events'.
    timeframe : dict
        Dictionary with 'start' and 'end' keys containing year strings.
    time_unit_lake : {'daily', 'monthly'}
        Time resolution for lake data.
    index : str
        Type of index to plot. Options:
        - 'mm': Plot all mm-based indices combined
        - 'days': Plot all days-based indices combined
        - Individual index names: 'monthly_total', 'Rx5day', 'SDII',
          'R40mm', 'R99p', 'CWD'

    Output
    ------
    Saves PNG: '{output_dir}/{FigNo}_{Lake}_FloodEvents-{time_unit_lake}[_index_info].png'
    
    Raises
    ------
    KeyError
        If required variables are missing from dataset.
    ValueError
        If time units or index are invalid.
    
    Notes
    -----
    The function automatically resamples to monthly resolution when needed by either
    lake or hydrological data. Station data is filtered to remove stations with no
    data in the specified timeframe.
    """
    # ========== VALIDATE INPUTS ==========
    # Check required variables
    required_vars = ['dea', 'dea_events', 'mean_catchment_rainfall']

    missing_vars = [v for v in required_vars if v not in ds.data_vars and v not in ds.coords]
    if missing_vars:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing_vars)}')
    
    # Validate time units
    if time_unit_lake not in ['daily', 'monthly']:
        raise ValueError(f'time_unit_lake must be "daily" or "monthly", got "{time_unit_lake}"')
    
    # Define available indices
    mm_indices = ['monthly_total', 'Rx5day', 'SDII']
    days_indices = ['R40mm', 'R99p', 'CWD']
    all_indices = mm_indices + days_indices
    
    # Validate index parameter
    valid_options = ['mm', 'days'] + all_indices
    if index not in valid_options:
        raise ValueError(f'index must be one of {valid_options}, got "{index}"')
    
    
    # ========== DATA PREPARATION ==========
    # Subset to timeframe and filter empty stations
    trimmed_ds = ds.sel(time=slice(timeframe['start'], timeframe['end']))
    
    # Determine if monthly resampling is needed
    needs_monthly_lake = (time_unit_lake == 'monthly')
    needs_monthly_hydro = (index == 'mm' or index in mm_indices)
    needs_monthly = needs_monthly_lake or needs_monthly_hydro
    
    # Resample if needed
    monthly_ds = resample_lake_dataset_to_monthly(trimmed_ds) if needs_monthly else None
    
    # Extract lake size data
    if time_unit_lake == 'daily':
        lake_size = trimmed_ds['dea'].sel(lake_variable='Size')
    else:
        lake_size = monthly_ds['dea'].sel(lake_variable='Size')
    
    # Extract event IDs
    events = np.unique(trimmed_ds['dea_events'].dropna(dim='time').values)
    
    # Extract hydrological data
    hydro_data = {}
    rainfall = trimmed_ds.mean_catchment_rainfall
    
    # Determine which indices to calculate
    indices_to_calc = []
    if index == 'mm':
        indices_to_calc = mm_indices
    elif index == 'days':
        indices_to_calc = days_indices
    else:
        indices_to_calc = [index]
    
    # Calculate required indices
    for idx in indices_to_calc:
        if idx == 'monthly_total':
            hydro_data['monthly_total'] = monthly_ds.mean_catchment_rainfall
            peaks = find_rainfall_peaks_near_lake_peaks(monthly_ds, percentile=0.8)
                
        elif idx == 'Rx5day':
            rx5day_rolling = rainfall.rolling(time=5, center=False).sum()
            hydro_data['Rx5day'] = rx5day_rolling.resample(time='ME').max(dim='time')
        
        elif idx == 'SDII':
            wet_days = rainfall >= 1
            total_wet_precip = rainfall.where(wet_days).resample(time='ME').sum(dim='time')
            n_wet_days = wet_days.resample(time='ME').sum(dim='time')
            hydro_data['SDII'] = total_wet_precip / n_wet_days.where(n_wet_days > 0)
            hydro_data['SDII'] = hydro_data['SDII'].fillna(0)
                
        elif idx == 'R40mm':
            heavy_days = rainfall > 40
            hydro_data['R40mm'] = heavy_days.resample(time='ME').sum(dim='time')
                
        elif idx == 'R99p':
            p99 = rainfall.quantile(0.99, dim='time')
            extreme_days = rainfall > p99
            hydro_data['R99p'] = extreme_days.resample(time='ME').sum(dim='time')
        
        elif idx == 'CWD':
            wet = rainfall >= 1
            wet_int = wet.astype(int)
            
            def max_consecutive_ones(arr):
                if len(arr) == 0:
                    return 0
                if arr.sum() == 0:
                    return 0
                return np.max(np.diff(np.where(np.concatenate(([arr[0]], arr[:-1] != arr[1:], [True])))[0])[::2])
            
            hydro_data['CWD'] = wet_int.resample(time='ME').apply(
                lambda x: xr.DataArray(max_consecutive_ones(x.values))
            )
    
    
    # ========== FILENAME CONSTRUCTION ==========
    filename_parts = [FigNo, Lake, f'FloodEvents-{time_unit_lake}', index]
    file_name = '_'.join(filename_parts)
    
    
    # ========== PLOTTING CONFIGURATION ==========
    # Colors for multiple stations
    #blue_colors = ['#4169E1', '#87CEFA', '#5F9EA0', '#87CEEB', '#ADD8E6']
    blue_colors = ['#08519c', '#3182bd', '#6baed6', '#2ca02c', '#74c476']
    
    # Determine plot style for hydrological data
    plot_style_hydro = {'linestyle': '-', 'marker': '', 'markersize': 0}
    
    # Configure labels
    labels = {}
    if index == 'mm' or index in mm_indices:
        labels['hydro_ylabel'] = 'Rainfall (mm)'
    else:
        labels['hydro_ylabel'] = 'Days'
        
    labels['lake_ylabel'] = f'Lake Size {lake_size.units}'
    
    
    # ========== CREATE FIGURE AND AXES ==========
    fig = plt.figure(figsize=(40, 10))
    ax1 = fig.add_subplot(111)
    
    # Set up axes based on what we're plotting
    ax2 = ax1.twinx()
    axis_hydro = ax1  # Hydrological data on left
    axis_dea = ax2    # Lake data on right
    
    
    # ========== PLOT HYDROLOGICAL DATA ==========
    
    # Plot index data
    for i, (key, value) in enumerate(hydro_data.items()):
        axis_hydro.plot(
            value.time.values,
            value.values,
            label=key,
            color=blue_colors[i % len(blue_colors)],
            zorder=3,
            **plot_style_hydro
        )
    
    if index == 'mm' or index == 'monthly_total':
        # Highlight rainfall peaks if applicable
        if 'monthly_total' in hydro_data:
            axis_hydro.plot(
                peaks.time.values,
                peaks.values,
                linestyle='', marker='o', markersize=8,
                label='Monthly Total Peaks',
                color='maroon',
                zorder=4
            )
    
    
    # ========== PLOT LAKE DATA AND EVENTS ==========
    # Plot all lake observations as grey background
    axis_dea.plot(
        lake_size.time.values, lake_size.values,
        color='grey', linestyle='', marker='o', markersize=6,
        alpha=0.4, zorder=1
    )
    
    # Highlight each detected event
    for event in events:
        event_sizes = lake_size.where(lake_size.dea_events == event, drop=True).dropna(dim='time')
        axis_dea.plot(
            event_sizes.time, event_sizes,
            linestyle='--', marker='o', markersize=6,
            color='black',
            label=f'Event {int(event)}',
            zorder=5
        )
        
        # Add event label at peak
        label_size = event_sizes.where(event_sizes.dea_event_max, drop=True)
        axis_dea.text(
            label_size.time, label_size + 3,
            f'Event {int(event)}',
            fontsize=20, ha='center', va='bottom',
            color='black', zorder=6
        )
    
    
    # ========== FORMAT AXES ==========
    # Set y-axis labels
    axis_hydro.set_ylabel(labels['hydro_ylabel'], fontsize=20)
    axis_hydro.tick_params(axis='y', labelsize=20)
    
    axis_dea.set_ylabel(labels['lake_ylabel'], fontsize=20)
    axis_dea.tick_params(axis='y', labelsize=20)
    
    # Format x-axis with yearly ticks
    ax1.grid(True, which='major', axis='both')
    ax1.xaxis.set_major_locator(mdates.YearLocator(1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax1.get_xticklabels(), rotation=90, ha='center', fontsize=20)
    
    # Add legend
    axis_hydro.legend(fontsize=20, loc='upper left')
    
    # ========== SAVE FIGURE ==========
    plt.savefig(f'{output_dir}/{file_name}.png', bbox_inches='tight')
    plt.close()
    
    
def plot_cumulative_rainfall_events(FigNo, ds, timeframe, magnitude=False, filtered=False):
    """
    Plot cumulative rainfall for all detected flood events.
    
    Generates cumulative rainfall time series aligned to each flood event's rise
    period (from start to peak). Events can be colored by distinct preset colors
    or ranked and colored by flood magnitude.
    
    Parameters
    ----------
    FigNo : str
        figure number (e.g., 'Fig1').
    ds : xarray.Dataset
        Dataset containing:
            - 'dea': lake variables with 'Size'
            - 'dea_events': event ID labels
            - 'dea_event_max': boolean marking event peaks
            - 'cumulative_rainfall_per_event': cumulative rainfall
            - 'dea_offset': time offset relative to event peak
            - 'cumulative_rainfall_per_event_filtered': filtered version (if filtered=True)
            - 'dea_offset_filtered': filtered offset (if filtered=True)
    timeframe : dict
        Dictionary with 'start' and 'end' keys containing year strings.
    magnitude : bool, default False
        If True, rank events by maximum lake size and color by magnitude gradient.
        If False, use distinct preset colors for each event.
    filtered : bool, default False
        If True, use filtered cumulative rainfall (removes low-intensity periods).
    
    Output
    ------
    Saves PNG: '{output_dir}/{FigNo}_{Lake}_CumulativeRainfall_AllEvents[_MagnitudeRanked][_Filtered].png'
    
    Raises
    ------
    KeyError
        If required variables are missing from dataset.
    
    Notes
    -----
    Only the rise period (dea_offset ≤ 0) is plotted for each event. The filtered
    option removes periods where daily rainfall is below 1% of maximum daily rainfall.
    """
    # ========== VALIDATE INPUTS ==========
    required_vars = ['dea', 'dea_events', 'dea_event_max', 
                     'cumulative_rainfall_per_event', 'dea_offset']
    if filtered:
        required_vars.extend(['dea_offset_filtered', 'cumulative_rainfall_per_event_filtered'])
    
    missing_vars = [v for v in required_vars if v not in ds]
    if missing_vars:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing_vars)}')
    
    
    # ========== DATA PREPARATION ==========
    # Select variables based on filtered option
    offset_var = 'dea_offset_filtered' if filtered else 'dea_offset'
    cum_rainfall_var = ('cumulative_rainfall_per_event_filtered' if filtered 
                       else 'cumulative_rainfall_per_event')
    
    # Subset to timeframe
    trimmed_ds = ds.sel(time=slice(timeframe['start'], timeframe['end']))
    
    # Extract cumulative rainfall
    cumulative_catchment_rainfall = trimmed_ds[cum_rainfall_var]
    
    # Extract rise phase only (offset ≤ 0)
    mask_rise = trimmed_ds[offset_var] <= 0
    event_offset_rise = trimmed_ds[offset_var].where(mask_rise, drop=True)
    cumulative_catchment_rainfall_rise = cumulative_catchment_rainfall.where(mask_rise).dropna(dim='time')
    
    # Prepare event ranking if using magnitude coloring
    if magnitude:
        lake_size_maximums = trimmed_ds['dea'].sel(lake_variable='Size').where(
            trimmed_ds.dea_event_max, drop=True
        )
        
        # Rank events by lake size (ascending for color gradient)
        ranked_lake_size_max = lake_size_maximums.sortby(lake_size_maximums, ascending=True)
        event_ranking = ranked_lake_size_max['dea_events'].values.astype(int)
    
    # Calculate x-axis range
    x_min = event_offset_rise.min(skipna=True).item()
    x_max = event_offset_rise.max(skipna=True).item()
    
    
    # ========== FILENAME CONSTRUCTION ==========
    filename_parts = [FigNo, Lake, 'CumulativeRainfall', 'AllEvents']
    if magnitude:
        filename_parts.append('MagnitudeRanked')
    if filtered:
        filename_parts.append('Filtered')
    
    file_name = '_'.join(filename_parts)
    
    
    # ========== PLOTTING CONFIGURATION ==========
    # Configure colors based on magnitude option
    if magnitude:
        # Create gradient colormap for magnitude ranking
        original_cmap = colormaps['Blues']
        cmap = truncate_colormap(original_cmap, minval=0.2, maxval=1.0)
        norm = plt.Normalize(vmin=1, vmax=event_ranking.size)
        
        # Generate colors and sort by event ranking
        unranked_colours = np.array([cmap(norm(i + 1)) for i in range(event_ranking.size)])
        colours = unranked_colours[np.argsort(event_ranking)]
    else:
        # Preset distinct colors for each event
        colours = ['tomato', 'darkred', 'orange', 'gold', 'olive', 'forestgreen', 'teal', 'aqua',
                   'steelblue', 'navy', 'purple', 'fuchsia', 'pink', 'maroon', 'lightcoral', 'red',
                   'sienna', 'tan', '#000033', '#336666', '#99cc33', '#339966', '#8EB28E', '#336600',
                   '#87CEEB', '#ADD8E6']
    
    # Labels
    labels = {
        'ylabel': f'Cumulative Rainfall ({cumulative_catchment_rainfall.units})'
    }
    
    # X-axis ticks every 16 days
    xticks = np.arange(x_min - (x_min % 16), x_max + 1, 16)
    xtick_labels = [f'{int(x)}' for x in xticks]
    
    
    # ========== CREATE FIGURE AND PLOT ==========
    fig = plt.figure(figsize=(30, 10))
    ax = fig.add_subplot(111)
    
    # Plot cumulative rainfall for each event
    for event, event_seg in cumulative_catchment_rainfall_rise.groupby('dea_events'):
        ax.plot(
            event_seg[offset_var], event_seg.values,
            color=colours[int(event - 1)],
            linestyle='-',
            label=f'Event {int(event)}',
            zorder=2
        )
    
    
    # ========== FORMAT AXES ==========
    # Set x-axis ticks and labels
    ax.set_xticks(xticks, xtick_labels, fontsize=15)
    ax.tick_params(axis='y', labelsize=15)
    
    # Add grid and y-axis label
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylabel(labels['ylabel'], fontsize=20)
    
    # Add legend or colorbar depending on magnitude option
    if magnitude:
        # Add colorbar showing magnitude ranking
        cbar_ax = inset_axes(
            ax, width=0.5, height=5, loc='upper left',
            bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes, borderpad=1
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_ticks([event_ranking.min(), event_ranking.max()])
        cbar.set_ticklabels(['Smallest', 'Largest'])
        cbar.ax.tick_params(labelsize=20)
    else:
        # Add legend with event numbers
        ax.legend(fontsize=20, loc='upper left')
    
    
    # ========== SAVE FIGURE ==========
    plt.savefig(f'{output_dir}/{file_name}.png', bbox_inches='tight')
    plt.close()


def plot_cumulative_rainfall_windows(FigNo, ds, threshold, time_interval, start=None, magnitude=False):
    """
    Plot cumulative rainfall windows categorized by event status and threshold.
    
    Creates a plot showing cumulative rainfall leading up to rolling sum peaks,
    with segments categorized into four groups:
    1. Lake-filling events above threshold (blue)
    2. Lake-filling events below threshold (maroon/ranked)
    3. Non-events above threshold (maroon)
    4. Non-events below threshold (grey)
    
    Parameters
    ----------
    FigNo : str
        figure number (e.g., 'Fig1').
    ds : xarray.Dataset
        Dataset containing:
            - 'dea': lake variables with 'Size' and 'dea_event_max'
            - 'dea_events': event ID labels
            - 'cumulative_rainfall_up_to_peaks': cumulative rainfall to peaks
            - 'cum_window_{time_interval}': window position coordinate
            - 'cum_window_is_event': event classification
            - 'rolling_peak_above_{threshold}': threshold flags
    threshold : float
        Rainfall threshold (mm) for categorization.
    time_interval : str
        Window specification (e.g., '128_days').
    start : str, optional
        Start year for filtering data. If None, uses all data.
    magnitude : bool, default False
        If True, color event segments by lake size magnitude gradient.
        If False, use fixed colors for each category.
    
    Output
    ------
    Saves PNG: '{output_dir}/{FigNo}_{Lake}_CumulativeRainfall_{window}{unit}-window_{start}to{end}[_MagnitudeRanked].png'
    
    Raises
    ------
    KeyError
        If required variables are missing from dataset.
    ValueError
        If time_interval cannot be parsed into window size and unit.
    
    Notes
    -----
    Segments are filtered to start from the first complete window after the start
    date. The magnitude option uses a blue gradient from smallest to largest events.
    """
    # ========== VALIDATE INPUTS ==========
    required_vars = [
        'dea', 'dea_events', 'dea_event_max',
        f'cum_window_{time_interval}',
        'cumulative_rainfall_up_to_peaks',
        'cum_window_is_event',
        f'rolling_peak_above_{threshold}'
    ]
    
    missing_vars = [v for v in required_vars if v not in ds]
    if missing_vars:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing_vars)}')
    
    # Parse time_interval into window size and unit
    try:
        window, time_unit = time_interval.split('_', 1)
        window = int(window)
    except Exception:
        raise ValueError(
            f'time_interval "{time_interval}" could not be parsed into window size and unit'
        )
    
    
    # ========== DATA PREPARATION ==========
    # Extract cumulative rainfall windows
    ds_windows = ds.where(ds[f'cum_window_{time_interval}'].dropna(dim='time'), drop=True)
    cum_to_peak = ds_windows.cumulative_rainfall_up_to_peaks
    
    # Filter to start date if provided
    if start is not None:
        # Find first complete window after start date
        start_segment = cum_to_peak.where(
            cum_to_peak[f'cum_window_{time_interval}'] == 1, drop=True
        ).sel(time=slice(start, None))
        start_date = start_segment[0].time
        cum_to_peak = cum_to_peak.sel(time=slice(start_date, None))
    
    # Categorize segments into four groups
    mask_no_event_below = (
        (~cum_to_peak[f'rolling_peak_above_{threshold}']) & 
        np.isnan(cum_to_peak.cum_window_is_event)
    )
    mask_no_event_above = (
        cum_to_peak[f'rolling_peak_above_{threshold}'] & 
        np.isnan(cum_to_peak.cum_window_is_event)
    )
    mask_event_below = (
        (~cum_to_peak[f'rolling_peak_above_{threshold}']) & 
        (~np.isnan(cum_to_peak.cum_window_is_event))
    )
    mask_event_above = (
        cum_to_peak[f'rolling_peak_above_{threshold}'] & 
        (~np.isnan(cum_to_peak.cum_window_is_event))
    )
    
    # Extract categorized data
    no_event_below = cum_to_peak.where(mask_no_event_below).dropna(dim='time')
    no_event_above = cum_to_peak.where(mask_no_event_above).dropna(dim='time')
    event_below = cum_to_peak.where(mask_event_below).dropna(dim='time')
    event_above = cum_to_peak.where(mask_event_above).dropna(dim='time')
    
    # Prepare event ranking if using magnitude coloring
    events = np.unique(ds.dea_events.dropna(dim='time').values)
    if magnitude:
        lake_size_max = ds['dea'].sel(lake_variable='Size').where(
            ds.dea_event_max, drop=True
        ).dropna(dim='time')
        ranked_lake_size_max = lake_size_max.sortby(lake_size_max, ascending=True)
        event_ranking = ranked_lake_size_max['dea_events'].values.astype(int)
    
    
    # ========== FILENAME CONSTRUCTION ==========
    start_ds = str(ds.time[0].dt.year.item())
    end_ds = str(ds.time[-1].dt.year.item())
    start_year = start_ds if start is None else start
    
    filename_parts = [
        FigNo,
        Lake, 'CumulativeRainfall',
        f'{window}{time_unit[:-1]}-window',
        f'{start_year}to{end_ds}'
    ]
    if magnitude:
        filename_parts.append('MagnitudeRanked')
    
    file_name = '_'.join(filename_parts)
    
    
    # ========== PLOTTING CONFIGURATION ==========
    # Configure colors based on magnitude option
    if magnitude:
        # Create magnitude gradient for events
        original_cmap = colormaps['Blues']
        cmap = truncate_colormap(original_cmap, minval=0.2, maxval=1.0)
        norm = plt.Normalize(vmin=1, vmax=event_ranking.size)
        colours = np.array([cmap(norm(i + 1)) for i in range(event_ranking.size)])
        colours_event_above = colours[np.argsort(event_ranking)]
        colours_event_below = colours_event_above
    else:
        # Fixed colors for each category
        colours_event_above = ['#005b96'] * len(events)  # Blue for events above
        colours_event_below = ['maroon'] * len(events)    # Maroon for events below
    
    # Non-event colors (same for both modes)
    colour_no_event_below = 'lightgrey'
    colour_no_event_above = 'maroon'
    
    # Labels
    axis_labels = {
        'xlabel': f'{time_unit.capitalize()} Leading up to Peak',
        'ylabel': f'Cumulative Rainfall ({cum_to_peak.units})'
    }
    
    # Legend configuration
    if magnitude:
        handles_config = {
            'handle1': {'colour': '#005b96', 'label': 'Lake Filling'},
            'handle2': {'colour': 'lightgrey', 'label': 'No Lake Filling                           '},
            'handle3': {'colour': 'white', 'label': ''},
            'handle4': {'colour': 'white', 'label': ''}
        }
    else:
        handles_config = {
            'handle1': {'colour': '#005b96', 'label': 'Lake Filling'},
            'handle2': {'colour': 'maroon', 'label': f'Lake Filling - below {threshold} mm'},
            'handle3': {'colour': 'maroon', 'label': f'No Lake Filling - above {threshold} mm'},
            'handle4': {'colour': 'lightgrey', 'label': 'No Lake Filling'}
        }
    
    # X-axis ticks
    xticks = np.arange(0, int(window), 10)
    xtick_labels = [f'{int(x)}' for x in xticks]
    
    
    # ========== CREATE FIGURE AND PLOT ==========
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(111)
    
    # Plot event segments below threshold
    if event_below.size > 0:
        event_below['cum_window_is_event'] = event_below['cum_window_is_event'].compute()
        for event_id, da_seg in event_below.groupby('cum_window_is_event'):
            ax1.plot(
                da_seg[f'cum_window_{time_interval}'].values, da_seg.values,
                color=colours_event_below[int(event_id - 1)],
                linestyle='-', zorder=4
            )
    
    # Plot event segments above threshold
    if event_above.size > 0:
        event_above['cum_window_is_event'] = event_above['cum_window_is_event'].compute()
        for event_id, da_seg in event_above.groupby('cum_window_is_event'):
            ax1.plot(
                da_seg[f'cum_window_{time_interval}'].values, da_seg.values,
                color=colours_event_above[int(event_id - 1)],
                linestyle='-', zorder=5
            )
    
    # Plot non-event segments below threshold
    if no_event_below.size > 0:
        for seg_id, da_seg in no_event_below.groupby('segment_id'):
            ax1.plot(
                da_seg[f'cum_window_{time_interval}'].values, da_seg.values,
                color=colour_no_event_below,
                linestyle='--', zorder=2
            )
    
    # Plot non-event segments above threshold
    if no_event_above.size > 0:
        for seg_id, da_seg in no_event_above.groupby('segment_id'):
            ax1.plot(
                da_seg[f'cum_window_{time_interval}'].values, da_seg.values,
                color=colour_no_event_above,
                linestyle='--', zorder=3
            )
    
    
    # ========== FORMAT AXES AND LEGEND ==========
    # Add threshold line
    ax1.axhline(y=threshold, color='#03396c', linestyle='--', linewidth=2, zorder=1)
    
    # Set labels and ticks
    ax1.set_xlabel(axis_labels['xlabel'], fontsize=18)
    ax1.set_ylabel(axis_labels['ylabel'], fontsize=18)
    ax1.set_xticks(xticks, xtick_labels, fontsize=13)
    ax1.tick_params(axis='both', labelsize=13)
    
    # Add grid
    ax1.grid(True)
    
    # Add colorbar if using magnitude ranking
    if magnitude:
        cbar_ax = inset_axes(
            ax1, width=3.1, height=0.15, loc='upper left',
            bbox_to_anchor=(0.05, 0.835, 1, 0.05),
            bbox_transform=ax1.transAxes, borderpad=0
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks([events.min(), events.max()])
        cbar.set_ticklabels(['Smallest', 'Largest'])
        cbar.ax.tick_params(labelsize=15)
    
    # Add legend
    handles = [
        mlines.Line2D([], [], color=handles_config['handle1']['colour'], 
                     linestyle='-', label=handles_config['handle1']['label']),
        mlines.Line2D([], [], color=handles_config['handle2']['colour'],
                     linestyle='-', label=handles_config['handle2']['label']),
        mlines.Line2D([], [], color=handles_config['handle3']['colour'],
                     linestyle='--', label=handles_config['handle3']['label']),
        mlines.Line2D([], [], color=handles_config['handle4']['colour'],
                     linestyle='--', label=handles_config['handle4']['label'])
    ]
    ax1.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, 1), fontsize=15)
    
    
    # ========== SAVE FIGURE ==========
    plt.savefig(f'{output_dir}/{file_name}.png', bbox_inches='tight')
    plt.close()
        
    
def plot_threshold_analysis(FigNo, ds, timeframe, window_max, threshold, time_unit, optimal_window):
    """
    Analyze and plot cumulative rainfall thresholds across multiple window sizes.
    
    Calculates rolling sums for windows from 1 to window_max, identifies peaks,
    filters to one peak per year, and plots results separated by event/non-event
    years. Events are color-coded based on whether they exceed the threshold.
    
    Parameters
    ----------
    FigNo : str
        figure number (e.g., 'Fig1').
    ds : xarray.Dataset
        Dataset containing:
            - 'mean_catchment_rainfall': daily or monthly rainfall
            - 'dea_events': event ID labels
            - 'dea_event_max': boolean marking event peaks
    timeframe : dict
        Dictionary with 'start' and 'end' keys containing year strings.
    window_max : int
        Maximum rolling window size to analyze.
    threshold : float
        Rainfall threshold (mm) for color coding events.
    time_unit : str
        Unit of rolling window: 'days' or 'months'.
    optimal_window : int
        Optimal window size to mark with vertical line.
    
    Output
    ------
    Saves PNG: '{output_dir}/{FigNo}_{Lake}_CumulativeRainfall_ThresholdAnalysis_{start}to{end}.png'
    
    Raises
    ------
    KeyError
        If required variables are missing from dataset.
    ValueError
        If time_unit is invalid or peak/trough counts are inconsistent.
    
    Notes
    -----
    Years are assigned by shifting peak dates forward by 90 days (daily) or 3 months
    (monthly) to handle seasonal peaks near year boundaries. First and last years
    are trimmed to avoid edge effects.
    """
    
    # ========== VALIDATE INPUTS ==========
    required_vars = ['mean_catchment_rainfall', 'dea_events', 'dea_event_max']
    missing_vars = [v for v in required_vars if v not in ds]
    if missing_vars:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing_vars)}')
    
    # Validate time_unit and set parameters
    if time_unit == 'days':
        distance_val = 221  # Minimum days between peaks
        move_date = 90      # Days to shift for year assignment
    elif time_unit == 'months':
        distance_val = 8    # Minimum months between peaks
        move_date = 3       # Months to shift for year assignment
    else:
        raise ValueError('time_unit must be "days" or "months"')
    
    
    # ========== DATA PREPARATION ==========
    # Subset to timeframe
    trimmed_ds = ds.sel(time=slice(timeframe['start'], timeframe['end']))
    catchment_rainfall = trimmed_ds['mean_catchment_rainfall'].compute()
    
    # Calculate rolling sums and extract annual peaks for each window size
    rolling_sums_dict = {}
    for window in range(1, window_max + 1):
        
        # Calculate rolling sum
        rainfall_sum = catchment_rainfall.rolling(time=window, center=False).sum()
        
        # Identify peaks and troughs
        peaks_and_troughs = identify_peaks_and_troughs(rainfall_sum, distance_val)
        
        # Validate peak/trough counts
        num_peaks = len(peaks_and_troughs['peaks'])
        num_troughs = len(peaks_and_troughs['troughs'])
        if not (num_peaks == num_troughs or num_peaks == num_troughs - 1):
            raise ValueError(
                f'Window {window}: peaks ({num_peaks}) must equal or be one less than troughs ({num_troughs})'
            )
        
        # Extract peak values
        rolling_sum_peaks = rainfall_sum[peaks_and_troughs['peaks']]
        
        # Assign year by shifting dates (handles seasonal boundary issues)
        rolling_sum_peaks = rolling_sum_peaks.assign_coords(
            year=(rolling_sum_peaks.time + pd.Timedelta(**{time_unit: move_date})).dt.year
        )
        
        # Keep only maximum peak per year
        yearly_max_peaks = rolling_sum_peaks.groupby('year').max()
        
        rolling_sums_dict[f'{window} {time_unit}'] = yearly_max_peaks
    
    # Combine all windows into single DataArray
    combined_da = xr.concat(rolling_sums_dict.values(), dim='window')
    combined_da = combined_da.assign_coords(window=list(rolling_sums_dict.keys()))
    combined_da = combined_da.assign_coords(
        year=('year', pd.to_datetime(combined_da.year.values, format='%Y'))
    )
    
    # Trim edge years to avoid boundary effects
    combined_da = combined_da.sel(year=slice(combined_da.year[1], combined_da.year[-2]))
    
    # Separate event and non-event years
    event_ids = ds.dea_events.where(ds.dea_event_max, drop=True)
    event_years = event_ids.time.values.astype('datetime64[Y]')
    mask_events = np.isin(combined_da['year'].values, event_years.astype('datetime64[ns]'))
    
    events_da = combined_da.isel(year=mask_events)
    non_events_da = combined_da.isel(year=~mask_events)
    
    # Fill gaps for continuous lines
    events_da_filled = events_da.ffill(dim='window').bfill(dim='window')
    non_events_da_filled = non_events_da.ffill(dim='window').bfill(dim='window')
    
    
    # ========== FILENAME CONSTRUCTION ==========
    filename_parts = [
        FigNo, Lake, 'CumulativeRainfall', 'ThresholdAnalysis',
        f'{timeframe["start"]}to{timeframe["end"]}'
    ]
    file_name = '_'.join(filename_parts)
    
    
    # ========== PLOTTING CONFIGURATION ==========
    # Colors for different categories
    colours = {
        'non_event': '#b3cde0',      # Light blue
        'event_below': 'maroon',      # Maroon for events below threshold
        'event_above': '#005b96',     # Dark blue for events above threshold
        'threshold_line': '#03396c'   # Threshold line
    }
    
    # Labels
    labels = {
        'xlabel': f'Cumulative {time_unit.capitalize()}',
        'ylabel': f'Cumulative Rainfall ({catchment_rainfall.units})'
    }
    
    # Legend
    legend_handles = [
        mlines.Line2D([], [], color=colours['non_event'], label='Non-event years'),
        mlines.Line2D([], [], color=colours['event_below'], label=f'Event years (< {threshold} mm)'),
        mlines.Line2D([], [], color=colours['event_above'], label=f'Event years (≥ {threshold} mm)'),
        mlines.Line2D([], [], color=colours['threshold_line'], linestyle='--', 
                     label=f'Threshold ({threshold} mm)')
    ]
    
    # X-axis ticks
    xticks = np.arange(0, int(window_max) + 1, 16)
    xtick_labels = [f'{int(x)}' for x in xticks]
    
    
    # ========== CREATE FIGURE AND PLOT ==========
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Plot non-event years
    for year in non_events_da_filled['year'].values:
        ax.plot(
            non_events_da_filled['window'].values,
            non_events_da_filled.sel(year=year).values,
            color=colours['non_event'], zorder=2
        )
    
    # Plot event years colored by threshold
    for year in events_da_filled['year'].values:
        yvals = events_da_filled.sel(year=year).values
        max_rainfall = np.nanmax(yvals)
        
        # Color based on threshold
        color = colours['event_below'] if max_rainfall < threshold else colours['event_above']
        
        ax.plot(events_da_filled['window'].values, yvals, color=color, zorder=3)
    
    # Add threshold reference lines
    ax.axhline(y=threshold, color=colours['threshold_line'], linestyle='--', 
              linewidth=2, zorder=1)
    ax.axvline(x=optimal_window, color=colours['threshold_line'], linestyle='--', 
              linewidth=2, zorder=1)
    
    
    # ========== FORMAT AXES ==========
    # Set labels and ticks
    ax.set_xlabel(labels['xlabel'], fontsize=18)
    ax.set_ylabel(labels['ylabel'], fontsize=18)
    ax.set_xticks(xticks, xtick_labels, fontsize=13)
    ax.tick_params(axis='both', labelsize=13)
    
    # Add grid and legend
    ax.grid(True)
    ax.legend(handles=legend_handles, loc='upper left', fontsize=15)
    
    
    # ========== SAVE FIGURE ==========
    plt.savefig(f'{output_dir}/{file_name}.png', bbox_inches='tight')
    plt.close()


def create_threshold_sensitivity_table(ds, timeframe, window_range, threshold, time_unit):
    """
    Create a table showing counts of event and non-event years for three threshold levels.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing rainfall and event data
    timeframe : dict
        Dictionary with 'start' and 'end' keys containing year strings
    window_range : tuple
        (min_window, max_window) - range of window sizes to analyze (e.g., (80, 160))
    threshold : float
        Central rainfall threshold (mm) for analysis
    time_unit : str
        Unit of rolling window: 'days' or 'months'
    
    Returns
    -------
    pd.DataFrame
        Table with columns for three thresholds (threshold-20, threshold, threshold+20):
        - Window Size
        - Events above {threshold-20}mm: Count of events meeting threshold-20
        - Non-Events above {threshold-20}mm: Count of non-events meeting threshold-20
        - Events above {threshold}mm: Count of events meeting threshold
        - Non-Events above {threshold}mm: Count of non-events meeting threshold
        - Events above {threshold+20}mm: Count of events meeting threshold+20
        - Non-Events above {threshold+20}mm: Count of non-events meeting threshold+20
    """
    
    # Validate inputs
    required_vars = ['mean_catchment_rainfall', 'dea_events', 'dea_event_max']
    missing_vars = [v for v in required_vars if v not in ds]
    if missing_vars:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing_vars)}')
    
    # Set parameters based on time unit
    if time_unit == 'days':
        distance_val = 221
        move_date = 90
    elif time_unit == 'months':
        distance_val = 8
        move_date = 3
    else:
        raise ValueError('time_unit must be "days" or "months"')
    
    # Define three threshold levels
    thresholds = {
        f'{threshold-20}mm': threshold - 20,
        f'{threshold}mm': threshold,
        f'{threshold+20}mm': threshold + 20
    }
    
    # Subset to timeframe
    trimmed_ds = ds.sel(time=slice(timeframe['start'], timeframe['end']))
    catchment_rainfall = trimmed_ds['mean_catchment_rainfall'].compute()
    
    # Get event years
    event_ids = ds.dea_events.where(ds.dea_event_max, drop=True)
    event_years = set(event_ids.time.dt.year.values)
    
    # Initialize results storage
    results = []
    
    # Process each window size
    for window in range(window_range[0], window_range[1] + 1):
        # Calculate rolling sum
        rainfall_sum = catchment_rainfall.rolling(time=window, center=False).sum()
        
        # Identify peaks and troughs
        peaks_and_troughs = identify_peaks_and_troughs(rainfall_sum, distance_val)
        
        # Extract peak values
        rolling_sum_peaks = rainfall_sum[peaks_and_troughs['peaks']]
        
        # Assign year by shifting dates
        rolling_sum_peaks = rolling_sum_peaks.assign_coords(
            year=(rolling_sum_peaks.time + pd.Timedelta(**{time_unit: move_date})).dt.year
        )
        
        # Keep only maximum peak per year
        yearly_max_peaks = rolling_sum_peaks.groupby('year').max()
        
        # Initialize row data
        row_data = {'Window Size': f'{window} {time_unit}'}
        
        # Calculate for each threshold
        for thresh_label, thresh_value in thresholds.items():
            # Find years exceeding this threshold
            exceeding_threshold = yearly_max_peaks[yearly_max_peaks >= thresh_value]
            years_exceeding = set(exceeding_threshold.year.values)
            
            # Count event and non-event years
            event_years_meeting = years_exceeding.intersection(event_years)
            non_event_years_meeting = years_exceeding - event_years
            
            # Add to row data
            if thresh_value == threshold - 20:
                row_data[f'Events above {threshold-20}mm'] = len(event_years_meeting)
                row_data[f'Non-Events above {threshold-20}mm'] = len(non_event_years_meeting)
            elif thresh_value == threshold:
                row_data[f'Events above {threshold}mm'] = len(event_years_meeting)
                row_data[f'Non-Events above {threshold}mm'] = len(non_event_years_meeting)
            else:  # threshold + 20
                row_data[f'Events above {threshold+20}mm'] = len(event_years_meeting)
                row_data[f'Non-Events above {threshold+20}mm'] = len(non_event_years_meeting)
        
        results.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns for clarity
    column_order = [
        'Window Size',
        f'Events above {threshold-20}mm', f'Non-Events above {threshold-20}mm',
        f'Events above {threshold}mm', f'Non-Events above {threshold}mm',
        f'Events above {threshold+20}mm', f'Non-Events above {threshold+20}mm'
    ]
    df = df[column_order]
    
    return df


def plot_scatter_RollingSumVsLakeSize(FigNo, df, ds, window, outlier_events=None, show_labels=True, 
                                     show_trendline=True):
    """
    Plot scatter of rolling sum vs lake size with outliers highlighted and trendline.
    
    Parameters
    ----------
    FigNo : str
        figure number (e.g., 'Fig1').
    df : pandas.DataFrame
        DataFrame containing the event data
    ds : xarray.Dataset
        Dataset containing:
            - 'dea': lake variables with 'Size' and 'dea_event_max'
            - 'rolling_sum_of_mean_catchment_rainfall': rolling sum rainfall
    window : int
        Window size for rolling sum calculation
    outlier_events : list, optional
        List of event indices to mark as outliers (default: None)
    show_labels : bool, optional
        Whether to show event labels on points (default: True)
    show_trendline : bool, optional
        Whether to show trendline for non-outlier points (default: True)
    
    Returns
    -------
    dict
        Dictionary containing regression statistics (slope, intercept, r_value, p_value)
    
    Raises
    ------
    KeyError
        If required variables are missing from dataset.
    ValueError
        If f'{window} day total', 'lake peaks' not in df.

    """
    # ========== VALIDATE INPUTS ==========
    required_vars = ['dea', 'rolling_sum_of_mean_catchment_rainfall']
    missing_vars = [v for v in required_vars if v not in ds]
    if missing_vars:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing_vars)}')

    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    
    required_cols = [f'{window} day total', 'lake peaks']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # ========== DATA PREPARATION ==========
    # Handle outliers
    if outlier_events is None:
        outlier_events = []
    
    # Get all indices and separate outliers from others
    all_indices = df.index.tolist()
    other_events = [x for x in all_indices if x not in set(outlier_events)]
    
    # Validate that outlier events exist in the dataframe
    valid_outliers = [x for x in outlier_events if x in all_indices]
    if outlier_events and len(valid_outliers) < len(outlier_events):
        print("Warning: Some outlier indices not found in DataFrame")
    outlier_events = valid_outliers
    
    # Extract values
    x = df.loc[other_events][f'{window} day total']
    y = df.loc[other_events]['lake peaks']
    
    if outlier_events:
        x_outliers = df.loc[outlier_events][f'{window_size_daily} day total']
        y_outliers = df.loc[outlier_events]['lake peaks']
    
    # ========== FILENAME CONSTRUCTION ==========
    filename_parts = [FigNo, Lake, 'Scatter', 'RollingSumVsLakeSize']
    file_name = '_'.join(filename_parts)
    
    # ========== PLOTTING CONFIGURATION ==========
    # Define specific colors
    colors = {
        'other': '#005b96',      # blue
        'outlier': '#ff7f0e',    # orange
        'trendline': '#005b96'   # match main data
    }
    
    # ========== CREATE FIGURE AND PLOT ==========
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(111)
    
    # Create scatter plots with specific colors
    ax1.scatter(x,
                y,
                color=colors['other'],
                alpha=0.8,
                s=50,
                label="Events")
    
    if outlier_events:
        ax1.scatter(x_outliers,
                    y_outliers,
                    color=colors['outlier'],
                    alpha=0.8,
                    s=100,
                    marker='^',  # different marker for outliers
                    label="Outliers")
    
    # ========== ADD LABELS (OPTIONAL) ==========
    if show_labels:
        # Add labels for non-outlier points
        for idx, x_val, y_val in zip(other_events, x, y):
            ax1.annotate(f'Event {str(idx)}', 
                        (x_val, y_val), 
                        xytext=(5, 5),  # offset from point
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.7)
        
        # Add labels for outlier points
        if outlier_events:
            for idx, x_val, y_val in zip(outlier_events, x_outliers, y_outliers):
                ax1.annotate(f'Event {str(idx)}', 
                            (x_val, y_val), 
                            xytext=(5, 5),  # offset from point
                            textcoords='offset points',
                            fontsize=8,
                            alpha=0.7,
                            color=colors['outlier'])  # match outlier color
    
    # ========== CALCULATE AND PLOT TRENDLINE (OPTIONAL) ==========
    results = {}
    if show_trendline and len(x) > 1:
        # Calculate trendline for non-outlier data
        slope, intercept, r_value, p_value, std_err = stats.linregress(x.values, y.values)
        
        # Store results
        results = {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        }
        
        # Create trendline points
        line_x = np.array([x.values.min(), x.values.max()])
        line_y = slope * line_x + intercept
        
        # Plot trendline
        ax1.plot(line_x, line_y, 
                 color=colors['trendline'], 
                 linestyle='--', 
                 linewidth=2, 
                 label=f'Trendline (non-outliers, R² = {r_value**2:.3f})')
        
        # Add R² value and equation as text
        equation_text = f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}\np-value = {p_value:.3e}'
        ax1.text(0.05, 0.95, equation_text, 
                 transform=ax1.transAxes, 
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== FORMAT AXES ==========
    # Add axis labels
    lake_units = ds.dea.sel(lake_variable='Size').units
    rainfall_units = ds.rolling_sum_of_mean_catchment_rainfall.units
    ax1.set_xlabel(f'{window_size_daily} Day Rainfall Total ({rainfall_units})', fontsize=14)
    ax1.set_ylabel(f'Lake Filling Peaks ({lake_units})', fontsize=14)
    
    # Improve legend
    ax1.legend(loc='best', fontsize=12)
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3)
    
    # ========== SAVE FIGURE ==========
    plt.savefig(f'{output_dir}/{file_name}.png', bbox_inches='tight')
    plt.close()
    
    return results
    

def plot_rolling_sum(ax_left, ax_right, ds, start_date, end_date, window, time_unit, 
                     threshold1, threshold2, peaks_troughs=None, show_legend=True):
    """
    Plot rolling sum rainfall with lake events, highlighting threshold exceedances.
    
    Creates a dual-axis plot showing rolling sum cumulative rainfall on the left axis
    with color-coded segments based on threshold exceedances, and lake size with
    detected events on the right axis. Optionally marks identified peaks and troughs.
    
    Parameters
    ----------
    ax_left : matplotlib.axes.Axes
        Axis for plotting cumulative rainfall and thresholds.
    ax_right : matplotlib.axes.Axes
        Axis for plotting lake size and events.
    ds : xarray.Dataset
        Dataset containing:
            - 'rolling_sum_of_mean_catchment_rainfall': rolling sum rainfall
            - 'dea': lake variables with 'Size'
            - 'dea_events': event ID labels
            - 'rolling_peak_above_{threshold1}': boolean for threshold1
            - 'rolling_peak_above_{threshold2}': boolean for threshold2
    start_date : str
        Start year for plotting. Determines grid spacing.
    end_date : str
        End year for plotting.
    window : int
        Rolling window size for cumulative rainfall.
    time_unit : str
        Unit of time for rolling sum: 'days' or 'months'.
    threshold1 : float
        Lower rainfall threshold (mm) to highlight.
    threshold2 : float
        Higher rainfall threshold (mm) to highlight.
    peaks_troughs : dict, optional
        Dictionary with 'peaks' and 'troughs' keys containing DataArrays.
        If provided, peaks and troughs are plotted.
    show_legend : bool, default True
        If True, displays legend on left axis.
    
    Raises
    ------
    KeyError
        If required variables are missing from dataset.
    ValueError
        If peaks_troughs is provided but not a dict or missing required keys.
    """
    # ========== VALIDATE INPUTS ==========
    required_vars = [
        'rolling_sum_of_mean_catchment_rainfall', 'dea', 'dea_events',
        f'rolling_peak_above_{threshold1}', f'rolling_peak_above_{threshold2}'
    ]
    missing_vars = [v for v in required_vars if v not in ds]
    if missing_vars:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing_vars)}')
    
    # Validate peaks_troughs if provided
    if peaks_troughs is not None:
        if not isinstance(peaks_troughs, dict):
            raise ValueError('peaks_troughs must be a dictionary')
        if 'peaks' not in peaks_troughs or 'troughs' not in peaks_troughs:
            raise ValueError('peaks_troughs must contain "peaks" and "troughs" keys')
    
    
    # ========== DATA PREPARATION ==========
    # Subset to date range
    trimmed_ds = ds.sel(time=slice(start_date, end_date))
    
    # Extract rainfall data
    rainfall_sum = trimmed_ds['rolling_sum_of_mean_catchment_rainfall']
    rainfall_sum_above_threshold1 = rainfall_sum.where(
        rainfall_sum[f'rolling_peak_above_{threshold1}']
    )
    rainfall_sum_above_threshold2 = rainfall_sum.where(
        rainfall_sum[f'rolling_peak_above_{threshold2}']
    )
    
    # Extract lake data
    lake_size = trimmed_ds['dea'].sel(lake_variable='Size')
    events = np.unique(trimmed_ds['dea_events'].dropna(dim='time').values)
    
    # Extract peaks and troughs if provided
    if peaks_troughs is not None:
        peaks = peaks_troughs['peaks']
        troughs = peaks_troughs['troughs']
    
    
    # ========== PLOTTING CONFIGURATION ==========
    # Grid spacing based on time range
    num_years = len(np.unique(trimmed_ds.time.dt.year.values))
    grid_spacing = 1 if num_years < 50 else 5
    
    # Colors for different elements
    colors = {
        'rolling_sum': '#b3cde0',      # Light blue
        'threshold1': '#011f4b',        # Dark blue
        'threshold2': 'maroon',         # Maroon
        'peaks': 'darkorange',          # Orange
        'troughs': 'green'              # Green
    }
    
    # Labels
    labels = {
        'rolling_sum': f'{window} {time_unit.capitalize()} Rolling Sum',
        'threshold1': f'Peak above {threshold1} mm',
        'threshold2': f'Peak above {threshold2} mm',
        'ylabel_left': f'Cumulative Rainfall ({rainfall_sum.units}/{window} {time_unit})',
        'ylabel_right': f'Lake Size {lake_size.units}'
    }
    
    
    # ========== PLOT RAINFALL DATA ==========
    # Plot base rolling sum
    ax_left.plot(
        rainfall_sum.time, rainfall_sum.values,
        linestyle='-', label=labels['rolling_sum'],
        alpha=0.5, color=colors['rolling_sum'], zorder=2
    )
    
    # Plot segments above threshold1
    ax_left.plot(
        rainfall_sum_above_threshold1.time, rainfall_sum_above_threshold1.values,
        linestyle='-', label=labels['threshold1'],
        alpha=0.5, color=colors['threshold1'], zorder=3
    )
    
    # Plot segments above threshold2
    ax_left.plot(
        rainfall_sum_above_threshold2.time, rainfall_sum_above_threshold2.values,
        linestyle='-', label=labels['threshold2'],
        alpha=0.5, color=colors['threshold2'], zorder=4
    )
    
    # Plot peaks and troughs if provided
    if peaks_troughs is not None:
        ax_left.plot(
            peaks.time, peaks.values,
            linestyle='', marker='o', label='Peaks',
            alpha=0.5, color=colors['peaks'], zorder=5
        )
        ax_left.plot(
            troughs.time, troughs.values,
            linestyle='', marker='o', label='Troughs',
            alpha=0.5, color=colors['troughs'], zorder=5
        )
    
    # Add threshold reference lines
    ax_left.axhline(y=threshold1, color=colors['threshold1'], linestyle='--', 
                   linewidth=2, zorder=1)
    ax_left.axhline(y=threshold2, color=colors['threshold2'], linestyle='--', 
                   linewidth=2, zorder=1)
    
    
    # ========== PLOT LAKE DATA ==========
    # Plot all lake observations
    ax_right.plot(
        lake_size.time.values, lake_size.values,
        color='grey', linestyle='', marker='o', markersize=8,
        alpha=0.4, zorder=1
    )
    
    # Highlight each detected event
    for event in events:
        event_sizes = lake_size.where(
            lake_size.dea_events == event, drop=True
        ).dropna(dim='time')
        
        ax_right.plot(
            event_sizes.time, event_sizes,
            linestyle='--', marker='o', markersize=6,
            color='black', label=f'Event {int(event)}', zorder=6
        )
        
        # Add event label at peak
        label_size = event_sizes.where(event_sizes.dea_event_max, drop=True)
        ax_right.text(
            label_size.time, label_size + 3,
            f'Event {int(event)}',
            fontsize=20, ha='center', va='bottom',
            color='black', zorder=6
        )
    
    # ========== FORMAT AXES ==========
    # Add legend if requested
    if show_legend:
        ax_left.legend(loc='upper left', fontsize=20)
    
    # Set y-axis labels and ticks
    ax_left.set_ylabel(labels['ylabel_left'], fontsize=20)
    ax_left.tick_params(axis='y', labelsize=20)
    
    ax_right.set_ylabel(labels['ylabel_right'], fontsize=20)
    ax_right.tick_params(axis='y', labelsize=20)
    
    # Format x-axis with yearly grid
    ax_left.grid(True, which='major', axis='both')
    ax_left.xaxis.set_major_locator(mdates.YearLocator(grid_spacing))
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_left.get_xticklabels(), rotation=90, ha='center', fontsize=20)        


def plot_rolling_sum_subplots(FigNo, ds, window, time_unit, threshold1, threshold2, 
                               event_timeframe, rain_timeframe, peaks_troughs=None):
    """
    Create a two-panel plot of rolling sum rainfall over different timeframes.
    
    Generates a figure with two subplots showing rolling sum rainfall and lake events.
    The top panel shows the event timeframe (typically shorter, recent period with
    satellite data), while the bottom panel shows the full rainfall record timeframe
    (typically longer, historical period).
    
    Parameters
    ----------
    FigNo : str
        figure number (e.g., 'Fig1').
    ds : xarray.Dataset
        Dataset containing:
            - 'rolling_sum_of_mean_catchment_rainfall': rolling sum rainfall
            - 'dea': lake variables with 'Size'
            - 'dea_events': event ID labels
            - 'rolling_peak_above_{threshold1}': boolean for threshold1
            - 'rolling_peak_above_{threshold2}': boolean for threshold2
    window : int
        Rolling window size for cumulative rainfall.
    time_unit : str
        Unit of time for rolling sum: 'days' or 'months'.
    threshold1 : float
        Lower rainfall threshold (mm) to highlight.
    threshold2 : float
        Higher rainfall threshold (mm) to highlight.
    event_timeframe : dict
        Dictionary with 'start' and 'end' keys for event period (top panel).
    rain_timeframe : dict
        Dictionary with 'start' and 'end' keys for full rainfall period (bottom panel).
    peaks_troughs : dict, optional
        Dictionary with 'peaks' and 'troughs' keys containing DataArrays.
        If provided, peaks and troughs are plotted on both panels.
    
    Output
    ------
    Saves PNG: '{output_dir}/{FigNo}_{Lake}_RollingSum_{window}{unit}-window.png'
    
    Raises
    ------
    KeyError
        If required variables are missing from dataset.
    ValueError
        If peaks_troughs is provided but not a dict or missing required keys.
    """
    # ========== VALIDATE INPUTS ==========
    required_vars = [
        'rolling_sum_of_mean_catchment_rainfall', 'dea', 'dea_events',
        f'rolling_peak_above_{threshold1}', f'rolling_peak_above_{threshold2}'
    ]
    missing_vars = [v for v in required_vars if v not in ds]
    if missing_vars:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing_vars)}')
    
    # Validate peaks_troughs if provided
    if peaks_troughs is not None:
        if not isinstance(peaks_troughs, dict):
            raise ValueError('peaks_troughs must be a dictionary')
        if 'peaks' not in peaks_troughs or 'troughs' not in peaks_troughs:
            raise ValueError('peaks_troughs must contain "peaks" and "troughs" keys')
    
    
    # ========== FILENAME CONSTRUCTION ==========
    filename_parts = [FigNo, Lake, 'RollingSum', f'{window}{time_unit[:-1]}-window']
    file_name = '_'.join(filename_parts)
    
    
    # ========== CREATE FIGURE AND SUBPLOTS ==========
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(40, 20))
    
    # Create twin axes for lake data
    ax1_2 = axes[0].twinx()  # Top panel
    ax2_2 = axes[1].twinx()  # Bottom panel
    
    # Top panel: Event timeframe (with legend)
    plot_rolling_sum(
        axes[0], ax1_2, ds,
        start_date=event_timeframe['start'],
        end_date=event_timeframe['end'],
        window=window,
        time_unit=time_unit,
        threshold1=threshold1,
        threshold2=threshold2,
        peaks_troughs=peaks_troughs,
        show_legend=True
    )
    
    # Bottom panel: Full rainfall timeframe (no legend)
    plot_rolling_sum(
        axes[1], ax2_2, ds,
        start_date=rain_timeframe['start'],
        end_date=rain_timeframe['end'],
        window=window,
        time_unit=time_unit,
        threshold1=threshold1,
        threshold2=threshold2,
        peaks_troughs=peaks_troughs,
        show_legend=False
    )
    
    # ========== SAVE FIGURE ==========
    plt.savefig(f'{output_dir}/{file_name}.png', bbox_inches='tight')
    plt.close()


def plot_rolling_sum_variable_window(ax_left, ax_right, ds, start_date, end_date, time_unit, threshold, window_min, window_max, show_legend=True):
    """
    Plot rolling cumulative sums of rainfall over a variable window and overlay lake events.
    
    This function calculates rolling sums of mean gridded rainfall for windows from window_min to window_max 
    with step size depending on time_unit. Peaks above a specified threshold are identified using troughs 
    to segment the time series. Cumulative rainfall and segments above the threshold are plotted on ax_left, 
    while the full lake size time series with events is plotted on ax_right.
    
    Parameters
    ----------
    ax_left : matplotlib.axes.Axes
        Axis on which to plot cumulative rainfall.
    ax_right : matplotlib.axes.Axes
        Axis on which to plot lake size and events.
    ds : xarray.Dataset
        Dataset containing variables:
            - 'mean_catchment_rainfall': daily or monthly rainfall
            - 'dea': lake size time series
            - 'dea_events': flood event IDs
    start_date : str
        Start year for plotting; affects x-axis grid spacing.
    time_unit : str
        Unit of the rolling window ('days' or 'months').
    threshold : float
        Rainfall threshold to highlight on the plot.
    window_min : int
        Minimum rolling window size.
    window_max : int
        Maximum rolling window size.
    
    Depends on
    ----------
    identify_peaks_and_troughs : function
        Identifies peaks and troughs in a 1D rainfall series.
    truncate_colormap : function
        Truncates a matplotlib colormap to avoid very light or very dark colors.
    
    Raises
    ------
    ValueError: If time_unit is not 'days' or 'months', or if peak/trough counts are inconsistent.
    KeyError: If required variables ('mean_catchment_rainfall', 'dea', 'dea_events') are missing from ds.
    """   
    # ========== VALIDATE INPUTS ==========
    # 1. Check required variables
    required_vars = ['dea', 'dea_events', 'mean_catchment_rainfall']
    missing_vars = [v for v in required_vars if v not in ds]
    if missing_vars:
        raise KeyError(f'Required variables missing from dataset: {missing_vars}')
    
    # 2. Validate time_unit and set parameters
    if time_unit == 'days':
        distance_val = 180
        time_step = 16
    elif time_unit == 'months':
        distance_val = 6
        time_step = 1
    else:
        raise ValueError('time_unit must be "days" or "months"')
    
    # 3. Validate window range
    if window_min >= window_max:
        raise ValueError('window_min must be less than window_max')


    # ========== DATA PREPARATION ==========
    # 1. Trim ds
    trimmed_ds = ds.sel(time=slice(start_date, end_date))

    # 2. Extract lake data
    lake_size = trimmed_ds['dea'].sel(lake_variable='Size')
    events = np.unique(trimmed_ds['dea_events'].dropna(dim='time').values)
    
    # 3. Extract
    catchment_rainfall = trimmed_ds['mean_catchment_rainfall'].reset_coords(drop=True).compute()
    
    # 4. Calculate number of windows to plot
    num_windows = int(((window_max - window_min) / time_step) + 1)


    # ========== PLOTTING CONFIGURATION ==========
    # Grid spacing based on number of years
    num_years = len(np.unique(trimmed_ds.time.dt.year.values))
    grid_spacing = 1 if num_years < 50 else 5
    
    # Colormaps
    norm = plt.Normalize(vmin=1, vmax=num_windows)
    cmap_blue = truncate_colormap(colormaps['Blues'], minval=0.2, maxval=1.0)
    colours_blue = [cmap_blue(norm(i + 1)) for i in range(num_windows)]
    cmap_red = truncate_colormap(colormaps['Reds'], minval=0.2, maxval=1.0)
    colours_red = [cmap_red(norm(i + 1)) for i in range(num_windows)]
    
    # Labels
    labels = {
        'title': f'Rolling Cumulative Sum of Rainfall\n(red: peak above {threshold} {catchment_rainfall.units})',
        'rainfall': f'Cumulative Rainfall ({catchment_rainfall.units})',
        'lake': f'Lake Size {lake_size.units}'
    }
    
    # Collect legend handles
    blue_handles = []
    red_handles = []
    blue_labels = []
    red_labels = []

    # ========== CALCULATE AND PLOT ROLLING SUMS FOR EACH WINDOW ==========
    for window in range(window_min,window_max+1,time_step):
        i = int((window-window_min)/time_step)
    
        # Step 1: Calculate rolling cumulative sum of mean rainfall
        rainfall_sum = catchment_rainfall.rolling(time=window, center=False).sum()
        
        # Step 2: check if peaks are above threshold
        # Step 2a: Troughs are used to segment the time series into rainfall episodes
        peaks_and_troughs = identify_peaks_and_troughs(rainfall_sum, distance_val)
    
        # Sanity check
        if not (len(peaks_and_troughs['peaks']) == len(peaks_and_troughs['troughs']) or len(peaks_and_troughs['peaks']) == len(peaks_and_troughs['troughs']) - 1):
            raise ValueError('Number of peaks must be equal to or one less than number of troughs.')     
    
        # Step 2b: Assign segment IDs based on trough positions
        # Create the segment boundaries
        trough_ids_plus_start = np.concatenate(([0], peaks_and_troughs['troughs']))
        trough_ids_plus_end = np.concatenate((peaks_and_troughs['troughs'], [len(rainfall_sum)]))
        
        # Calculate segment lengths
        segment_lengths = trough_ids_plus_end - trough_ids_plus_start
        
        # Create segment IDs by repeating each ID by its segment length
        segment_id_values = np.repeat(np.arange(len(segment_lengths)), segment_lengths)
        
        # Create DataArray and assign as coordinate
        segment_ids = xr.DataArray(segment_id_values, coords={'time': rainfall_sum.time}, 
                                   dims=['time'])
        
        # Sanity check
        assert segment_ids.time.size == rainfall_sum.time.size, \
            f"Size mismatch! segment_ids has {segment_ids.time.size} time steps, " \
            f"but rainfall_sum has {rainfall_sum.time.size}"
            
        # Add as coordinate to the copy
        rainfall_sum = rainfall_sum.assign_coords(segment_id=segment_ids)
 
        # Step 2c: Create boolean data arrays depending on threshold
        # Boolean array determening peak values
        above_thresh = rainfall_sum[peaks_and_troughs['peaks']] > threshold
        above_thresh_plus_start_and_end = np.concatenate(([False], above_thresh, [False])) # Before first trough and after last trough
        
        # Boolean da determening peak values + time before and after first trough added
        above_thresh_plus_start_and_end_da = xr.DataArray(
            above_thresh_plus_start_and_end,
            dims=['segment_id'],
            coords={'segment_id': np.arange(len(above_thresh_plus_start_and_end), dtype=float)}
        )
                
        # Step 2d: maps each segment's boolean value to all times in that segment
        peak_above_thresh = above_thresh_plus_start_and_end_da.sel(segment_id=rainfall_sum.segment_id)
        
        # Step 2e: Assign boolean coordinates to rolling_sum dynamically based on threshold values (3b)
        rainfall_sum = rainfall_sum.assign_coords({
            f'rolling_peak_above_{threshold}': peak_above_thresh
            })

        # Step 3: Filter out segments that exceed the threshold at their peak
        rainfall_sum_threshold = rainfall_sum.where(rainfall_sum[f'rolling_peak_above_{threshold}'], drop=False)
 
        # Step 4: Plot
        # Only label windows divisible by 16
        if window % 16 == 0:
            label_text = f'{window} {time_unit}'
        else:
            label_text = None
        
        # Plot and save handles
        line_blue = ax_left.plot(rainfall_sum.time, rainfall_sum.values, 
                                 linestyle='-', alpha=0.5, color=colours_blue[i], zorder=2)[0]
    
        line_red = ax_left.plot(rainfall_sum_threshold.time, rainfall_sum_threshold.values,
                 linestyle='-', alpha=0.5, color=colours_red[i], zorder=3)[0]
        
        # Save for legend
        if label_text is not None:
            blue_handles.append(line_blue)
            blue_labels.append(label_text)
            red_handles.append(line_red)
            red_labels.append(label_text)
        
    # ========== PLOT LAKE DATA ==========
    # Plot full time series of lake size
    ax_right.plot(lake_size.time.values, lake_size.values,
                 color='grey', linestyle='', marker='o', markersize=8, alpha=0.4, zorder=1)
    
    # Plot events with labels
    for event in events:
        event_sizes = lake_size.where(lake_size.dea_events == event, drop=True).dropna(dim='time')
        ax_right.plot(event_sizes.time, event_sizes, linestyle='--', marker='o', markersize=6,
                     color='black', label=f'Event {int(event)}', zorder=5)
    
        label_size = event_sizes.where(event_sizes.dea_event_max, drop=True)
        ax_right.text(label_size.time, label_size + 3, f'Event {int(event)}',
                     fontsize=20, ha='center', va='bottom',
                     color='black', zorder=6)


    # ========== FORMAT AXES ==========
    # Create two-column legend: blues left, reds right
    if show_legend:
        all_handles = blue_handles + red_handles
        all_labels = blue_labels + red_labels
        
        ax_left.legend(all_handles, all_labels, title=labels['title'], 
                      loc='upper left', fontsize=15, title_fontsize=16, 
                      ncol=2)  # Two columns
    
    # Set y-axis label 
    ax_left.set_ylabel(labels['rainfall'], fontsize=20)
    ax_right.set_ylabel(labels['lake'], fontsize=20)
    
    # Format x-axis (dates)
    ax_left.grid(True, which='major', axis='both')
    ax_left.xaxis.set_major_locator(mdates.YearLocator(grid_spacing))
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_left.get_xticklabels(), rotation=90, ha='center', fontsize=20)
    
    # Set tick label sizes
    ax_left.tick_params(axis='y', labelsize=20)
    ax_right.tick_params(axis='y', labelsize=20)
        

def plot_rolling_sum_variable_window_subplots(FigNo, ds, time_unit, threshold, window_min, window_max, 
                                               event_timeframe, rain_timeframe):
    """
    Create a two-panel plot of rolling sums across multiple window sizes.
    
    Generates a figure with two subplots showing rolling sum rainfall and lake events
    across a range of window sizes. The top panel shows the event timeframe (typically
    shorter, recent period with satellite data), while the bottom panel shows the full
    rainfall record timeframe (typically longer, historical period).
    
    Parameters
    ----------
    FigNo : str
        figure number (e.g., 'Fig1').
    ds : xarray.Dataset
        Dataset containing:
            - 'mean_catchment_rainfall': daily or monthly rainfall
            - 'dea': lake variables with 'Size'
            - 'dea_events': event ID labels
    time_unit : str
        Unit of rolling window: 'days' or 'months'.
    threshold : float
        Rainfall threshold (mm) for color coding segments.
    window_min : int
        Minimum rolling window size.
    window_max : int
        Maximum rolling window size.
    event_timeframe : dict
        Dictionary with 'start' and 'end' keys for event period (top panel).
    rain_timeframe : dict
        Dictionary with 'start' and 'end' keys for full rainfall period (bottom panel).
    
    Output
    ------
    Saves PNG: '{output_dir}/{FigNo}_{Lake}_RollingSum_{window_min}to{window_max}{unit}-window.png'
    
    Raises
    ------
    KeyError
        If required variables are missing from dataset.
    ValueError
        If time_unit is invalid or window_min >= window_max.
    """
    # ========== VALIDATE INPUTS ==========
    required_vars = ['dea', 'dea_events', 'mean_catchment_rainfall']
    missing_vars = [v for v in required_vars if v not in ds]
    if missing_vars:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing_vars)}')
    
    # Validate time_unit
    if time_unit not in ['days', 'months']:
        raise ValueError('time_unit must be "days" or "months"')
    
    # Validate window range
    if window_min >= window_max:
        raise ValueError('window_min must be less than window_max')
    
    
    # ========== FILENAME CONSTRUCTION ==========
    filename_parts = [FigNo, Lake, 'RollingSum', f'{window_min}to{window_max}{time_unit[:-1]}-window']
    file_name = '_'.join(filename_parts)
    
    
    # ========== CREATE FIGURE AND SUBPLOTS ==========
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(40, 20))
    
    # Create twin axes for lake data
    ax1_2 = axes[0].twinx()  # Top panel
    ax2_2 = axes[1].twinx()  # Bottom panel
    
    # Top panel: Event timeframe (with legend)
    plot_rolling_sum_variable_window(
        axes[0], ax1_2, ds,
        start_date=event_timeframe['start'],
        end_date=event_timeframe['end'],
        time_unit=time_unit,
        threshold=threshold,
        window_min=window_min,
        window_max=window_max,
        show_legend=True
    )
    
    # Bottom panel: Full rainfall timeframe (no legend)
    plot_rolling_sum_variable_window(
        axes[1], ax2_2, ds,
        start_date=rain_timeframe['start'],
        end_date=rain_timeframe['end'],
        time_unit=time_unit,
        threshold=threshold,
        window_min=window_min,
        window_max=window_max,
        show_legend=False
    )
        
    # ========== SAVE FIGURE ==========
    plt.savefig(f'{output_dir}/{file_name}.png', bbox_inches='tight')
    plt.close()
        

def plot_NT_event_maps(ax_no, da, timeframe, event_timeframe, lake_mask, threshold, event=False):
    """
    Plot Northern Territory rainfall map for event or non-event periods.
    
    Subsets rainfall data based on timeframe and event status, calculates the temporal
    mean, and plots as a map with lake boundaries. Handles different event classification
    methods for pre-DEA (threshold-based) and post-DEA (satellite-based) periods.
    
    Parameters
    ----------
    ax_no : cartopy.mpl.geoaxes.GeoAxesSubplot
        Axis on which to plot the map.
    da : xarray.DataArray
        Rainfall data with coordinates 'time', 'lat', 'lon', and coordinates
        'above_threshold' and (optionally) 'dea_events'.
    timeframe : str
        Time period to plot: 'pre-dea', 'post-dea', or 'entire'.
    event_timeframe : dict
        Dictionary with 'start' and 'end' keys containing DEA period years as strings.
    lake_mask : xarray.DataArray
        Boolean mask of lake locations with coordinates 'lat' and 'lon'.
    threshold : float
        Rainfall threshold (mm) for colormap vmax.
    event : bool, default False
        If True, plots periods with lake-filling events.
        If False, plots non-event periods.
    
    Returns
    -------
    matplotlib.collections.QuadMesh
        Image object for colorbar creation.
    
    Raises
    ------
    KeyError
        If required coordinates are missing from da or timeframe is invalid.
    ValueError
        If DataArray starts after DEA period start.
    
    Notes
    -----
    Event classification differs by period:
    - Pre-DEA: Uses 'above_threshold' coordinate (threshold-based classification)
    - Post-DEA: Uses 'dea_events' coordinate (satellite-observed events)
    This accounts for the lack of satellite data before the DEA observation period.
    """
    # ========== VALIDATE INPUTS ==========
    required_vars = ['above_threshold']
    if event:
        required_vars.append('dea_events')
    
    missing_vars = [v for v in required_vars if v not in da.coords]
    if missing_vars:
        raise KeyError(f'Required coordinates missing from DataArray: {", ".join(missing_vars)}')
    
    # Validate timeframe
    if timeframe not in ['dea', 'pre-dea', 'entire']:
        raise KeyError(f'timeframe must be one of: "dea", "pre-dea", "entire", got "{timeframe}"')
    
    # Ensure data covers pre-DEA period
    entire_start = str(da.time[0].dt.year.item())
    dea_start = event_timeframe['start']
    if int(entire_start) > int(dea_start):
        raise ValueError(
            f'DataArray must start before DEA period. da starts: {entire_start}, DEA starts: {dea_start}'
        )
    
    
    # ========== DATA PREPARATION ==========
    # Split data at DEA period boundary
    year_prior_dea_start = str(int(dea_start) - 1)
    pre_dea = da.sel(time=slice(entire_start, year_prior_dea_start))
    post_dea = da.sel(time=slice(dea_start, event_timeframe['end']))
    
    # Filter by event status using appropriate method for each period
    if event:
        # Event periods: threshold-based for pre-DEA, satellite-based for post-DEA
        pre_dea_subset = pre_dea.where(pre_dea.above_threshold > 0, drop=True)
        post_dea_subset = post_dea.where(~post_dea.dea_events.isnull().compute(), drop=True)
    else:
        # Non-event periods
        pre_dea_subset = pre_dea.where(pre_dea.above_threshold == 0, drop=True)
        post_dea_subset = post_dea.where(post_dea.dea_events.isnull().compute(), drop=True)
    
    # Select subset based on requested timeframe
    if timeframe == 'pre-dea':
        subset = pre_dea_subset
    elif timeframe == 'post-dea':
        subset = post_dea_subset
    else:  # entire
        subset = xr.concat([pre_dea_subset, post_dea_subset], dim='time')
    
    # Calculate temporal mean
    mean_data = subset.mean(dim='time', skipna=True)
    
    
    # ========== PLOT MAP ==========
    # Add coastlines
    ax_no.coastlines(color='black')
    
    # Plot rainfall data
    im0 = ax_no.pcolormesh(
        mean_data.coords['lon'].values,
        mean_data.coords['lat'].values,
        mean_data.values,
        cmap='Blues',
        norm=Normalize(vmin=0, vmax=threshold),
        transform=ccrs.PlateCarree()
    )
    
    # Add lake boundary
    ax_no.contour(
        lake_mask.lon, lake_mask.lat, lake_mask.values,
        colors='k', linewidths=0.75, transform=ccrs.PlateCarree()
    )
    
    # Add gridlines with labels
    gl = ax_no.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = LongitudeLocator()
    gl.ylocator = LatitudeLocator()
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    
    # Return image for colorbar
    return im0


def plot_NT_event_maps_subplots(FigNo, da, event_timeframe, lake_mask, window_size, time_unit, threshold):
    """
    Create a 3x2 grid of Northern Territory rainfall composite maps.
    
    Generates composite maps showing mean rainfall patterns across three time periods
    (entire record, pre-DEA, DEA period) and two event categories (events vs non-events).
    Each panel shows the temporal mean rainfall for the specified subset.
    
    Parameters
    ----------
    FigNo : str
        figure number (e.g., 'Fig1').
    da : xarray.DataArray
        Rainfall data with rolling window sum and event coordinates
        ('above_threshold', 'dea_events').
    event_timeframe : dict
        Dictionary with 'start' and 'end' keys containing DEA period years as strings.
    lake_mask : xarray.DataArray
        Boolean mask of lake locations with 'lat' and 'lon' coordinates.
    window_size : int
        Size of rolling window used (for colorbar label).
    time_unit : str
        Unit of rolling window: 'days' or 'months'.
    threshold : float
        Rainfall threshold (mm) for colormap vmax.
    
    Output
    ------
    Saves PNG: '{output_dir}/{FigNo}_{Lake}_NAus_composite_maps.png'
    
    Raises
    ------
    KeyError
        If required coordinates are missing from DataArray.
    ValueError
        If DataArray starts after DEA period start.
    
    Notes
    -----
    The grid layout is:
    - Rows: Entire record, Pre-DEA period, DEA period
    - Columns: Non-event years, Event years
    
    End dates are adjusted to ensure data coverage extends through the full rolling
    window, accounting for the window size in the date calculation.
    """
    # ========== VALIDATE INPUTS ==========
    required_vars = ['above_threshold', 'dea_events']
    missing_vars = [v for v in required_vars if v not in da.coords]
    if missing_vars:
        raise KeyError(f'Required coordinates missing from DataArray: {", ".join(missing_vars)}')
    
    # Ensure data covers pre-DEA period
    entire_start = str(da.time[0].dt.year.item())
    dea_start = event_timeframe['start']
    if int(entire_start) > int(dea_start):
        raise ValueError(
            f'DataArray must start before DEA period. da starts: {entire_start}, DEA starts: {dea_start}'
        )
    
    
    # ========== DATA PREPARATION ==========
    rainfall_units = da.units
    
    
    # ========== FILENAME CONSTRUCTION ==========
    filename_parts = [FigNo, Lake, 'NAus_composite_maps']
    file_name = '_'.join(filename_parts)
    
    
    # ========== PLOTTING CONFIGURATION ==========
    # Calculate end date accounting for window coverage
    dea_end = event_timeframe['end']
    entire_end = str(
        (da.time[-1] + pd.Timedelta(**{time_unit: window_size})).dt.year.item()
    )
    
    # Use earlier of data end or DEA end
    end = entire_end if int(entire_end) < int(dea_end) else dea_end
    yr_prior_dea = str(int(dea_start) - 1)
    
    # Subplot layout labels
    rows = [
        f'{entire_start}\n–\n{end}',           # Entire record
        f'{entire_start}\n–\n{yr_prior_dea}',  # Pre-DEA
        f'{dea_start}\n–\n{end}'               # DEA period
    ]
    columns = ['Non-Event Years', 'Event Years']
    
    # Plot arguments: (timeframe, event_flag)
    plot_args = [
        ('entire', False),   # Entire record, non-events
        ('entire', True),    # Entire record, events
        ('pre-dea', False),  # Pre-DEA, non-events
        ('pre-dea', True),   # Pre-DEA, events
        ('dea', False),      # DEA period, non-events
        ('dea', True)        # DEA period, events
    ]
    
    # Colorbar configuration
    num_ticks = int(threshold / 100) + 1
    colorbar_config = {
        'position': [0.25, 0.05, 0.5, 0.02],
        'ticks': np.linspace(0, threshold, num_ticks),
        'label': f'Rainfall ({rainfall_units}/ {window_size} {time_unit})',
        'fontsize': 10
    }
    
    
    # ========== CREATE FIGURE AND PLOT ==========
    fig, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(30, 18),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    axes = axes.flatten()
    
    # Generate each subplot
    for ax, (timeframe, ev) in zip(axes, plot_args):
        im0 = plot_NT_event_maps(
            ax, da,
            timeframe=timeframe,
            event_timeframe=event_timeframe,
            lake_mask=lake_mask,
            threshold=threshold,
            event=ev
        )
    
    
    # ========== ADD COLORBAR ==========
    cbar_ax = fig.add_axes(colorbar_config['position'])
    cbar_ax.set_frame_on(False)
    cb1 = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
    cb1.set_label(colorbar_config['label'], fontsize=colorbar_config['fontsize'])
    cb1.set_ticks(colorbar_config['ticks'])
    cb1.ax.tick_params(labelsize=colorbar_config['fontsize'])
    cb1.outline.set_color('lightgray')
    
    
    # ========== ADD ROW AND COLUMN LABELS ==========
    for i, ax in enumerate(axes):
        r = i // 2  # Row index
        c = i % 2   # Column index
        
        # Add row labels on leftmost column
        if c == 0:
            ax.text(
                -0.05, 0.5, rows[r],
                transform=ax.transAxes,
                fontsize=14, fontweight='bold',
                va='center', ha='right',
                multialignment='center'
            )
        
        # Add column labels on top row
        if r == 0:
            ax.text(
                0.5, 1.05, columns[c],
                transform=ax.transAxes,
                fontsize=14, fontweight='bold',
                va='bottom', ha='center'
            )
        
    # ========== SAVE FIGURE ==========
    plt.savefig(f'{output_dir}/{file_name}.png', bbox_inches='tight')
    plt.close()
    

def plot_rolling_sum_with_driver(ax_left, ax_right1, ax_right2, ds, start_date, end_date, 
                                  driver, driver_index, indices_dict, window, time_unit, 
                                  threshold1, threshold2, show_legend=True):
    """
    Plot rolling sum rainfall, lake events, and climate driver on a three-axis figure.
    
    Creates a plot with three y-axes showing:
    1. Left axis: Climate driver index (ENSO or IPO) as filled positive/negative areas
    2. Right axis 1: Rolling cumulative rainfall with threshold highlighting
    3. Right axis 2: Lake size time series with detected events
    
    Parameters
    ----------
    ax_left : matplotlib.axes.Axes
        Axis for plotting climate driver.
    ax_right1 : matplotlib.axes.Axes
        Axis for plotting rolling cumulative rainfall.
    ax_right2 : matplotlib.axes.Axes
        Axis for plotting lake size.
    ds : xarray.Dataset
        Dataset containing:
            - 'rolling_sum_of_mean_catchment_rainfall': rolling sum rainfall
            - 'dea': lake variables with 'Size'
            - 'dea_events': event ID labels
            - 'dea_event_max': boolean marking event peaks
            - 'enso' or 'ipo': climate driver data
            - 'rolling_peak_above_{threshold1}' and 'rolling_peak_above_{threshold2}'
    start_date : str
        Start year for plotting.
    end_date : str
        End year for plotting.
    driver : str
        Climate driver to plot: 'enso' or 'ipo'.
    driver_index : str
        Name of specific driver index (e.g., 'Nino3.4', 'TPI').
    indices_dict : dict
        Nested dictionary mapping driver types to their available indices.
    window : int
        Rolling window size for rainfall calculation.
    time_unit : str
        Unit of rolling window: 'days' or 'months'.
    threshold1 : float
        Lower rainfall threshold (mm) to highlight.
    threshold2 : float
        Higher rainfall threshold (mm) to highlight.
    show_legend : bool, default True
        If True, displays combined legend on left axis.
    
    Raises
    ------
    KeyError
        If required variables are missing from dataset.
    ValueError
        If driver is not 'enso' or 'ipo'.
    
    Notes
    -----
    The third axis (ax_right2) is positioned with an outward spine offset to avoid
    overlap with the second axis. Climate driver values are filled above/below zero
    with different colors to show positive/negative phases.
    """
    # ========== VALIDATE INPUTS ==========
    required_vars = [
        'rolling_sum_of_mean_catchment_rainfall', 'dea', 'dea_events', 'dea_event_max',
        f'rolling_peak_above_{threshold1}', f'rolling_peak_above_{threshold2}'
    ]
    missing_vars = [v for v in required_vars if v not in ds]
    if missing_vars:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing_vars)}')
    
    # Validate driver type
    if driver not in ['enso', 'ipo']:
        raise ValueError(f'driver must be "enso" or "ipo", got "{driver}"')
    
    # Check driver data exists
    driver_var = driver  # 'enso' or 'ipo'
    if driver_var not in ds:
        raise KeyError(f'Driver variable "{driver_var}" missing from dataset')
    
    
    # ========== DATA PREPARATION ==========
    # Subset to date range
    trimmed_ds = ds.sel(time=slice(start_date, end_date))
    
    # Extract specific driver index
    index = indices_dict[driver][driver_index]
    if driver == 'enso':
        driver_da = trimmed_ds['enso'].sel(enso_indices=index).sel(
            time=slice(f'{start_date}-01-01', trimmed_ds.time[-1])
        )
    else:  # ipo
        driver_da = trimmed_ds['ipo'].sel(ipo_indices=index).sel(
            time=slice(f'{start_date}-01-01', trimmed_ds.time[-1])
        )
    
    # Extract rainfall data
    rainfall_sum = trimmed_ds['rolling_sum_of_mean_catchment_rainfall']
    rainfall_sum_above_threshold1 = rainfall_sum.where(
        rainfall_sum[f'rolling_peak_above_{threshold1}']
    )
    rainfall_sum_above_threshold2 = rainfall_sum.where(
        rainfall_sum[f'rolling_peak_above_{threshold2}']
    )
    
    # Extract lake data
    lake_size = trimmed_ds['dea'].sel(lake_variable='Size')
    events = np.unique(trimmed_ds['dea_events'].dropna(dim='time').values)
    
    
    # ========== PLOTTING CONFIGURATION ==========
    # Grid spacing based on time range
    num_years = len(np.unique(trimmed_ds.time.dt.year.values))
    grid_spacing = 1 if num_years < 50 else 5
    
    # Colors
    colors = {
        'driver_positive': 'lightcoral',   # Positive driver phase
        'driver_negative': 'lightblue',    # Negative driver phase
        'rolling_sum': '#b3cde0',          # Base rolling sum
        'threshold1': '#011f4b',           # Dark blue
        'threshold2': 'maroon'             # Maroon
    }
    
    # Labels
    labels = {
        'driver': f'{driver.upper()} ({index})',
        'rainfall': f'{window} {time_unit.capitalize()} Cumulative Precipitation ({rainfall_sum.units}/{window} {time_unit})',
        'lake': f'Lake Size {lake_size.units}',
        'driver_positive': f'Positive {driver.upper()}',
        'driver_negative': f'Negative {driver.upper()}',
        'rolling_sum': f'{window} {time_unit.capitalize()} Rolling Sum',
        'threshold1': f'Peak above {threshold1} mm',
        'threshold2': f'Peak above {threshold2} mm'
    }
    
    # Axis ticks configuration
    if driver == 'enso':
        driver_ticks = np.arange(-3, 3.1, 1)
    else:  # ipo
        driver_ticks = np.arange(-0.6, 0.7, 0.2)
    
    driver_tick_labels = [f'{v:.1f}' for v in driver_ticks]
    rainfall_ticks = np.arange(-1500, 1501, 500)
    lake_ticks = np.arange(-750, 751, 250)
    
    # Third axis positioning
    right_spine_offset = 120
    
    
    # ========== PLOT DATA ==========
    # Position third axis spine
    ax_right2.spines['right'].set_position(('outward', right_spine_offset))
    
    # Plot climate driver (filled areas for positive/negative)
    ax_left.fill_between(
        driver_da.time, driver_da.values,
        where=driver_da.values >= 0,
        label=labels['driver_positive'],
        color=colors['driver_positive'],
        alpha=0.5, zorder=1
    )
    ax_left.fill_between(
        driver_da.time, driver_da.values,
        where=driver_da.values < 0,
        label=labels['driver_negative'],
        color=colors['driver_negative'],
        alpha=0.5, zorder=1
    )
    
    # Plot rolling cumulative rainfall
    ax_right1.plot(
        rainfall_sum.time, rainfall_sum.values,
        linestyle='-', label=labels['rolling_sum'],
        alpha=0.5, color=colors['rolling_sum'], zorder=3
    )
    
    ax_right1.plot(
        rainfall_sum_above_threshold1.time, rainfall_sum_above_threshold1.values,
        linestyle='-', label=labels['threshold1'],
        alpha=0.5, color=colors['threshold1'], zorder=4
    )
    
    ax_right1.plot(
        rainfall_sum_above_threshold2.time, rainfall_sum_above_threshold2.values,
        linestyle='-', label=labels['threshold2'],
        alpha=0.5, color=colors['threshold2'], zorder=5
    )
    
    # Add threshold reference lines
    ax_right1.axhline(y=threshold1, color=colors['threshold1'], linestyle='--', 
                     linewidth=2, zorder=2)
    ax_right1.axhline(y=threshold2, color=colors['threshold2'], linestyle='--', 
                     linewidth=2, zorder=2)
    
    # Plot lake size
    ax_right2.plot(
        lake_size.time.values, lake_size.values,
        color='grey', linestyle='', marker='o', markersize=8,
        alpha=0.4, zorder=2
    )
    
    # Highlight each detected event
    for event in events:
        event_sizes = lake_size.where(
            lake_size.dea_events == event, drop=True
        ).dropna(dim='time')
        
        ax_right2.plot(
            event_sizes.time, event_sizes,
            linestyle='--', marker='o', markersize=6,
            color='black', label=f'Event {int(event)}', zorder=6
        )
        
        # Add event label at peak
        label_size = event_sizes.where(event_sizes.dea_event_max, drop=True)
        ax_right2.text(
            label_size.time, label_size + 3,
            f'Event {int(event)}',
            fontsize=20, ha='center', va='bottom',
            color='black', zorder=7
        )
    
    
    # ========== FORMAT AXES ==========
    # Add combined legend if requested
    if show_legend:
        handles_left, labels_left = ax_left.get_legend_handles_labels()
        handles_right, labels_right = ax_right1.get_legend_handles_labels()
        
        all_handles = handles_left + handles_right
        all_labels = labels_left + labels_right
        
        ax_left.legend(all_handles, all_labels, loc='lower left', fontsize=20)
    
    # Set y-axis labels
    ax_left.set_ylabel(labels['driver'], fontsize=20)
    ax_right1.set_ylabel(labels['rainfall'], fontsize=20)
    ax_right2.set_ylabel(labels['lake'], fontsize=20)
    
    # Set y-axis ticks
    ax_left.set_yticks(driver_ticks)
    ax_left.set_yticklabels(driver_tick_labels, fontsize=20)
    
    ax_right1.set_yticks(rainfall_ticks)
    ax_right1.set_yticklabels([str(t) for t in rainfall_ticks], fontsize=20)
    
    ax_right2.set_yticks(lake_ticks)
    ax_right2.set_yticklabels([str(t) for t in lake_ticks], fontsize=20)
    
    # Format x-axis with yearly grid
    ax_left.grid(True, which='major', axis='both')
    ax_left.xaxis.set_major_locator(mdates.YearLocator(grid_spacing))
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_left.get_xticklabels(), rotation=90, ha='center', fontsize=20)
        

def plot_rolling_sum_with_driver_subplots(FigNo, ds, driver, driver_index, indices_dict, window, 
                                           time_unit, threshold1, threshold2, 
                                           event_timeframe, rain_timeframe):
    """
    Create a two-panel plot of rolling sum rainfall with climate driver.
    
    Generates a figure with two subplots, each showing three y-axes (climate driver,
    rainfall, and lake size). The top panel shows the event timeframe (typically
    shorter, recent period with satellite data), while the bottom panel shows the full
    rainfall record timeframe (typically longer, historical period).
    
    Parameters
    ----------
    FigNo : str
        figure number (e.g., 'Fig1').
    ds : xarray.Dataset
        Dataset containing:
            - 'rolling_sum_of_mean_catchment_rainfall': rolling sum rainfall
            - 'dea': lake variables with 'Size'
            - 'dea_events': event ID labels
            - 'enso' or 'ipo': climate driver data
    driver : str
        Climate driver to plot: 'enso' or 'ipo'.
    driver_index : str
        Name of specific driver index (e.g., 'Nino3.4', 'TPI').
    indices_dict : dict
        Nested dictionary mapping driver types to their available indices.
    window : int
        Rolling window size for rainfall calculation.
    time_unit : str
        Unit of rolling window: 'days' or 'months'.
    threshold1 : float
        Lower rainfall threshold (mm) to highlight.
    threshold2 : float
        Higher rainfall threshold (mm) to highlight.
    event_timeframe : dict
        Dictionary with 'start' and 'end' keys for event period (top panel).
    rain_timeframe : dict
        Dictionary with 'start' and 'end' keys for full rainfall period (bottom panel).
    
    Output
    ------
    Saves PNG: '{output_dir}/{FigNo}_{Lake}_RollingSum_{window}{unit}-window_{DRIVER}_{index}.png'
    
    Raises
    ------
    KeyError
        If required variables are missing from dataset.
    ValueError
        If driver is not 'enso' or 'ipo'.
    """
    # ========== VALIDATE INPUTS ==========
    required_vars = ['rolling_sum_of_mean_catchment_rainfall', 'dea', 'dea_events']
    missing_vars = [v for v in required_vars if v not in ds]
    if missing_vars:
        raise KeyError(f'Required variables missing from dataset: {", ".join(missing_vars)}')
    
    if driver not in ['enso', 'ipo']:
        raise ValueError(f'driver must be "enso" or "ipo", got "{driver}"')
    
    
    # ========== FILENAME CONSTRUCTION ==========
    filename_parts = [FigNo, Lake, 'RollingSum', f'{window}{time_unit[:-1]}-window', 
                     driver.upper(), driver_index]
    file_name = '_'.join(filename_parts)
    
    
    # ========== CREATE FIGURE AND SUBPLOTS ==========
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(40, 20))
    
    # Create twin axes for three y-axes per panel
    ax1_2 = axes[0].twinx()  # Top panel: rainfall axis
    ax1_3 = axes[0].twinx()  # Top panel: lake axis
    ax2_2 = axes[1].twinx()  # Bottom panel: rainfall axis
    ax2_3 = axes[1].twinx()  # Bottom panel: lake axis
    
    
    # ========== PLOT SUBPLOTS ==========
    # Top panel: Event timeframe (no legend)
    plot_rolling_sum_with_driver(
        axes[0], ax1_2, ax1_3, ds,
        start_date=event_timeframe['start'],
        end_date=event_timeframe['end'],
        driver=driver,
        driver_index=driver_index,
        indices_dict=indices_dict,
        window=window,
        time_unit=time_unit,
        threshold1=threshold1,
        threshold2=threshold2,
        show_legend=False
    )
    
    # Bottom panel: Full rainfall timeframe (with legend)
    plot_rolling_sum_with_driver(
        axes[1], ax2_2, ax2_3, ds,
        start_date=rain_timeframe['start'],
        end_date=rain_timeframe['end'],
        driver=driver,
        driver_index=driver_index,
        indices_dict=indices_dict,
        window=window,
        time_unit=time_unit,
        threshold1=threshold1,
        threshold2=threshold2,
        show_legend=True
    )
    
    # ========== SAVE FIGURE ==========
    plt.savefig(f'{output_dir}/{file_name}.png', bbox_inches='tight')
    plt.close()


#### ========= Functions: Plot formatting =========
def truncate_colormap(cmap, minval=0.2, maxval=1.0, n=256):
    """
    Truncate a matplotlib colormap to avoid very light or very dark colors.
    
    This function creates a new colormap using only a subset of the input colormap's range.
    
    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        A matplotlib colormap instance to truncate.
    minval : float, optional
        Lower bound of the colormap to keep (default 0.2, must be between 0 and 1).
    maxval : float, optional
        Upper bound of the colormap to keep (default 1.0, must be between 0 and 1).
    n : int, optional
        Number of discrete color levels in the new colormap (default 256).
    
    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        A truncated version of the input colormap.
    
    Raises
    ------
    ValueError
        If minval or maxval are not between 0 and 1, or if minval >= maxval.
    """
    
    if not (0 <= minval < maxval <= 1):
        raise ValueError('minval and maxval must satisfy 0 <= minval < maxval <= 1')
    
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


#%% Load Data

# ========== LOAD LAKE MASKS ==========
# Load coarse resolution mask
Lake_mask_r005_netcdf = xr.open_dataset(Lake_mask_r005_file, chunks={'time': dask_chunks})
Lake_mask_r005_netcdf = normalise_latlon(Lake_mask_r005_netcdf)
Lake_mask_r005 = Lake_mask_r005_netcdf['Mask']

# Create fine resolution mask for plotting (10x interpolation)
REPEAT = 10
orig_lat = Lake_mask_r005['lat'].values
orig_lon = Lake_mask_r005['lon'].values
new_lat = np.linspace(orig_lat.min(), orig_lat.max(), len(orig_lat) * REPEAT)
new_lon = np.linspace(orig_lon.min(), orig_lon.max(), len(orig_lon) * REPEAT)

lake_mask_fine_da = Lake_mask_r005.interp(lat=new_lat, lon=new_lon, method='nearest')
lake_mask_fine_da = lake_mask_fine_da.rename('lake_mask_fine')
lake_mask_fine_da.attrs = Lake_mask_r005.attrs


# ========== LOAD GRIDDED RAINFALL DATA ==========
agcd_daily_ds = xr.open_dataset(agcd_daily_file, chunks={'time': dask_chunks})
agcd_daily_ds = normalise_latlon(agcd_daily_ds)
agcd_daily_ds = ensure_time_coord(agcd_daily_ds)

# Mask to catchment and calculate catchment mean
agcd_daily_ds = assign_mask_with_check(agcd_daily_ds, Lake_mask_r005)
agcd_daily_ds['precip'] = agcd_daily_ds['precip'].where(agcd_daily_ds['precip'].catchment_mask == 1)
agcd_daily_ds = add_attrs('agcd', agcd_daily_ds, 'gridded_catchment_rainfall', ds_attrs)

agcd_daily_ds['mean_catchment_rainfall'] = agcd_daily_ds['gridded_catchment_rainfall'].mean(
    dim=('lon', 'lat'), skipna=True
)
agcd_daily_ds = add_attrs('agcd', agcd_daily_ds, 'mean_catchment_rainfall', ds_attrs)

# Subset to analysis period
lake_daily_ds = agcd_daily_ds.sel(time=slice(ds_start, ds_end))

agcd_timeframe = {
    'start': str(lake_daily_ds.time[0].dt.year.item()),
    'end': str(lake_daily_ds.time[-1].dt.year.item())
}


# ========== LOAD STATION DATA ==========
# Load runoff stations
runoff_stations_ds = xr.open_dataset(runoff_stations_file, chunks={'time': dask_chunks})
runoff_stations_ds = ensure_time_coord(runoff_stations_ds)
runoff_stations_ds_aligned = runoff_stations_ds.reindex(time=lake_daily_ds.time, method=None)
lake_daily_ds['station_runoff'] = add_attrs('station', runoff_stations_ds_aligned, 'runoff', ds_attrs)['runoff']
lake_daily_ds = filter_out_empty_stations(lake_daily_ds, 'runoff')

# Load rainfall stations
rainfall_stations_ds = xr.open_dataset(rainfall_stations_file, chunks={'time': dask_chunks})
rainfall_stations_ds = ensure_time_coord(rainfall_stations_ds)
rainfall_stations_ds_aligned = rainfall_stations_ds.reindex(time=lake_daily_ds.time, method=None)
lake_daily_ds['station_rainfall'] = add_attrs('station', rainfall_stations_ds_aligned, 'rainfall', ds_attrs)['rainfall']
lake_daily_ds = filter_out_empty_stations(lake_daily_ds, 'rainfall')


# ========== LOAD REGIONAL RAINFALL DATA ==========
NT_mask_r005_netcdf = xr.open_dataset(NT_mask_r005_file, chunks={'time': dask_chunks})
NT_mask_r005_netcdf = normalise_latlon(NT_mask_r005_netcdf)
NT_mask_r005 = NT_mask_r005_netcdf['landmask']

agcd_NT_daily_ds = xr.open_dataset(agcd_NT_daily_file, chunks={'time': dask_chunks})
agcd_NT_daily_ds = normalise_latlon(agcd_NT_daily_ds)
agcd_NT_daily_ds = ensure_time_coord(agcd_NT_daily_ds)

agcd_NT_daily_ds = assign_mask_with_check(agcd_NT_daily_ds, NT_mask_r005)
agcd_NT_daily_ds['precip'] = agcd_NT_daily_ds['precip'].where(agcd_NT_daily_ds['precip'].catchment_mask == 1)


# ========== LOAD SATELLITE LAKE OBSERVATIONS ==========
dea_ds = xr.open_dataset(dea_file, chunks={'time': dask_chunks})
dea_ds = ensure_time_coord(dea_ds)
dea_ds_aligned = dea_ds.reindex(time=lake_daily_ds.time, method=None)
lake_daily_ds['dea'] = add_attrs('dea', dea_ds_aligned, 'lake_observations', ds_attrs)['lake_observations']

dea_timeframe = {
    'start': str(dea_ds.time[0].dt.year.item()),
    'end': str(dea_ds.time[-1].dt.year.item())
}


# ========== LOAD CLIMATE DRIVERS ==========
# Load ENSO indices
ENSO_noaa = read_driver_files(enso_file, 'enso')
ENSO_noaa = ensure_time_coord(ENSO_noaa)
ENSO_noaa_monthly_da = ENSO_noaa.sel(time=slice(lake_daily_ds.time.min(), lake_daily_ds.time.max()))
ENSO_noaa_daily_da = resample_drivers_monthly_to_daily(ENSO_noaa_monthly_da)
lake_daily_ds['enso'] = ENSO_noaa_daily_da.reindex(time=lake_daily_ds.time, method=None)

# Load IPO indices
IPOtripole_noaa = read_driver_files(ipo_file, 'ipo')
IPOtripole_noaa = ensure_time_coord(IPOtripole_noaa)
IPOtripole_noaa_monthly_da = IPOtripole_noaa.sel(time=slice(lake_daily_ds.time.min(), lake_daily_ds.time.max()))
IPOtripole_noaa_daily_da = resample_drivers_monthly_to_daily(IPOtripole_noaa_monthly_da)
lake_daily_ds['ipo'] = IPOtripole_noaa_daily_da.reindex(time=lake_daily_ds.time, method=None)


#%% Analyse Data: Identify Filling Events from DEA

# ========== EVENT DETECTION METHODOLOGY ==========
# Filling events are defined as distinct peaks in lake size, separated by periods where 
# the lake is considered 'empty'. The 10% threshold eliminates small dry-season fluctuations 
# and ensures only substantial filling episodes are captured. Events must span more than a 
# single timestep to exclude isolated noise. The timestep before each event onset is included 
# to capture full filling dynamics.

# ========== EXTRACT AND THRESHOLD LAKE SIZE ==========
# Extract lake size time series
dea_lake_size_da = lake_daily_ds['dea'].sel(lake_variable='Size').dropna(dim='time')

# Set threshold at 10% of max lake size to define 'empty' conditions
threshold = dea_lake_size_da.max() / 10  # ~70th percentile of non-zero lake sizes

# Create boolean mask for event periods (lake not empty)
is_event = dea_lake_size_da > threshold


# ========== SEGMENT INTO EVENTS ==========
# Convert to numpy for efficient manipulation
event_mask = is_event.compute().values.astype(bool)

# Detect event start points (transitions from False → True)
group_change = np.diff(event_mask.astype(int), prepend=0)

# Assign preliminary group ID to each event
group_id = np.cumsum((group_change == 1) & event_mask).astype(float)
group_id[~event_mask] = np.nan  # Set non-event days to NaN


# ========== FILTER OUT SINGLE-TIMESTEP EVENTS ==========
# Remove events consisting of only one timestep (likely noise)
unique_vals, counts = np.unique(group_id[~np.isnan(group_id)], return_counts=True)
singleton_ids = unique_vals[counts == 1]
mask_singletons = np.isin(group_id, singleton_ids)
group_id[mask_singletons] = np.nan

# Reindex remaining events to be consecutive starting from 1
unique_ids = np.unique(group_id[~np.isnan(group_id)])
id_map = {old: new for new, old in enumerate(unique_ids, start=1)}
group_id = np.array([id_map[val] if val in id_map else np.nan for val in group_id])


# ========== INCLUDE PRE-EVENT TIMESTEP ==========
# Add the timestep before each event start to capture full rise dynamics
valid = ~np.isnan(group_id)  # Event timesteps
prev = np.isnan(np.concatenate(([np.nan], group_id[:-1])))  # Previous timestep was NaN
starts = np.where(valid & prev)[0]  # Indices of event start timesteps
idx = starts[starts > 0]  # Exclude first index (no previous timestep)
group_id[idx - 1] = group_id[idx]  # Assign event ID to previous timestep

# Validate all previous slots were filled
assert not any(np.isnan(group_id[idx - 1]))


# ========== CONVERT TO XARRAY AND ALIGN ==========
# Convert back to xarray DataArray
event_id_da = xr.DataArray(group_id, coords=dea_lake_size_da.coords, dims=dea_lake_size_da.dims)
event_id_aligned_da = event_id_da.reindex(time=lake_daily_ds['time'])

# Fill interior gaps (where forward and backward fill agree)
ff = event_id_aligned_da.ffill(dim='time')
bf = event_id_aligned_da.bfill(dim='time')
mask_enclosed = ff.notnull() & (ff == bf)
event_id_aligned_filled_da = xr.where(mask_enclosed, ff, event_id_aligned_da)


# ========== ADD TO DATASET ==========
# Validate alignment
assert event_id_aligned_filled_da.sizes['time'] == lake_daily_ds.sizes['time']
assert np.all(event_id_aligned_filled_da.time.values == lake_daily_ds.time.values)
assert np.all(np.isnan(event_id_aligned_filled_da) | 
              (event_id_aligned_filled_da == np.floor(event_id_aligned_filled_da)))

# Assign event IDs as coordinate
lake_daily_ds['dea'] = lake_daily_ds['dea'].assign_coords(
    dea_events=('time', event_id_aligned_filled_da.values)
)
lake_daily_ds['dea'].coords['dea_events'].attrs['long_name'] = 'DEA event number'


# ========== CALCULATE EVENT CHARACTERISTICS ==========
# Identify peak of each event
lake_daily_ds = lake_daily_ds.assign_coords(
    dea_event_max=find_dea_event_max_coord(lake_daily_ds)
)

# Calculate time offset from event peak
lake_daily_ds['dea'] = lake_daily_ds['dea'].assign_coords(
    dea_offset=calculate_event_offset_coord(lake_daily_ds, 'daily')
)

# Calculate cumulative rainfall per event
lake_daily_ds['cumulative_rainfall_per_event'] = calculate_cumulative_rainfall_per_event(
    lake_daily_ds, 'mean_catchment_rainfall'
)
lake_daily_ds = add_attrs('agcd', lake_daily_ds, 'cumulative_rainfall_per_event', ds_attrs)

# Filter cumulative rainfall to remove low-intensity periods
lake_daily_ds['cumulative_rainfall_per_event_filtered'] = (
    filter_cumulative_rainfall(lake_daily_ds)
    .reindex(time=lake_daily_ds.time, fill_value=np.nan)
)


#%% Analyse Data: Daily Rolling Cumulative Dataset (for Figures 4-8)

# ========== CALCULATE ROLLING SUM ==========
# Calculate rolling cumulative sum over specified window
rolling_sum = lake_daily_ds['mean_catchment_rainfall'].rolling(
    time=window_size_daily, center=False
).sum()


# ========== IDENTIFY PEAKS AND TROUGHS ==========
# Identify local maxima and minima for segmentation
# Troughs are used to segment the time series into distinct rainfall episodes
peaks_and_troughs = identify_peaks_and_troughs(rolling_sum, distance_val=180)
peak_ids = peaks_and_troughs['peaks']
trough_ids = peaks_and_troughs['troughs']

# Validate peak/trough counts
if not (len(peak_ids) == len(trough_ids) or len(peak_ids) == len(trough_ids) - 1):
    raise ValueError(
        f'Peak/trough mismatch: {len(peak_ids)} peaks, {len(trough_ids)} troughs. '
        'Peaks must equal or be one less than troughs.'
    )


# ========== CLASSIFY SEGMENTS BY THRESHOLD ==========
# Add boolean coordinates indicating whether each segment exceeds thresholds
rolling_sum_classified = classify_peak_segments(
    rolling_sum, peaks_and_troughs, 
    lower_threshold_daily, higher_threshold_daily
)

lake_daily_ds['rolling_sum_of_mean_catchment_rainfall'] = rolling_sum_classified
lake_daily_ds = add_attrs('agcd', lake_daily_ds, 'rolling_sum_of_mean_catchment_rainfall', ds_attrs)


# ========== CALCULATE CUMULATIVE RAINFALL TO PEAKS ==========
# For each peak, calculate cumulative rainfall over the preceding window
cumulative_rainfall_to_peak = calculate_cumulative_rainfall_for_window(
    lake_daily_ds, peak_ids, window_size_daily, time_unit='days'
)

# Align to full time axis
lake_daily_ds['cumulative_rainfall_up_to_peaks'] = (
    cumulative_rainfall_to_peak.reindex(time=lake_daily_ds.time, fill_value=np.nan)
)
lake_daily_ds = add_attrs('agcd', lake_daily_ds, 'cumulative_rainfall_up_to_peaks', ds_attrs)


#%% Generate Figures

# ========== FIGURE 1: CATCHMENT MAP ==========
plot_EA_map(
    'Fig1',
    lake_mask_fine=lake_mask_fine_da
)

plot_catchment_map(
    'Fig1',
    lake_daily_ds, 
    timeframe=dea_timeframe, 
    lake_mask_coarse=Lake_mask_r005, 
    lake_mask_fine=lake_mask_fine_da
)


# ========== FIGURE 2: DEA FLOOD EVENT TIMESERIES ==========
# Plot with monthly rainfall (mean and stations)
plot_dea_timeseries(
    'Fig2',
    lake_daily_ds, 
    timeframe=dea_timeframe, 
    time_unit_lake='daily', 
    hydro_type='rainfall', 
    time_unit_data='monthly', 
    plot_mean=True, 
    plot_station=True
)

# Plot with daily runoff (stations only)
plot_dea_timeseries(
    'Fig2',
    lake_daily_ds, 
    timeframe=dea_timeframe, 
    time_unit_lake='daily', 
    hydro_type='runoff', 
    time_unit_data='daily', 
    plot_station=True
)


# Plot rainfall indices (mm)
plot_dea_indices(
    'Fig2',
    lake_daily_ds, 
    timeframe=dea_timeframe, 
    time_unit_lake='daily', 
    index = 'mm'
)

# Plot rainfall indices (days)
plot_dea_indices(
    'Fig2',
    lake_daily_ds, 
    timeframe=dea_timeframe, 
    time_unit_lake='daily', 
    index = 'days'
)

# Plot rainfall index Monthly Total
plot_dea_indices(
    'Fig2',
    lake_daily_ds, 
    timeframe=dea_timeframe, 
    time_unit_lake='daily', 
    index = 'monthly_total'
)

# Plot rainfall index Rx5day
plot_dea_indices(
    'Fig2',
    lake_daily_ds, 
    timeframe=dea_timeframe, 
    time_unit_lake='daily', 
    index = 'Rx5day'
)

# Plot rainfall index SDII
plot_dea_indices(
    'Fig2',
    lake_daily_ds, 
    timeframe=dea_timeframe, 
    time_unit_lake='daily', 
    index = 'SDII'
)

# Plot rainfall indix R40mm
plot_dea_indices(
    'Fig2',
    lake_daily_ds, 
    timeframe=dea_timeframe, 
    time_unit_lake='daily', 
    index = 'R40mm'
)

# Plot rainfall indix R99p
plot_dea_indices(
    'Fig2',
    lake_daily_ds, 
    timeframe=dea_timeframe, 
    time_unit_lake='daily', 
    index = 'R99p'
)

# Plot rainfall indix CWD
plot_dea_indices(
    'Fig2',
    lake_daily_ds, 
    timeframe=dea_timeframe, 
    time_unit_lake='daily', 
    index = 'CWD'
)



# ========== FIGURE 3: CUMULATIVE RAINFALL PER EVENT ==========
# Plot all events with distinct colors
plot_cumulative_rainfall_events('Fig3', lake_daily_ds, timeframe=dea_timeframe)

# Plot events sorted by magnitude
plot_cumulative_rainfall_events('Fig3', lake_daily_ds, timeframe=dea_timeframe, magnitude=True)

# Plot filtered events sorted by magnitude
plot_cumulative_rainfall_events(
    'Fig3', 
    lake_daily_ds, 
    timeframe=dea_timeframe, 
    magnitude=True, 
    filtered=True
)


# ========== FIGURE 4: CUMULATIVE RAINFALL BUILD-UP ==========
# Plot cumulative rainfall over window
plot_cumulative_rainfall_windows(
    'Fig4', 
    ds=lake_daily_ds, 
    threshold=lower_threshold_daily, 
    time_interval=f'{window_size_daily}_days', 
    start=dea_timeframe['start']
)

# Plot sorted by magnitude
plot_cumulative_rainfall_windows(
    'Fig4', 
    ds=lake_daily_ds, 
    threshold=lower_threshold_daily, 
    time_interval=f'{window_size_daily}_days', 
    start=dea_timeframe['start'], 
    magnitude=True  # Fixed: was string 'True', should be boolean True
)


# ========== FIGURE 5: THRESHOLD ANALYSIS ==========
# Analyze optimal rolling sum window and threshold
plot_threshold_analysis(
    'Fig5', 
    lake_daily_ds, 
    timeframe=dea_timeframe, 
    window_max=208, 
    threshold=lower_threshold_daily, 
    time_unit='days', 
    optimal_window=window_size_daily
)

table_df = create_threshold_sensitivity_table(
    ds=lake_daily_ds,
    timeframe=dea_timeframe,
    window_range=(1, 208),
    threshold=lower_threshold_daily,
    time_unit='days'
)
table_df.to_csv(f'{output_dir}/Table_ThresholdSensitivity.csv', index=False)


# ========== FIGURE 6: ROLLING SUM VS LAKE MAX SCATTER & TABLE ==========
#Events Table
events_table = (
    pd.DataFrame({
        'event_no': np.unique(lake_daily_ds.dea_events.dropna(dim='time').values).astype(int),
        'start_date': lake_daily_ds.time.groupby('dea_events').min().values,
        'end_date': lake_daily_ds.where(lake_daily_ds.dea_event_max, drop=True).time.values,
        'duration in days (filtered)': abs(lake_daily_ds.dea_offset_filtered.groupby('dea_events').min()).values,
        'cum precip (filtered)': lake_daily_ds.cumulative_rainfall_per_event_filtered.groupby('dea_events').max().values.round(2),
        f'{window_size_daily} day total': lake_daily_ds.rolling_sum_of_mean_catchment_rainfall.groupby('dea_events').max().values.round(2),
        'lake peaks': lake_daily_ds.dea.sel(lake_variable='Size').groupby('dea_events').max().values.round(2),
    })
    .set_index('event_no')
)
events_table.to_csv(f'{output_dir}/events_table.csv')

# Basic usage with outliers
results = plot_scatter_RollingSumVsLakeSize(
    'Fig6', 
    df=events_table,
    ds=lake_daily_ds,
    window=window_size_daily,
    outlier_events=[10, 14, 16, 1, 3, 12]
)

# Without outliers
#results = plot_scatter_RollingSumVsLakeSize(
#    'Fig6', 
#    df=events_table,
#    window=window_size_daily,
#)

# Without labels for cleaner look
#results = plot_scatter_RollingSumVsLakeSize(
#    'Fig6', 
#    df=events_table,
#    window=window_size_daily,
#    outlier_events=[10, 14, 16, 1, 3, 12],
#    show_labels=False
#)

# Access results
if results:
    print(f"Slope: {results['slope']:.3f}")
    print(f"R-squared: {results['r_squared']:.3f}")
    print(f"P-value: {results['p_value']:.3e}")
    
    
# ========== FIGURE 7: ROLLING SUM RAINFALL TIMESERIES ==========
# Plot fixed window rolling sum
plot_rolling_sum_subplots(
    'Fig7', 
    lake_daily_ds, 
    window=window_size_daily, 
    time_unit='days',
    threshold1=lower_threshold_daily, 
    threshold2=higher_threshold_daily,
    event_timeframe=dea_timeframe, 
    rain_timeframe=agcd_timeframe
)

# Plot variable window rolling sum (80-224 days)
plot_rolling_sum_variable_window_subplots(
    'Fig7', 
    lake_daily_ds, 
    time_unit='days', 
    threshold=lower_threshold_daily, 
    window_min=80, 
    window_max=224,
    event_timeframe=dea_timeframe, 
    rain_timeframe=agcd_timeframe
)


# ========== FIGURE 8: NORTHERN AUSTRALIA COMPOSITE MAPS ==========
# Calculate window sums for regional rainfall data
NT_daily_rainfall_window_sum_da = calculate_sum_over_given_windows( 
    ds=lake_daily_ds, 
    da=agcd_NT_daily_ds['precip'], 
    window_size=window_size_daily
)

# Generate composite maps for event/non-event periods
plot_NT_event_maps_subplots(
    'Fig8', 
    NT_daily_rainfall_window_sum_da, 
    event_timeframe=dea_timeframe, 
    lake_mask=lake_mask_fine_da,
    window_size=window_size_daily, 
    time_unit='days', 
    threshold=lower_threshold_daily
)


# ========== FIGURE 9 & 10: CLIMATE DRIVERS WITH ROLLING SUM ==========
# Define available driver indices
driver_indices_dict = {
    'enso': {
        'Nino34_ERSST': 'Nino 3.4 (ERSST)',
        'Nino34_HadISST': 'Nino 3.4 (HadISST)',
        'ONI': 'ONI'
    },
    'ipo': {
        'HadISST': 'HadISST',
        'ERSSTv5': 'ERSSTv5',
        'COBE': 'COBE'
    }
}

# Common plotting parameters
plot_params = {
    'indices_dict': driver_indices_dict,
    'window': window_size_daily,
    'time_unit': 'days',
    'threshold1': lower_threshold_daily,
    'threshold2': higher_threshold_daily,
    'event_timeframe': dea_timeframe,
    'rain_timeframe': agcd_timeframe
}


# ========== ENSO INDICES ==========
# Nino 3.4 (ERSST)
plot_rolling_sum_with_driver_subplots(
    'Fig9', 
    lake_daily_ds, 
    driver='enso', 
    driver_index='Nino34_ERSST',
    **plot_params
)

# Nino 3.4 (HadISST)
plot_rolling_sum_with_driver_subplots(
    'Fig9', 
    lake_daily_ds, 
    driver='enso', 
    driver_index='Nino34_HadISST',
    **plot_params
)

# ONI
plot_rolling_sum_with_driver_subplots(
    'Fig9', 
    lake_daily_ds, 
    driver='enso', 
    driver_index='ONI',
    **plot_params
)


# ========== IPO INDICES ==========
# HadISST
plot_rolling_sum_with_driver_subplots(
    'Fig10', 
    lake_daily_ds, 
    driver='ipo', 
    driver_index='HadISST',
    **plot_params
)

# ERSST
plot_rolling_sum_with_driver_subplots(
    'Fig10', 
    lake_daily_ds, 
    driver='ipo', 
    driver_index='ERSSTv5',
    **plot_params
)

# COBE
plot_rolling_sum_with_driver_subplots(
    'Fig10', 
    lake_daily_ds, 
    driver='ipo', 
    driver_index='COBE',
    **plot_params
)









      