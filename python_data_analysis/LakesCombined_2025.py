#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 10:49:38 2025

@author: leasophiegrunau
"""

import glob
import pandas as pd
import xarray as xr
import numpy as np
from scipy.signal import find_peaks
import scipy.stats as stats
from collections import Counter
#import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize  # already available via your imports
from matplotlib import colormaps
import matplotlib.lines as mlines
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
import warnings
import cftime
CF = cftime.DatetimeNoLeap
import regionmask

"""


TO DO

- fix rainfall_station_cum_filtered to start depending on rate of change of diff stations
- Check plot 12 after and make sure it works
- 128d mean rainfall for each event on one plot (two colours: above/ below 600mm)
- NT composite map
- Look at  -- Extra Plots -- Karteikarte

"""

Lake = 'LW'
Lake_longname = 'LakeWoods'

Lakes = ['LE', 'LW', 'LB', 'LG']
Lakes_longname = ['LakeEyre', 'LakeWoods', 'LakeBuchanan', 'LakeGeorge']

input_dir = r'/Users/leasophiegrunau/Desktop/PhD_Australia/Programming/Python/Data.nosync'
output_dir = r'/Users/leasophiegrunau/Desktop/PhD_Australia/Programming/Python/Output'

presentation_bom_colours = ['#000033','#336666', '#99cc33', '#339966','#8EB28E','#336600']
blue_colors = ['#0000FF', '#00008B', '#4169E1', '#6495ED', '#87CEFA', '#4682B4', '#5F9EA0', '#7B68EE', '#87CEEB', '#ADD8E6']
blue_colors_map = ['#FFFFFF', '#F0F8FF', '#B0C4DE', '#87CEEB', '#6A5ACD', '#483D8B', '#4169E1', '#0000CD', '#000080', '#000033']
dea_colour_list = ['tomato', 'darkred', 'orange', 'gold', 'olive', 'forestgreen', 'teal', 'aqua', 'steelblue', 'navy', 'purple', 'fuchsia', 'pink', 'maroon', 'lightcoral', 'red', 'sienna', 'tan', '#000033','#336666', '#99cc33', '#339966','#8EB28E','#336600', '#87CEEB', '#ADD8E6']
event_colour_list = [ 'maroon', 'orange', 'teal', 'steelblue', 'fuchsia', 'tomato']


Lake_codes = {'LE': 'r4ctum36x_v3', 
              'LW': 'quyftexks_v3',
              'LB': 'rhptpmn9u_v3',
              'LG': 'r3f225n9t_v3'}

Lake_names = {'LE': 'Lake Eyre', 
              'LW': 'Lake Woods',
              'LB': 'Lake Buchanan',
              'LG': 'Lake George'}

Lake_code = Lake_codes[Lake]
Lake_name = Lake_names[Lake]    


#-----------------------------------------------------------------------------------------------------------------------------------

#### ========= Functions: Prepare data =========

def filter_out_empty_stations(ds, station_type):
    """
    Filter out all variables with a given station dimension that contain only NaNs across time.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing station-based variables.
    station_type : str
        Either 'rainfall' or 'runoff'.

    Returns
    -------
    xarray.Dataset
        Dataset with empty stations (all-NaN over time) removed from relevant variables.
    """
    dim_name = f'{station_type}_station'
    
    # Step 1: Identify one key variable to test which stations have data
    reference_var = f'station_{station_type}'
    stations_with_data = ~ds[reference_var].isnull().all(dim='time')
    
    # Step 2: Build new dataset with filtered station axis in all matching variables
    variables_to_filter = [var for var in ds.data_vars if dim_name in ds[var].dims]
    coords_to_filter = [coord for coord in ds.coords if dim_name in ds[coord].dims or coord == dim_name]
    
    # Filter all relevant variables and coordinates
    filtered_vars = {
        var: ds[var].isel({dim_name: stations_with_data})
        for var in variables_to_filter + coords_to_filter
    }
    
    # Drop original variables and reassign filtered ones
    ds_filtered = ds.drop_vars(variables_to_filter + coords_to_filter)
    ds_filtered = ds_filtered.assign(filtered_vars)
    
    return ds_filtered


def read_ipo_file():
    """
    Reads IPO Tripole Index time series data files and returns a combined xarray DataArray.

    The function searches for files matching the pattern 
    'ClimaticDrivers_indices/tpi.timeseries*.txt' in `input_dir`. 
    Each file is read into a pandas DataFrame, missing values coded as -99.0 are replaced 
    with NaN, and the data is reshaped from wide to long format (months as values). 
    All files are then concatenated along the columns, preserving the time index.

    Returns
    -------
    xr.DataArray
        A DataArray of shape (time, ipo_indices) with:
        - `time` coordinate: datetime index corresponding to year and month.
        - `ipo_indices` coordinate: string labels from the metadata for each index.
        - Attributes:
            - `units`: extracted from the metadata (e.g., 'degC').
            - `reference`: text reference from the metadata.
            - `url`: source URL from the metadata.
            - `date`: creation or update date from the metadata.
        - Name: 'IPO tripole index'
    """
    file_paths = glob.glob(f'{input_dir}/ClimaticDrivers_indices/tpi.timeseries*.txt')
    
    df_list = []
    df_metadata_list = []
    for path in file_paths:
        n_lines = sum(1 for _ in open(path))
        df_metadata = pd.read_csv(path, sep='   ', header=None, engine='python', skiprows=n_lines-11)
        df_metadata_list.append(df_metadata)

        df = pd.read_csv(path,
            sep='   ', header=None, engine='python', skiprows=1, skipfooter=11)
        df = df.replace(-99.000, float('NaN'))
        df = pd.melt(df, id_vars=[0], value_vars=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], value_name=df_metadata.iloc[2,0])
        df.index = pd.to_datetime((df[0].astype(str) + df['variable'].astype(str)), format='%Y%m')
        df = df.drop(columns=[0, 'variable'])
        df = df.sort_index()
        df_list.append(df)
        
    combined_df = pd.concat(df_list, axis=1)
    
    da = xr.DataArray(
        data = combined_df.values,
        dims = ("time", "ipo_indices"),
        coords = {
            "time": combined_df.index,
            "ipo_indices": combined_df.columns.astype(str),
            },
        name = 'IPO tripole index',
        attrs = {
            "units": df_metadata_list[0].iloc[10,0].split('"')[1],
            "reference": df_metadata_list[0].iloc[1,0],
            "url": df_metadata_list[0].iloc[7,0],
            "date": df_metadata_list[0].iloc[9,0],
            "description": (
                "Combined IPO dataset containing multiple sea surface temperature anomaly indices along an 'ipo_indices' dimension. "
                "Includes the Tripole Index (TPI) derived from ERSSTv5, HadISST, and COBE datasets."
            )
        }
    )
    return da


def find_rainfall_peaks_near_lake_peaks(ds, percentile=0.8):
    """
    Find the most significant monthly rainfall peaks closest to each lake peak.

    Parameters:
        ds (xr.Dataset): Dataset containing 'dea', 'dea_events', and 'mean_gridded_rainfall'.
        percentile (float): Threshold to select top rainfall peaks (e.g., 0.8 keeps top 20%).

    Returns:
        xr.DataArray: rainfall peaks closest to each lake peak, filtered by magnitude.
    """
    # Extract relevant variables
    lake_size = ds['dea'].sel(lake_variable='Size')
    event_id = ds['dea_events']
    rainfall_monthly = ds['mean_gridded_rainfall'].resample(time='MS').sum()
    events = np.unique(event_id.dropna(dim='time').values)

    # Step 1: Select dates of peak lake size for each event
    dea_peak_dates = np.array([
        lake_size.where(event_id == event, drop=True).idxmax(dim='time').values
        for event in events
    ])

    # Step 2: Find local peaks in rainfall
    rainfall_diff = rainfall_monthly.diff(dim='time')
    peak_mask = (rainfall_diff > 0) & (rainfall_diff.shift(time=-1) < 0)
    rainfall_peaks = rainfall_monthly.where(peak_mask, drop=True)

    # Step 3: Keep only major/ significant peaks
    threshold = rainfall_peaks.quantile(percentile)
    significant_peaks = rainfall_peaks.where(rainfall_peaks > threshold, drop=True)

    # Step 4: Match each lake peak to the closest rainfall peak
    sig_peak_dates = significant_peaks.time.values
    closest_dates = [
        sig_peak_dates[np.argmin(np.abs(sig_peak_dates - date))]
        for date in dea_peak_dates
    ]

    return rainfall_monthly.sel(time=np.array(closest_dates))


def find_lake_event_max_coord(ds):
    """
    Create a boolean coordinate indicating the maximum point of each detected flood event.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with variables:
        - 'dea': lake size with lake_variable='Size'
        - 'dea_events': event IDs

    Returns
    -------
    xarray.DataArray
        Boolean coordinate: True for the max of each event, False otherwise.
        Same time dimension as `dea`.
    """
    lake_size = ds['dea'].sel(lake_variable='Size')
    events = np.unique(ds.dea_events.dropna(dim='time').values)

    # Initialize boolean array with False
    event_max = xr.DataArray(np.zeros(lake_size.time.size, dtype=bool),
                             coords={"time": lake_size.time},
                             dims=["time"])

    # Set True for the maximum point of each event
    for event in events:
        peak_time = lake_size.where(lake_size.dea_events == event, drop=True).idxmax(dim='time')
        event_max.loc[dict(time=peak_time)] = True

    return event_max


def resample_lake_dataset_to_monthly(ds_daily):
    """
    Resample a daily lake dataset to monthly resolution.

    - Sums variables related to rainfall and runoff.
    - Applies max to DEA-related variables (e.g. lake size events).
    - Carries over static variables without a time dimension.

    Parameters:
        ds_daily (xr.Dataset): Input dataset with daily time steps.

    Returns:
        xr.Dataset: Monthly resampled dataset.
    """
    ds_monthly = xr.Dataset()
    
    # Variables with a time dimension
    vars_with_time = [var for var in ds_daily.data_vars 
                      if 'time' in ds_daily[var].dims]
    
    # Variables to sum monthly (any containing 'rainfall', 'runoff')
    vars_to_sum = [var for var in vars_with_time 
                   if any(key in var for key in ['rainfall', 'runoff'])]
    for var in vars_to_sum:
        ds_monthly[var] = ds_daily[var].resample(time='1MS').sum()
    
    # Variables to take the monthly maximum (any containing 'dea')
    vars_to_max = [var for var in vars_with_time if 'dea' in var]
    for var in vars_to_max:
        ds_monthly[var] = ds_daily[var].resample(time='1MS').max()
        ds_events_monthly = ds_daily[var].coords['dea_events'].resample(time='1MS').max()
        ds_event_max_monthly = ds_daily[var].coords['dea_event_max'].resample(time='1MS').max()
        ds_monthly[var] = ds_monthly[var].assign_coords(dea_events=ds_events_monthly, dea_event_max=ds_event_max_monthly)
    
    # Static (non-time) variables ('lat_bnds', 'lon_bnds', 'mask', 'crs')
    static_vars = [var for var in ds_daily.data_vars
                   if 'time' not in ds_daily[var].dims]
    for var in static_vars:
        ds_monthly[var] = ds_daily[var]
    
    # Update long_name and description to reflect monthly resampling
    for var in ds_monthly.data_vars:
        if "long_name" in ds_monthly[var].attrs:
            ds_monthly[var].attrs["long_name"] = ds_monthly[var].attrs["long_name"].replace("Daily", "Monthly")
        if "description" in ds_monthly[var].attrs:
            ds_monthly[var].attrs["description"] = ds_monthly[var].attrs["description"].replace("daily", "monthly")
            ds_monthly[var].attrs["description"] = ds_monthly[var].attrs["description"].replace("Daily", "Monthly")
    
    return ds_monthly


def calculate_event_offset_coord(ds, time_unit):
    """
    Calculate event offset relative to the peak of each detected flood event,
    returning it as a coordinate-ready DataArray with the same time dimension as `ds`.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'dea' (lake size with lake_variable='Size') and 'dea_events'.
    time_unit : str
        'daily' or 'monthly', used for offset calculation.

    Returns
    -------
    xarray.DataArray
        Offset of each time point relative to the peak of its event.
        Same time dimension as `ds`.
        NaN for times not part of any event.
    """
    dea = ds['dea'].sel(lake_variable='Size')
    events = np.unique(ds.dea_events.dropna(dim='time').values)

    offsets_list = []

    for event in events:
        event_dates = dea.where(dea.dea_events == event, drop=True).time
        peak_date = dea.where(dea.dea_events == event, drop=True).idxmax(dim='time')
        full_event_dates = dea.sel(time=slice(event_dates.min(), event_dates.max())).time

        if time_unit == 'daily':
            offset = (full_event_dates - peak_date.values) / np.timedelta64(1, 'D')
        else:  # monthly
            offset = (full_event_dates.dt.year - peak_date.dt.year.item()) * 12 + \
                     (full_event_dates.dt.month - peak_date.dt.month.item())

        offsets_list.append(xr.DataArray(offset, coords={"time": full_event_dates}, dims=["time"]))

    # Concatenate all events and reindex to full ds.time
    offset_da = xr.concat(offsets_list, dim="time").sortby("time")
    offset_da = offset_da.reindex(time=ds.time)  # fill NaNs where time not part of any event
    offset_da.name = "dea_offset"

    return offset_da


def cumulative_rainfall_per_event(ds, variable):
    
    # Extract variables from dataset
    events = np.unique(ds.dea_events.dropna(dim='time').values)         # Array: unique event numbers
    rainfall = ds[variable]
    event_offset = ds.dea_offset
    event_offset_rise = event_offset.where(event_offset <= 0, drop=True)

    #Calculate cumulative rainfall per event
    rainfall_cumsums = []
    for event in events:
        event_offset_dates = event_offset_rise.where(event_offset_rise.dea_events == event, drop=True).time
        event_offset_dates_min = event_offset_dates.min()
        event_offset_dates_max = event_offset_dates.max()
        rainfall_event = rainfall.sel(time=slice(event_offset_dates_min, event_offset_dates_max))
        rainfall_event_cum = rainfall_event.cumsum(dim='time')
        rainfall_cumsums.append(rainfall_event_cum)
    return xr.concat(rainfall_cumsums, dim="time").sortby("time")


def filter_cumulative_rainfall(ds):

    ### ========= Extract variables from dataset =========
    precip_cum = ds['mean_gridded_rainfall_event_cum']
    precip = ds['mean_gridded_rainfall']
    # ====================================================
    
    events = np.unique(ds.dea_events.dropna(dim='time').values)
    event_offset_rise = ds.dea_offset.where(ds.dea_offset <= 0, drop=True)
    
    ###  === Filter cumulative precip by rate of change === 
    threshold_value = precip.max()/100 # 1% of the max daily rainfall
    
    precip_cum_filtered = []
    for event in events:
        offset_event_rise = event_offset_rise.where(event_offset_rise.dea_events == event, drop=True)
        min_date = offset_event_rise.time.min()
        max_date = offset_event_rise.time.max()
    
        # Select the diff and cum for this event
        event_cum = precip_cum.sel(time=slice(min_date, max_date))
        event_cum_diff = precip.sel(time=slice(min_date, max_date))
        
        mask = event_cum_diff > threshold_value
    
        ##  === Filter with first mask === 
        # Find the first and last time where diff exceeds threshold
        true_indices = mask.where(mask, drop=True)
        
        if not true_indices.any(): # If the threshold is never exceeded, skip this event
            continue
        
        #Filter cumulative precip
        event_cum_filtered = event_cum.sel(time=slice(true_indices.isel(time=0).time, true_indices.isel(time=-1).time))
        event_cum_filtered = event_cum_filtered.assign_coords(dea_offset_filtered=("time", (event_cum_filtered.dea_offset - event_cum_filtered.dea_offset.max()).values))
        precip_cum_filtered.append(event_cum_filtered)
    
    
    # Concatenate across time
    return xr.concat(precip_cum_filtered, dim="time").sortby("time")

def identify_peaks_and_troughs(series_values, distance_val):
    
    """
    Identify and clean local peaks and troughs in a 1D time series.

    This function finds local maxima (peaks) and minima (troughs) in the input time series
    using prominence and distance filters, ensures the series starts with a trough,
    and removes any consecutive peaks or troughs that are not separated by the opposite type.
    For double troughs or peaks, the deeper/larger one is kept.

    Parameters
    ----------
    series_values : xr.DataArray
        A 1D xarray DataArray of the time series values to analyse. Must have a 'time' dimension.
    distance_val : int
        Minimum distance (in time steps) between consecutive peaks/troughs.

    Returns
    -------
    peaks_and_troughs : dict
        Dictionary with two NumPy arrays:
            - 'peaks': indices of cleaned peaks
            - 'troughs': indices of cleaned troughs

    Notes
    -----
    - Prominence threshold is set as the 25th percentile of the series values (excluding NaNs).
    - If no peaks or troughs are found, returns empty arrays for both.
    - Output indices refer to positions in the original `series_values` array.
    """    
    
    # Step 1: Define prominence threshold & Identify peaks and troughs
    prominence_val = np.percentile(series_values.dropna(dim='time'), 25)
    
    peaks, _ = find_peaks(series_values.values, prominence=prominence_val, distance=distance_val)
    troughs, _ = find_peaks(-series_values.values, prominence=prominence_val, distance=distance_val)
    
    # Return empty if nothing found
    if len(peaks) == 0 or len(troughs) == 0:
        return {'troughs': np.array([]), 'peaks': np.array([])}

    # Step 2: Ensure that series starts with a trough
    if peaks[0] < troughs[0]:
        peaks = peaks[1:]
    
    # Step 3: Filter out double troughs
    cleaned_troughs = [troughs[0]]
    for i in range(len(troughs)-1):
        t0 = troughs[i]
        t1 = troughs[i+1]
        
        # Is there any peak between these two troughs?
        in_between_peaks = [p for p in peaks if t0 < p < t1]
    
        # Peak → add new value (t1)
        if in_between_peaks:
            cleaned_troughs.append(t1)
            
        # No peak → keep only the lower of two troughs
        else:
            # First (already existing) is lower → don't add new value (t1)
            if series_values[t0] < series_values[t1]:
                continue
            # First (already existing) is higher → remove existing value (t0) and add new value (t1)
            else:
                del cleaned_troughs[-1]
                cleaned_troughs.append(t1)
            
    # Step 4: Filter out double peaks
    cleaned_peaks = [peaks[0]]
    for i in range(len(peaks)-1):
        p0 = peaks[i]
        p1 = peaks[i+1]
        
        # Is there any trough between these two peaks?
        in_between_troughs = [t for t in troughs if p0 < t < p1]

        # Trough → add new value (p1)    
        if in_between_troughs:
            cleaned_peaks.append(p1)
            
        # No trough → keep only the higher of two peaks
        else:
            # First (already existing) is higher → don't add new value (p1)
            if series_values[p0] > series_values[p1]:
                continue
            else:
            # First (already existing) is lower → remove existing value (p0) and add new value (p1)
                del cleaned_peaks[-1]
                cleaned_peaks.append(p1)
                
    # Step 5: Turn into dict with arrays
    return {
        'troughs': np.array(cleaned_troughs),
        'peaks': np.array(cleaned_peaks)
        }

def extract_peak_segments(ds, distance, time_unit, window, threshold1, threshold2):
    """
    Extract rainfall segments between troughs from the rolling cumulative sum 
    and compute cumulative rainfall leading up to each peak. 
    
    Boolean coordinates are added to `rolling_sum` to indicate which time 
    points belong to a segment where the peak exceeded specified thresholds.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'mean_gridded_rainfall'.
    distance : int
        Minimum distance (in time steps) between consecutive peaks/troughs.
    time_unit : {'months', 'days'}
        Time unit for the cumulative window.
    window : int
        Number of months/days to include before and including each peak.
    threshold1 : float
        First threshold for peak magnitude.
    threshold2 : float
        Second threshold for peak magnitude.
    
    Returns
    -------
    dict
        {
            'rolling_sum' : xr.DataArray
                Rolling cumulative rainfall with boolean coordinates 
                (e.g., 'peak_above_100') marking segments where the 
                peak exceeds thresholds.
            'rainfall_cum_{window}_{time_unit}' : xr.DataArray
                Cumulative rainfall within the specified window preceding each peak.
        }
    
    Notes
    -----
    - Depends on external function: `identify_peaks_and_troughs`.
    - Boolean coordinates are named dynamically using the threshold values.
    """
    # Step 1: Calculate rolling cumulative sum of mean rainfall
    rolling_sum = ds['mean_gridded_rainfall'].rolling(time=window, center=False).sum()
    
    
    # Step 2: Identify local maxima (peaks) and minima (troughs) in the rolling sum
    # Troughs are used to segment the time series into rainfall episodes
    peaks_and_troughs = identify_peaks_and_troughs(rolling_sum, distance)

    peak_ids = peaks_and_troughs['peaks']
    trough_ids = peaks_and_troughs['troughs']

    # Sanity check
    if not (len(peak_ids) == len(trough_ids) or len(peak_ids) == len(trough_ids) - 1):
        raise ValueError("Number of peaks must be equal to or one less than number of troughs.")

    peak_dates = rolling_sum.time[peak_ids]
    rolling_sum_peaks = rolling_sum.sel(time=peak_dates)


    # Step 3: Loop through each segment defined by two troughs
    # Step 3a: For each segment, add boolean coord depending on threshold1 and threshold2
    # Step 3b: For each segment, find peak magnitude and examine the 128 days leading up to it

    # Initialize boolean coordinates (3a)
    peak_above_thresh1 = xr.DataArray(np.zeros(rolling_sum.time.size, dtype=bool),
                                      coords={"time": rolling_sum.time},
                                      dims=["time"])
    peak_above_thresh2 = xr.DataArray(np.zeros(rolling_sum.time.size, dtype=bool),
                                      coords={"time": rolling_sum.time},
                                      dims=["time"])

    # Empty list for cumulative rainfall leading up to peak (3b)
    cum_list = []

    for i in range(len(trough_ids) - 1):
        # Time slice between consecutive troughs (3 a+b)
        start_idx = trough_ids[i]
        end_idx = trough_ids[i+1]
        time_slice = rolling_sum.time[start_idx:end_idx]

        # Peak info (3 a+b)
        peak = rolling_sum_peaks[i].values
        peak_date = peak_dates[i]

        # Slice original rainfall data to the window leading up to the peak (3a)
        # --- Use dynamic time offset based on unit ---
        if time_unit == 'months':
            start_date = pd.Timestamp(peak_date.values) - pd.DateOffset(months=window - 1)
        elif time_unit == 'days':
            start_date = pd.Timestamp(peak_date.values) - pd.Timedelta(days=window - 1)
        else:
            raise ValueError("Unsupported unit. Use 'months' or 'days'.")

        # Extract window and compute cumulative sum (3a)
        mean_rainfall_window = ds['mean_gridded_rainfall'].sel(time=slice(start_date, peak_date))
        mean_rainfall_window_cum = mean_rainfall_window.cumsum(dim='time')

        # Assign dynamic coordinate name (3a)
        coord_name = f'cum_window_{window}_months' if time_unit == 'months' else f'cum_window_{window}_days'
        cum_coord = xr.DataArray(np.arange(1, len(mean_rainfall_window_cum) + 1),
                                 dims=["time"], coords={"time": mean_rainfall_window_cum.time})
        mean_rainfall_window_cum = mean_rainfall_window_cum.assign_coords({coord_name: cum_coord})
        cum_list.append(mean_rainfall_window_cum)

        # Mark boolean coordinates for thresholds (3b)
        if peak >= threshold1:
            peak_above_thresh1.loc[dict(time=time_slice)] = True
        if peak >= threshold2:
            peak_above_thresh2.loc[dict(time=time_slice)] = True

    # Concatenate cumulative rainfall (3a)
    cum_da = xr.concat(cum_list, dim="time").sortby("time")

# Assign boolean coordinates to rolling_sum dynamically based on threshold values (3b)
    rolling_sum = rolling_sum.assign_coords({
    f'rolling_peak_above_{threshold1}': peak_above_thresh1,
    f'rolling_peak_above_{threshold2}': peak_above_thresh2
})

    return {
        'rolling_sum': rolling_sum,
        f'rainfall_cum_{window}_{time_unit}': cum_da
    }


def classify_rainfall_windows_by_event_and_threshold(ds, cum_rainfall_over_window_da, window, time_unit, threshold):
    """
    Classify rainfall windows by event presence and threshold exceedance.
    
    Adds two boolean coordinates:
    - 'is_event'   : True if the window overlaps with an event.
    - 'above_<threshold>' : True if cumulative rainfall in the window >= threshold.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'dea_events'.
    cum_rainfall_over_window_da : xr.DataArray
        Concatenated rainfall segments (e.g., from peak-based extraction).
    event_no : xr.DataArray
        DataArray with labeled event numbers (NaN where no event).
    window : int
        Number of time steps (e.g., months or days) per window segment.
    threshold : float
        Rainfall threshold used to classify each window.

    Returns
    -------
    xr.DataArray
        Same rainfall windows, with two classification coordinates added.
    """
    # Initialize coordinates
    cum_window_is_event = xr.DataArray(np.full(cum_rainfall_over_window_da.size, np.nan),
                                   coords={"time": cum_rainfall_over_window_da.time},
                                   dims=["time"])
    cum_window_above_threshold = xr.DataArray(np.zeros(cum_rainfall_over_window_da.size, dtype=bool),
                                         coords={"time": cum_rainfall_over_window_da.time},
                                         dims=["time"])
    
    event_no_rise = ds.dea_events.where(ds.dea_events.dea_offset<=0)
    
    # Loop through each window
    for i in range(int(cum_rainfall_over_window_da.size / window)):
        # Slice out the current window
        cum_rainfall_window = cum_rainfall_over_window_da[window * i : window * (i + 1)]
        event_in_window = event_no_rise.sel(time=cum_rainfall_window.time)
        
        # Event classification
        if event_in_window.notnull().any().item():
            is_event  = np.unique(event_in_window.dropna(dim='time').values)
            if is_event.size > 1:
                raise ValueError("More than one event in a window.")
            else:
                cum_window_is_event.loc[dict(time=cum_rainfall_window.time)] = is_event
        
        # Threshold classification
        above_thresh = (cum_rainfall_window.max(dim="time") >= threshold).item()
        cum_window_above_threshold.loc[dict(time=cum_rainfall_window.time)] = above_thresh

    #Compare events captured to actual events
    window_events = np.unique(cum_window_is_event.dropna(dim='time').values)
    events = np.unique(ds.dea_events.dropna(dim='time').values)
    missing_events = np.setdiff1d(events, window_events)
    
    # Loop through each window again to check for missing events
    if missing_events.size:
        for i in range(int(cum_rainfall_over_window_da.size / window)):
            # Slice out the current window
            cum_rainfall_window = cum_rainfall_over_window_da[window * i : window * (i + 1)]
            start_date = cum_rainfall_window.time.min()
            end_date = pd.Timestamp(cum_rainfall_window.time.max().values)+pd.DateOffset(**{time_unit: window})
            event_in_window = event_no_rise.sel(time=slice(start_date, end_date))
            
            # Event classification
            if event_in_window.notnull().any().item():
                is_event  = np.unique(event_in_window.dropna(dim='time').values)
                if is_event.size > 1:
                    raise ValueError("More than one event in a window.")
                else:
                    if np.any(np.isin(is_event, missing_events)):
                        cum_window_is_event.loc[dict(time=cum_rainfall_window.time)] = is_event

    #Compare events captured to actual events
    new_window_events = np.unique(cum_window_is_event.dropna(dim='time').values)
    remaining_missing_events = np.setdiff1d(events, new_window_events)
    if remaining_missing_events.size:
        raise ValueError("There are still events missing after window extension.")
    
    # Assign coordinates to the original DataArray
    classified_da = cum_rainfall_over_window_da.assign_coords({
        'cum_window_is_event': cum_window_is_event,
        f'cum_window_above_{threshold}': cum_window_above_threshold
    })

    return classified_da
   

#### ========= Functions: Plot data =========

def plot_dea_events(ds, axes_no, zorder_plot, zorder_text):
    """
    Plot lake events from DEA with markers and labels at the maximum lake size for each event.

    Parameters:
        ds (xarray.Dataset): Dataset containing 'dea' with lake_variable dimension and 'dea_events'.
        axes_no (matplotlib.axes.Axes): The axis to plot on.
        zorder_plot (int): Drawing order of the line plot.
        zorder_text (int): Drawing order of the text labels.
    
    Notes
    -----
    - Assumes that `presentation_bom_colours[0]` is defined globally.
    """
    lake_size = ds['dea'].sel(lake_variable='Size')
    events = np.unique(ds['dea_events'].dropna(dim='time').values)
    
    for event in events:
        event_sizes = lake_size.where(lake_size.dea_events == event, drop=True).dropna(dim='time')
        axes_no.plot(event_sizes.time, event_sizes, linestyle='--', marker='o', markersize=6,
                     color=presentation_bom_colours[0], label=f'Event {int(event)}', zorder=zorder_plot)
    
        label_size = event_sizes.where(event_sizes.dea_event_max, drop=True)
        axes_no.text(label_size.time, label_size + 3, f'Event {int(event)}',
                     fontsize=20, ha='center', va='bottom',
                     color=presentation_bom_colours[0], zorder=zorder_text)


def plot_timeseries(time_unit, timeseries_type, plot_type='lines', mean_data=None, station_data=None):
    """
    Plots a time series of rainfall or runoff over a catchment area.
    
    Parameters
    ----------
    time_unit : str
        Time resolution of the data (e.g., 'daily', 'monthly'). Used for labels and file naming.
    timeseries_type : str
        Type of variable to plot, e.g., 'rainfall' or 'runoff'. Used for labels and file naming.
    plot_tye : str
        Type of plot style to use (line plot or dot plot)
    mean_data : xarray.DataArray, optional
        Gridded or averaged time series data with `.time`, `.values`, and `.units` attributes.
    station_data : pandas.DataFrame or similar, optional
        Station observation data to overlay on the plot.`.
    
    Notes
    -----
    - If both `mean_data` and `station_data` are None, the function aborts and prints a message.
    - Requires global variables `Lake`, `Lake_name`, and `output_dir` to be defined.
    - Saves the figure as:
      {output_dir}/{Lake}/{Lake}_{timeseries_type}_{time_unit}{file_suffix_mean}{file_suffix_station}.png
    """
    if mean_data is None and station_data is None:
        print("Nothing to plot. Aborting.")
        return
    
    if plot_type == 'lines':
        mean_linestyle = '-'
        mean_marker = ''
        mean_markersize = 0
    elif plot_type == 'dots':
        mean_linestyle = ''
        mean_marker = 'o'
        mean_markersize = 6

    fig, ax = plt.subplots(figsize=(40, 10))
    
    # === Plot timeseries of mean data ===
    if mean_data is not None:
        file_suffix_mean = '_mean'
        ax.plot(mean_data.time.values, mean_data.values, linestyle=mean_linestyle, marker=mean_marker, markersize=mean_markersize,
                label = f'{time_unit.capitalize()} mean {timeseries_type} over catchment area', color='maroon', zorder=1)
    else:
        file_suffix_mean = ''
        
    # === Plot timeseries of station data ===
    if station_data is not None:
        file_suffix_station = '_station'
        for i in range(station_data[f'{timeseries_type}_station'].size):
            ts = station_data.isel({f'{timeseries_type}_station': i})
            color = blue_colors[i % len(blue_colors)]
            station_id = ts[f'{timeseries_type}_station'].item()
            station_name = ts[f'{timeseries_type}_station_name'].item()
            ax.plot(ts.time.values, ts.values,
                         linestyle='', marker='o', markersize=6, alpha=1, label=f'{station_id}: {station_name}',color=color, zorder=1)
    else:
        file_suffix_station = ''
    
    # === Axis Formatting ===
    if mean_data is not None:
        format_variable = mean_data
    elif station_data is not None:
        format_variable = station_data
    ax.grid()
    ax.set_xlim(format_variable.time.min().values, format_variable.time.max().values)
    ax.legend(loc='upper left', bbox_to_anchor=(0.58, 0.98), fontsize=20, ncol=2)
    ax.set_title(f'{time_unit.capitalize()} {timeseries_type.capitalize()} {Lake_name}', fontsize=20)
    ax.set_ylabel(f'{timeseries_type.capitalize()} ({format_variable.units})', fontsize=20)
    plot_yearly_xticks(ax) # Set x-axis major ticks to yearly
    
    # === Save figure ===
    plt.savefig(f'{output_dir}/{Lake}/Timeseries/{timeseries_type.capitalize()}_timeseries/{Lake}_{timeseries_type}_{time_unit}{file_suffix_mean}{file_suffix_station}.png', bbox_inches='tight')
    plt.close()


def rainfall_scatter(da_station, da_griddded, percentile):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    
    for i in range(da_station.rainfall_station.size):
        station_id = da_station.rainfall_station.values[i]
        station_name = da_station.rainfall_station_name.values[i]
        x = da_station.isel(rainfall_station=i).values
        y = da_griddded.values
    
        # Drop missing data (keep only valid time steps)    
        valid_mask = (~np.isnan(x)) & (~np.isnan(y))
        x_vals = x[valid_mask]
        y_vals = y[valid_mask]
    
        # Filter to remove outliers based on both x and y
        x_thresh = np.percentile(x_vals,percentile)
        y_thresh = np.percentile(y_vals, percentile)
        outlier_mask = (x_vals <= x_thresh) & (y_vals <= y_thresh)
        
        x_vals = x_vals[outlier_mask]
        y_vals = y_vals[outlier_mask]
    
        ax.scatter(x_vals, y_vals, color=blue_colors[i], label=f'{station_id}: {station_name}')
    
        # Linear fit
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        ax.plot(x_vals, slope * x_vals + intercept, color=blue_colors[i], linestyle='--')
    
        # Correlation
        r, p = stats.pearsonr(x_vals, y_vals)
    
        # Add correlation label
        x_text = x_vals.max()
        y_text = slope * x_text + intercept
        
        ax.text(x_text + 0.2, y_text + 0.2, f'{station_id} r: {r:.2f}', color=blue_colors[i], fontsize=15)
        
        # Correlation
        r, p = stats.pearsonr(x_vals, y_vals)
        print(f'{station_id} → Correlation Coefficient: {r:.2f}, P-value: {p:.4f}')
    
    # Axis labels and title
    plt.xlabel('Station rainfall', fontsize=20)
    plt.ylabel('Catchment Mean rainfall (gridded)', fontsize=20)
    plt.title('Scatter Plot of gridded vs Station rainfall', fontsize=20)
    
    ax.tick_params(axis='both', labelsize=20)
    
    # Legend and save
    plt.legend(fontsize=20)
    plt.savefig(f'{output_dir}/{Lake}/Rainfall_scatter/{Lake}_station_rainfall_scatter_daily_{str(percentile)}.png', bbox_inches='tight')
    plt.close()
 
    
def plot_dea_timeseries(ds, time_unit_lake, event_labels=True, timeseries_type=None, time_unit_data=None, plot_mean=False, plot_station=False, peaks=False):
    """
    Plots DEA-derived lake size and detected flood events, optionally with catchment rainfall or runoff data.
    
    This function creates a time series plot showing lake surface area (from DEA) with highlighted flood events.
    Optionally, it can also overlay gridded mean data (e.g., catchment rainfall or runoff) and/or station observations 
    on a secondary y-axis. If neither plot_mean nor plot_station are set to True, only DEA events and lake size are plotted.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'dea' (lake size with dimension 'lake_variable') and 'dea_events'.
    time_unit_lake : str, optional
        Time resolution of the lake data (e.g., 'daily', 'monthly').
    event_labels : bool, optional
        If True, events are labeled in dea data. Default is True.
    timeseries_type : str, optional
        Type of hydrological data to plot (e.g., 'rainfall' or 'runoff').
    time_unit_data : str, optional
        Time resolution of the hydrological data (e.g., 'daily', 'monthly').
    plot_mean : bool, optional
        If True, overlay gridded mean data on the plot. Default is False.
    plot_station : bool, optional
        If True, overlay station observation data on the plot. Default is False.
    peaks : bool, optional
        If True, add precipitation peaks to plot.
    
    Notes
    -----
    - If both `plot_mean` and `plot_station` are False, only DEA-derived lake size and events are plotted.
    - Uses a dual y-axis: left for mean/station hydrological data, right for lake size.
    - Requires global variables: `Lake`, `output_dir`.
    - Depends on external functions:
        - `plot_yearly_xticks()` – formats x-axis with yearly ticks.
        - 'find_rainfall_peaks_near_lake_peaks' - indetifies rainfall peaks closest to events
    - Saves the figure to:
      `{output_dir}/{Lake}/{Lake}_FloodEvents-{time_unit_lake}_{timeseries_type.capitalize()}-{time_unit_data}{file_suffix_mean}{file_suffix_station}.png`
    """
    
    ## File naming
    # Only dea data plotted
    if not plot_mean and not plot_station:	
        if event_labels: file_name = f'{Lake}_FloodEvents-{time_unit_lake}'
        else: file_name = f'{Lake}_FloodEvents-{time_unit_lake}-not_labeled'
            
    # Dea data and mean and / or station data is plotted 
    else:
        # Peaks: only highlight if all conditions met. If not, issue warning and disable peaks
        if peaks and plot_mean and time_unit_data == 'monthly' and timeseries_type == 'rainfall':
            file_name_part1 = f'{Lake}_FloodEvents-{time_unit_lake}_RainfallPeaks-monthly'
        else:
            warnings.warn("Peaks can only be highlighted for monthly rainfall; plotting without peak markers.")            
            file_name_part1 = f'{Lake}_FloodEvents-{time_unit_lake}_{timeseries_type.capitalize()}-{time_unit_data}'
            peaks = None
        
        # Check if conditions for plotting mean and / or station data are met
        if timeseries_type is not None and time_unit_data is not None:
            file_suffix_mean = '_mean' if plot_mean else ''
            file_suffix_station = '_station' if plot_station else ''
            # Suffixes are added to already existing filename
            file_name = f'{file_name_part1}{file_suffix_mean}{file_suffix_station}'
        else:        	
            print("timeseries_type ('rainfall'/'runoff') or time_unit_data ('daily'/'monthly') missing. Aborting.")
            return
      
    ## Extract Data
    # DEA: use input ds if daily, otherwise convert to monthly. Abort if wrong time_unit.
    if time_unit_lake == 'daily': lake_size = ds['dea'].sel(lake_variable='Size')
    elif time_unit_lake == 'monthly':
        monthly_ds = resample_lake_dataset_to_monthly(ds)
        lake_size = monthly_ds['dea'].sel(lake_variable='Size')
    else: 
        print("time_unit_lake ('daily'/'monthly') missing or wrong. Aborting.")
	
    # Runoff/ Rainfall: Check conditions for plotting mean and / or station data
    if plot_mean or plot_station:
        if time_unit_data == 'daily':
            plot_type = 'dots'
            if plot_mean: mean_data = ds[f'mean_gridded_{timeseries_type}']
            if plot_station: station_data = ds[f'station_{timeseries_type}']
        elif time_unit_data == 'monthly':
            plot_type = 'lines'
            # If the ds has not been converted previously, convert to monthly.
            if time_unit_lake != 'monthly':
                monthly_ds = resample_lake_dataset_to_monthly(ds)
            if plot_mean:
                mean_data = monthly_ds[f'mean_gridded_{timeseries_type}']
                if peaks: rainfall_peaks_near_lake_peaks = find_rainfall_peaks_near_lake_peaks(ds, percentile=0.8)
            if plot_station: station_data = monthly_ds[f'station_{timeseries_type}']


    ### Create a large figure with two y-axes (twin axis)
    fig = plt.figure(figsize=(40, 10))
    ax1 = fig.add_subplot(111)        # Left y-axis for rainfall
    if not plot_mean and not plot_station:
        axis_dea = ax1
    else:
        ax2 = ax1.twinx()     # Right y-axis for lake size
        axis_dea = ax2
        axis_data = ax1
    
    if plot_mean or plot_station:	
        if plot_type == 'lines':
            linestyle_type = '-'
            marker_type = ''
            markersize_type = 0
        elif plot_type == 'dots':
            linestyle_type = ''
            marker_type = 'o'
            markersize_type = 6

    # === Plot timeseries of mean data ===
    if plot_mean:
        if peaks:
            mean_colours = blue_colors[2]
            linestyle_type = '--'
            ax1.plot(rainfall_peaks_near_lake_peaks.time.values, rainfall_peaks_near_lake_peaks.values,
                     linestyle='', marker='o', markersize=8, label = 'Catchment Rainfall Peaks', color='maroon', zorder=1)       
        else:
            mean_colours = 'maroon'
        axis_data.plot(mean_data.time.values, mean_data.values, linestyle=linestyle_type, marker=marker_type, markersize=markersize_type,
                       label = f'{time_unit_data.capitalize()} mean {timeseries_type} over catchment area', color=mean_colours, zorder=4)

    # === Plot timeseries of station data ===
    if plot_station:
        for i in range(station_data[f'{timeseries_type}_station'].size):
            ts = station_data.isel({f'{timeseries_type}_station': i})
            color = blue_colors[i % len(blue_colors)]
            station_id = ts[f'{timeseries_type}_station'].item()
            station_name = ts[f'{timeseries_type}_station_name'].item()
            axis_data.plot(ts.time.values, ts.values,
                         linestyle=linestyle_type, marker=marker_type, markersize=markersize_type, alpha=0.8,
                         label=f'{station_id}: {station_name}',color=color, zorder=3)
    
    # === Plot full time series of lake size as grey dots ===
    axis_dea.plot(lake_size.time.values, lake_size.values,
                 color='grey', linestyle='', marker='o', markersize=6, alpha=0.4, zorder=1)
    
    # === Plot each detected event with label ===
    if event_labels == True:
        plot_dea_events(ds, axis_dea, zorder_plot=5, zorder_text=10)
    
    ## === Axis Formatting ===
    if plot_mean:
        axis_data.set_ylabel(f'{timeseries_type.capitalize()} ({mean_data.units})', fontsize=20)
        axis_data.tick_params(axis='y', labelsize=20)
    elif plot_station:
        axis_data.set_ylabel(f'{timeseries_type.capitalize()} ({station_data.units})', fontsize=20)
        axis_data.tick_params(axis='y', labelsize=20)

    axis_dea.set_ylabel(f'Lake Size {lake_size.units}', fontsize=20)
    axis_dea.tick_params(axis='y', labelsize=20)

    ax1.grid(True, which='major', axis='both') # Add grid to the background
    plot_yearly_xticks(ax1) # Yearly ticks on x-axis
    
    # === Save figure ===
    plt.savefig(f'{output_dir}/{Lake}/Timeseries/DEA_timeseries/{file_name}.png', bbox_inches='tight')
    plt.close()          
 
    
        
def plot_rainfall_each_event(ds, rise=False, offset=False):
    """
    Plots rainfall and DEA lake surface area for each detected flood event.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing:        
        - 'rainfall': Catchment-averaged rainfall.
        - 'station_rainfall': Station-level rainfall (optional, but plotted if present).
        - 'dea': Lake size (with 'lake_variable'='Size').
        - 'dea_event_max': Maximum lake size per event.
        
    rise : bool, optional
        If True, only plot the rising limb of the lake size up to its peak during the event.
        Default is False (plots full event duration).

    offset : bool, optional
        If True, the xaxis is plotted as offset from max instead of dates.
        Default is False (plots full event duration).
    
    Notes:
    ------
    - The function uses global variables `output_dir` and `Lake` to construct file save paths.
    - Requires external functions: `plot_weekly_xticks`.
    """    
    
    # Extract variables
    events = np.unique(ds.dea_events.dropna(dim='time').values)
    rainfall = ds['mean_gridded_rainfall']
    station_rain_data = ds['station_rainfall']
    dea = ds['dea'].sel(lake_variable='Size')
    dea_max = dea.where(ds.dea_event_max, drop=True)
       
    # Loop through each event
    for event in events:
        event_size = dea.where(dea.dea_events == event, drop=True)
        event_max = dea_max.where(dea_max.dea_events == event, drop=True).dropna(dim='time')
        
        if rise:
            event_size = event_size.sel(time=slice(None, event_max.time.max()))
            file_suffix = 'Rise'
        else:
            file_suffix = 'EntireEvent'
        
        if offset:
            y_coord = 'dea_offset'
            file_suffix_offset = '_Offset'
            bar_width= 1
        else:
            y_coord = 'time'
            file_suffix_offset = '_Dates'
            bar_width = pd.Timedelta(days=1)
                
        rainfall_event = rainfall.sel(time=event_size.time)
        station_rainfall_event = station_rain_data.sel(time=event_size.time)
        station_rainfall_event = station_rainfall_event.dropna(dim="rainfall_station", how="all")
        
        # === Plot ===
        fig = plt.figure(figsize=(40, 10))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        
        #Rainfall
        ax1.plot(rainfall_event[y_coord], rainfall_event.values, color='maroon', linestyle='', marker='o', markersize=6, label = 'Catchment rainfall', zorder=1)
        
        for i in range(station_rainfall_event.sizes["rainfall_station"]):
            color = blue_colors[i % len(blue_colors)]
            ts = station_rainfall_event.isel(rainfall_station=i)
            station_id = ts.rainfall_station.item()
            station_name = ts.rainfall_station_name.item()
        
            ax1.plot(ts[y_coord], ts.values, linestyle='', marker='o', markersize=6, alpha=0.5,
            label=f'{station_id}: {station_name}', color=color, zorder=0)
        
        # DEA 
        ax2.bar(event_size[y_coord], event_size.values+1,width=bar_width, color=presentation_bom_colours[0])
        
        # Annotate lake size values
        for value in event_size.dropna(dim='time'):
            label = int(round(value.item(), 2))
            ax2.text(value[y_coord], label + 1, str(label), fontsize=20)
        
        # Mark event max in red
        ax2.bar(event_max[y_coord], event_max.values, width=bar_width, color='maroon', zorder=20)
        
        
        # === Axis Formatting ===
        ax1.grid()
        ax1.set_ylabel(f'rainfall ({rainfall_event.units}/day)', fontsize=20)
        ax2.set_ylabel(f'Lake Size ({event_size.units})', fontsize=20)

        # define tick settings        
        if offset:
            xticks = np.arange(event_size[y_coord].min(skipna=True), event_size[y_coord].max(skipna=True) + 1, 16)
            xtick_labels = [f"{int(x)}" for x in xticks]
            ax1.set_xticks(xticks, xtick_labels, fontsize=15)
        else:
            plot_weekly_xticks(ax1)
       
        ax1.tick_params(axis='y', labelsize=15)
        ax2.tick_params(axis='both', labelsize=15)
        #plt.title(f'Event {int(event)}', loc='left', fontsize=20)
        if ax1.get_legend_handles_labels()[0]:  # checks if there are any legend entries
            ax1.legend(fontsize=20, loc='upper left')
                
         # Save figure
        plt.savefig(f'{output_dir}/{Lake}/SingleFloodEvents/FloodEvents{file_suffix_offset}/Rainfall/{file_suffix}/{Lake}_FloodEvents_Rainfall{file_suffix}{file_suffix_offset}_Event{int(event)}.png', bbox_inches='tight')
        plt.close()
        
      
def plot_runoff_each_event(ds, rise=False, offset=False, cum=False):
    """
    Plots runoff and DEA lake surface area for each detected flood event.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing:        
        - 'rainfall': Catchment-averaged rainfall.
        - 'station_runoff': Station-level runoff (optional, but plotted if present).
        - 'dea': Lake size (with 'lake_variable'='Size').
        - 'dea_event_max': Maximum lake size per event.
        
    rise : bool, optional
        If True, only plot the rising limb of the lake size up to its peak during the event.
        Default is False (plots full event duration).

    offset : bool, optional
        If True, the xaxis is plotted as offset from max instead of dates.
        Default is False (plots full event duration).
    
    Notes:
    ------
    - The function uses global variables `output_dir` and `Lake` to construct file save paths.
    - Requires external functions: `plot_weekly_xticks`.
    """    
    
    # Extract variables
    events = np.unique(ds.dea_events.dropna(dim='time').values)
    station_runoff_data = ds['station_runoff']
    #station_runoff_cum = ds['station_runoff_cum']
    dea = ds['dea'].sel(lake_variable='Size')
    dea_max = dea.where(ds.dea_event_max, drop=True)
       
    # Loop through each event
    for event in events:
        event_size = dea.where(dea.dea_events == event, drop=True)
        event_max = dea_max.where(dea_max.dea_events == event, drop=True).dropna(dim='time')
        
        if cum:
            event_size = event_size.sel(time=slice(None, event_max.time.max()))
            file_suffix = 'Cumulative'
        else:
            if rise:
                event_size = event_size.sel(time=slice(None, event_max.time.max()))
                file_suffix = 'Rise'
            else:
                file_suffix = 'EntireEvent'
        
        if offset:
            y_coord = 'dea_offset'
            file_suffix_offset = '_Offset'
            bar_width= 1
        else:
            y_coord = 'time'
            file_suffix_offset = '_Dates'
            bar_width = pd.Timedelta(days=1)
                
        station_runoff_event = station_runoff_data.sel(time=event_size.time)
        station_runoff_event = station_runoff_event.dropna(dim="runoff_station", how="all")
        
        #cumulative_station_runoff_event = station_runoff_cum.sel(time=slice(min_date, max_date))
        #cumulative_station_runoff_event = cumulative_station_runoff_event.dropna(dim="runoff_station", how="all")

        # === Plot ===
        fig = plt.figure(figsize=(40, 10))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        
        #Runoff        
        for i in range(station_runoff_event.sizes["runoff_station"]):
            color = blue_colors[i % len(blue_colors)]
            ts = station_runoff_event.isel(runoff_station=i)
            station_id = ts.runoff_station.item()
            station_name = ts.runoff_station_name.item()
        
            ax1.plot(ts[y_coord], ts.values, linestyle='', marker='o', markersize=6, alpha=0.5,
            label=f'{station_id}: {station_name}', color=color, zorder=0)

        #Cumulative rainfall
        #if cum:            
            #for i in range(cumulative_station_runoff_event.sizes["runoff_station"]):
               # color = blue_colors[i % len(blue_colors)]
               # ts = cumulative_station_runoff_event.isel(runoff_station=i)
               # station_id = ts.runoff_station.item()
               # station_name = ts.runoff_station_name.item()
        
               # ax1.plot(ts[y_coord], ts.values, linestyle='-', label=f'{station_id}: {station_name}', color=color, zorder=0)
        
        # DEA 
        ax2.bar(event_size[y_coord], event_size.values+1,width=bar_width, color=presentation_bom_colours[0])
        
        # Annotate lake size values
        for value in event_size.dropna(dim='time'):
            label = int(round(value.item(), 2))
            ax2.text(value[y_coord], label + 1, str(label), fontsize=20)
        
        # Mark event max in red
        ax2.bar(event_max[y_coord], event_max.values, width=bar_width, color='maroon', zorder=20)
        
        
        # === Axis Formatting ===
        ax1.grid()
        ax1.set_ylabel(f'Runoff ({station_runoff_event.units}/day)', fontsize=20)
        ax2.set_ylabel(f'Lake Size ({event_size.units})', fontsize=20)

        # define tick settings        
        if offset:
            xticks = np.arange(event_size[y_coord].min(skipna=True), event_size[y_coord].max(skipna=True) + 1, 16)
            xtick_labels = [f"{int(x)}" for x in xticks]
            ax1.set_xticks(xticks, xtick_labels, fontsize=15)
        else:
            plot_weekly_xticks(ax1)
       
        ax1.tick_params(axis='y', labelsize=15)
        ax2.tick_params(axis='both', labelsize=15)
        #plt.title(f'Event {int(event)}', loc='left', fontsize=20)
        if ax1.get_legend_handles_labels()[0]:  # checks if there are any legend entries
            ax1.legend(fontsize=20, loc='upper left')
        
         # Save figure
        plt.savefig(f'{output_dir}/{Lake}/SingleFloodEvents/FloodEvents{file_suffix_offset}/Runoff/{file_suffix}/{Lake}_FloodEvents_Runoff{file_suffix}{file_suffix_offset}_Event{int(event)}.png', bbox_inches='tight')
        plt.close()


def plot_cumulative_rainfall_each_event(ds, offset=False):

    ### ========= Extract variables from dataset =========
    events = np.unique(ds.dea_events.dropna(dim='time').values)
    event_offset_rise = ds.dea_offset.where(ds.dea_offset <= 0, drop=True)

    rainfall = ds['mean_gridded_rainfall']
    station_rainfall = ds['station_rainfall']

    rainfall_cum = ds['mean_gridded_rainfall_event_cum']
    rainfall_cum_filtered = ds['mean_gridded_rainfall_event_cum_filtered']
    station_rainfall_cum = ds['station_rainfall_event_cum']

    dea = ds['dea'].sel(lake_variable='Size')
    dea_max = dea.where(ds.dea_event_max, drop=True)
    # ====================================================
    
    if offset:
        y_coord = 'dea_offset'
        file_suffix_offset = '_Offset'
        bar_width= 1
    else:
        y_coord = 'time'
        file_suffix_offset = '_Dates'
        bar_width = pd.Timedelta(days=1)

    # Loop through each event
    for event in events:
        offset_event_rise = event_offset_rise.where(event_offset_rise.dea_events == event, drop=True)    
        min_date = offset_event_rise.time.min()
        max_date = offset_event_rise.time.max()
 

        # === Plot ===
        fig = plt.figure(figsize=(40, 10))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        #Rainfall
        rainfall_event = rainfall.sel(time=slice(min_date, max_date))
        ax1.plot(rainfall_event[y_coord], rainfall_event.values, color='maroon', linestyle='', marker='o', markersize=6, label = 'Catchment rainfall', zorder=1)
        
        station_rainfall_event = station_rainfall.sel(time=slice(min_date, max_date)).dropna(dim="rainfall_station", how="all")
        for i in range(station_rainfall_event.sizes["rainfall_station"]):
            color = blue_colors[i % len(blue_colors)]
            ts = station_rainfall_event.isel(rainfall_station=i)
            station_id = ts.rainfall_station.item()
            station_name = ts.rainfall_station_name.item()
        
            ax1.plot(ts[y_coord], ts.values, linestyle='', marker='o', markersize=6, alpha=0.5,
            label=f'{station_id}: {station_name}', color=color, zorder=0)

        #Cumulative Rainfall
        rainfall_cum_event = rainfall_cum.sel(time=slice(min_date, max_date))
        ax1.plot(rainfall_cum_event[y_coord], rainfall_cum_event.values, color='maroon', linestyle='-', label = 'Cumulative Catchment Rainfall', zorder=1)   

        station_rainfall_cum_event = station_rainfall_cum.sel(time=slice(min_date, max_date))
        station_rainfall_cum_event = station_rainfall_cum_event.sel(
            rainfall_station=~(station_rainfall_cum_event == 0).all(dim="time")).dropna(dim="rainfall_station", how="all")
        for i in range(station_rainfall_cum_event.sizes["rainfall_station"]):
            color = blue_colors[i % len(blue_colors)]
            ts = station_rainfall_cum_event.isel(rainfall_station=i)
            station_id = ts.rainfall_station.item()
            station_name = ts.rainfall_station_name.item()
    
            ax1.plot(ts[y_coord], ts.values, linestyle='-', label=f'{station_id}: {station_name}', color=color, zorder=0)
            
        #Cumulative Rainfall Filtered
        rainfall_cum_filtered_event = rainfall_cum_filtered.sel(time=slice(min_date, max_date))
        ax1.plot(rainfall_cum_filtered_event[y_coord], rainfall_cum_filtered_event.values+10, color='green', linestyle='-', label = 'Cumulative Catchment Rainfall Filtered', zorder=1)   

        # DEA 
        event_size = dea.sel(time=slice(min_date, max_date)).dropna(dim='time')
        ax2.bar(event_size[y_coord], event_size.values+1, width=bar_width, color=presentation_bom_colours[0])
        
        # Annotate lake size values
        for value in event_size:
            label = int(round(value.item(), 2))
            ax2.text(value[y_coord], label + 1, str(label), fontsize=20)
        
        # Mark event max in red
        event_max = dea_max.where(dea_max.dea_events == event, drop=True).dropna(dim='time')
        ax2.bar(event_max[y_coord], event_max.values, width=bar_width, color='maroon', zorder=20)
        
    
        # === Axis Formatting ===
        ax1.grid()
        ax1.set_ylabel(f'Rainfall ({rainfall_event.units}/day)', fontsize=20)
        ax2.set_ylabel(f'Lake Size ({event_size.units})', fontsize=20)

        # define tick settings              
        if offset:
            xticks = np.arange(event_size[y_coord].min(skipna=True), event_size[y_coord].max(skipna=True) + 1, 16)
            xtick_labels = [f"{int(x)}" for x in xticks]
            ax1.set_xticks(xticks, xtick_labels, fontsize=15)
        else:
            plot_weekly_xticks(ax1)
       
        ax1.tick_params(axis='y', labelsize=15)
        ax2.tick_params(axis='both', labelsize=15)
        if ax1.get_legend_handles_labels()[0]:  # checks if there are any legend entries
            ax1.legend(fontsize=20, loc='upper left')
                
        # Save figure
        plt.savefig(f'{output_dir}/{Lake}/SinglefloodEvents/FloodEvents{file_suffix_offset}/Rainfall/Cumulative/{Lake}_FloodEvents_RainfallCumulative_Filtered{file_suffix_offset}_Event{int(event)}.png', bbox_inches='tight')
        plt.close()


def plot_cumulative_rainfall_all_events(ds, magnitude=False, filtered=False):
    """
    Plot cumulative rainfall for all detected flood events in a dataset.

    This function generates cumulative rainfall time series aligned to the
    flood event rise (dea_offset <= 0). Each event is plotted separately, either
    with distinct preset colours or ranked and coloured by flood magnitude.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the following variables:
            - 'dea_events': flood event identifiers (per timestep).
            - 'dea_event_max': maximum lake size values for each event.
            - 'mean_gridded_rainfall': daily mean rainfall over catchment area.
            - 'mean_gridded_rainfall_cum': cumulative mean rainfall.
            - 'dea_offset': day offsets relative to flood event peaks.
    magnitude : bool, optional (default=False)
        If True, events are ranked by maximum lake size and coloured by magnitude.
        If False, events are coloured using a predefined colour list.
    filtered : bool, optional (default=False)
        If True, cumulative rainfall is filtered by rate of chage thresholds.
        If False, the entire cumulative rainfall from start to end of event is plotted.

    Output
    ------
    Saves a PNG figure in the following location:
        {output_dir}/{Lake}/SingleFloodEvents/FloodEvents_Offset/{Lake}_FloodEvents_rainfall_Offset_Rise_Cumulative_AllEvents{suffix}_1.png

    Notes
    -----
    - Axis ticks are set to 16-day intervals.
    - When `magnitude=True`, a colourbar is added indicating smallest → largest
      flood event.
    - When `magnitude=False`, a legend is added with event IDs.
    """
    
    if filtered:
    	filterd_var = '_filtered'
    else:
        filterd_var = ''
		
    # Extract variables
    events = np.unique(ds.dea_events.dropna(dim='time').values)
    dea_max = ds['dea'].sel(lake_variable='Size').where(ds.dea_event_max, drop=True)
    mean_rainfall = ds['mean_gridded_rainfall']                             
    mean_gridded_rainfall_cum = ds[f'mean_gridded_rainfall_event_cum{filterd_var}'] 
    event_offset_rise = ds.dea_offset.where(ds.dea_offset <= 0, drop=True)
    if filtered:
    	event_offset_rise_filtered = ds['dea_offset_filtered'].where(ds['dea_offset_filtered'] <= 0, drop=True)
    
    if magnitude:
        #Rank lake size maximas in descending order
        lake_size_maximums = dea_max.dropna(dim='time')
        ranked_lake_size_max = lake_size_maximums.sortby(lake_size_maximums, ascending=True)
        event_ranking = ranked_lake_size_max['dea_events'].values.astype(int).tolist()
    
        # Colourmap
        original_cmap = colormaps['Blues']
        cmap = truncate_colormap(original_cmap, minval=0.2, maxval=1.0)
        norm = plt.Normalize(vmin=1, vmax=len(event_ranking))
        colours = [cmap(norm(i + 1)) for i in range(len(event_ranking))]
        
        suffix = '_Magnitude'
        
    else:
        event_ranking = events
        colours = dea_colour_list
        suffix = '' 

    # === PLOT ===
    # Set up the figure and axis
    fig = plt.figure(figsize=(30, 10))
    ax = fig.add_subplot(111)
    
    for i, event in enumerate(event_ranking):
        offset_event_rise = event_offset_rise.where(event_offset_rise.dea_events == event, drop=True)
        min_date = offset_event_rise.time.min()
        max_date = offset_event_rise.time.max()
        if filtered:
            offset_event_rise_filtered = event_offset_rise_filtered.sel(time=slice(min_date, max_date))
            if offset_event_rise_filtered.any():
                min_date_filtered = offset_event_rise_filtered.time.min()
                max_date_filtered = offset_event_rise_filtered.time.max()
                cum_rainfall_event_filtered = mean_gridded_rainfall_cum.sel(time=slice(min_date_filtered, max_date_filtered))
                #rainfall
                ax.plot(cum_rainfall_event_filtered.dea_offset_filtered, cum_rainfall_event_filtered.values, color=colours[i], linestyle='-', label = f'Event {int(event)}', zorder=1)        	
        	
        else:
            cum_rainfall_event = mean_gridded_rainfall_cum.sel(time=slice(min_date, max_date))
            #rainfall
            ax.plot(cum_rainfall_event.dea_offset, cum_rainfall_event.values, color=colours[i], linestyle='-', label = f'Event {int(event)}', zorder=1)
    
    # === Axis Formatting ===
    # define tick settings
    xticks = np.arange(event_offset_rise.min(skipna=True).item(), event_offset_rise.max(skipna=True).item() + 1, 16)
    xtick_labels = [f"{int(x)}" for x in xticks]
    ax.set_xticks(xticks, xtick_labels, fontsize=15)
    ax.tick_params(axis='y', labelsize=15)
    
    #Grid and Labels
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylabel(f'Cumulative Rainfall ({mean_rainfall.units})', fontsize=20)
    
    if magnitude:
        #colourbar
        cbar_ax = inset_axes(ax, width=0.5, height=5, loc='upper left', bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes, borderpad=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_ticks([events.min(), events.max()])
        cbar.set_ticklabels(['smallest', 'highest'])
        cbar.ax.tick_params(labelsize=20) 
    else:
        # Legend
        ax.legend(fontsize=20, loc='upper left')
    
    # save plot
    plt.savefig(f'{output_dir}/{Lake}/Cumulative_Rainfall/Cumulative_Rainfall_Offset/{Lake}_FloodEvents_Rainfall_Offset_Rise_Cumulative_AllEvents{suffix}{filterd_var}.png', bbox_inches='tight')
    plt.close() 

        
def plot_rolling_sum_window(ds, start_date, window, time_unit, threshold1, threshold2, plot_peaks_troughs=False, peaks_troughs=None):
    """
    Plot cumulative rainfall and lake events, highlighting periods above specified thresholds.

    Parameters:
        ds (xr.Dataset): Dataset containing rainfall and DEA lake data.
        peaks_troughs (dict): Dictionary with 'peaks' and 'troughs' as xr.DataArrays.
        start_date (str): Start of the plotting period.
        window (int): Rolling window size.
        time_unit (str): 'months' or 'days'.
        threshold1 (float): Lower rainfall threshold to highlight.
        threshold2 (float): Higher rainfall threshold to highlight.
    """
    if start_date=='1900':
        grid_spacing=5
    else:
        grid_spacing=1

    #Extract variables
    rainfall_sum = ds['mean_gridded_rainfall_rolling_sum']
    rainfall_sum_above_threshold1 = rainfall_sum.where(rainfall_sum[f'rolling_peak_above_{threshold1}'])
    rainfall_sum_above_threshold2 = rainfall_sum.where(rainfall_sum[f'rolling_peak_above_{threshold2}'])
    dea = ds['dea'].sel(lake_variable='Size')
		
    # Create a large figure with two y-axes (twin axis)
    fig = plt.figure(figsize=(40, 10))
    ax1 = fig.add_subplot(111)        # Left y-axis for rainfall
    ax2 = ax1.twinx()                 # Right y-axis for lake size

    # === Plot rolling cumulative rainfall ===
    ax1.plot(rainfall_sum.time, rainfall_sum.values,
             linestyle='-', label=f'{window} {time_unit.capitalize()} Rolling Cumulative Sum of Rainfall', alpha=0.5, color='#b3cde0', zorder=2)
    
    ax1.plot(rainfall_sum_above_threshold1.time, rainfall_sum_above_threshold1.values,
             linestyle='-', label=f'Peak above {threshold1} mm', alpha=0.5, color='#011f4b', zorder=3)
    
    ax1.plot(rainfall_sum_above_threshold2.time, rainfall_sum_above_threshold2.values,
             linestyle='-', label=f'Peak above {threshold2} mm', alpha=0.5, color='maroon', zorder=4)
    
    # === Plot peaks and troughs ===
    if plot_peaks_troughs:
        if peaks_troughs is None:
            raise ValueError("If plot_peaks_troughs is True, you must provide a peaks_troughs dictionary.")

        peaks = peaks_troughs['peaks']
        troughs = peaks_troughs['troughs']

        ax1.plot(peaks.time, peaks.values, linestyle='', marker='o', label='Peaks', alpha=0.5, color='maroon', zorder=2)
        ax1.plot(troughs.time, troughs.values,linestyle='', marker='o', label='Troughs', alpha=0.5, color='green', zorder=2)
    
    # === Plot full time series of lake size with event labels ===    
    ax2.plot(dea.time.values, dea.values,
                 color='grey', linestyle='', marker='o', markersize=8, alpha=0.4, zorder=1)

    plot_dea_events(ds, ax2, zorder_plot=5, zorder_text=10)
    
    # === Add a horizontal line at threshold levels ===
    ax1.axhline(y=threshold1, color='#03396c', linestyle='--')
    ax1.axhline(y=threshold2, color='maroon', linestyle='--')
    	
    # === Axis Formatting ===
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=20)
    ax1.grid(True, which='major', axis='both') # Add grid to the background
    ax1.xaxis.set_major_locator(mdates.YearLocator(grid_spacing))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax1.get_xticklabels(), rotation=90, ha='center', fontsize=20)
    ax1.set_ylabel(f'Cumulative Rainfall ({rainfall_sum.units}/{window} {time_unit})', fontsize=20)    	    
    ax1.tick_params(axis='y', labelsize=20)
    ax2.set_ylabel(f'Lake Size {dea.units}', fontsize=20)    	    
    ax2.tick_params(axis='y', labelsize=20)

    # === Save figure ===
    plt.savefig(f'{output_dir}/{Lake}/Timeseries/Rolling_timeseries/{Lake}_{start_date}to2024_{window}{time_unit}.pdf', bbox_inches='tight')
    plt.close()


def plot_rolling_sum_variable_window(ds, start_date, time_unit, threshold, window_min, window_max):
    
    if time_unit == 'days':
        distance_val = 180
        time_step = 16
    elif time_unit == 'months':
        distance_val = 6
        time_step = 1
    else:
        print("Wrong time_unit. Needs to 'days' or 'months'.")
        return
    
    # Colourmap
    norm = plt.Normalize(vmin=1, vmax=((window_max - window_min)/time_step)+1)
    cmap_blue = truncate_colormap(colormaps['Blues'], minval=0.2, maxval=1.0)
    colours_blue = [cmap_blue(norm(i + 1)) for i in range(int(((window_max - window_min)/time_step)+1))]
    cmap_red = truncate_colormap(colormaps['Reds'], minval=0.2, maxval=1.0)
    colours_red = [cmap_red(norm(i + 1)) for i in range(int(((window_max - window_min)/time_step)+1))]
    
    #Extract variables
    dea = ds['dea'].sel(lake_variable='Size')

    # Create a large figure with two y-axes (twin axis)
    fig = plt.figure(figsize=(40, 10))
    ax1 = fig.add_subplot(111)        # Left y-axis for rainfall
    ax2 = ax1.twinx()                 # Right y-axis for lake size
    
    for window in range(window_min,window_max+1,time_step):
        i = int((window-window_min)/time_step)
    
        # Step 1: Calculate rolling cumulative sum of mean rainfall
        rainfall_sum = ds['mean_gridded_rainfall'].rolling(time=window, center=False).sum()
        
        # Step 2: check if peaks are above threshold
        # Step 2a: Troughs are used to segment the time series into rainfall episodes
        peaks_and_troughs = identify_peaks_and_troughs(rainfall_sum, distance_val)
    
        # Sanity check
        if not (len(peaks_and_troughs['peaks']) == len(peaks_and_troughs['troughs']) or len(peaks_and_troughs['peaks']) == len(peaks_and_troughs['troughs']) - 1):
            raise ValueError("Number of peaks must be equal to or one less than number of troughs.")
    
        peak_dates = rainfall_sum.time[peaks_and_troughs['peaks']]
        rolling_sum_peaks = rainfall_sum.sel(time=peak_dates)
    
        # Step 2b: Loop through each segment defined by two troughs
        peak_above_thresh = xr.DataArray(np.zeros(rainfall_sum.time.size, dtype=bool),
                                          coords={"time": rainfall_sum.time}, dims=["time"])
        
        for j in range(len(peaks_and_troughs['troughs']) - 1):
            # Time slice between consecutive troughs
            start_idx = peaks_and_troughs['troughs'][j]
            end_idx = peaks_and_troughs['troughs'][j+1]
            time_slice = rainfall_sum.time[start_idx:end_idx]
    
            # Peak info
            peak = rolling_sum_peaks[j].values
    
            # Mark boolean coordinates for thresholds
            if peak >= threshold:
                peak_above_thresh.loc[dict(time=time_slice)] = True
    
        # Assign boolean coordinates to rolling_sum dynamically based on threshold values (3b)
            rainfall_sum = rainfall_sum.assign_coords({
            'rolling_peak_above_threshold': peak_above_thresh
        })
        
        rainfall_sum_above_threshold = rainfall_sum.where(rainfall_sum.rolling_peak_above_threshold, drop=False)
    
        # Step 3: Plot
        ax1.plot(rainfall_sum.time, rainfall_sum.values,
                 linestyle='-', label=f'{window} {time_unit.capitalize()} Rolling Cumulative Sum of Rainfall', alpha=0.5, color=colours_blue[i], zorder=2)
    
        ax1.plot(rainfall_sum_above_threshold.time, rainfall_sum_above_threshold.values,
                 linestyle='-', label=f'{window} {time_unit.capitalize()} Rolling Cumulative Sum of Rainfall', alpha=0.5, color=colours_red[i], zorder=2)
        
    # === Plot full time series of lake size with event labels ===    
    ax2.plot(dea.time.values, dea.values,
                 color='grey', linestyle='', marker='o', markersize=8, alpha=0.4, zorder=1)
    
    plot_dea_events(ds, ax2, zorder_plot=5, zorder_text=10)
        
    # === Axis Formatting ===
    ax1.set_ylabel(f'Cumulative Rainfall ({rainfall_sum.units})', fontsize=20)    	
    #ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=20)
    ax1.grid(True, which='major', axis='both') # Add grid to the background
    plot_yearly_xticks(ax1) # Yearly ticks on x-axis
    	
    # === Save figure ===
    plt.savefig(f'{output_dir}/{Lake}/Timeseries/Rolling_timeseries/{Lake}_{start_date}to2024_window_{window_min}-{window_max}_{time_unit}.png', bbox_inches='tight')
    plt.close()
    

def plot_cum_rainfall_with_threshold(ds, threshold, time_interval):
    """
    Plot cumulative rainfall segments associated with lake-filling events
    and non-events, categorized by whether they fall below or above a given threshold.
    
    Parameters:
        ds (xr.Dataset): Input dataset containing cumulative rainfall segments.
        threshold (float): rainfall threshold to distinguish event types.
        time_interval (str): Unit of time with window (e.g. '128_days').
    """
    
    window, time_unit = time_interval.split("_", 1)
    
    cum_to_peak = ds.mean_gridded_rainfall_cum_to_peak
    
    # === No-Event Segments (after 1987) ===
    no_event = cum_to_peak.where(cum_to_peak.cum_window_is_event.isnull(), drop=True).dropna(dim='time')
    dates = no_event.where(no_event[f'cum_window_{time_interval}'] == 1, drop=True).sel(time=slice('1987', None))
    if dates.size > 0:
        start = dates[0].time
        # Filter only segments after that date
        no_event = no_event.sel(time=slice(start, None))
    else:
        no_event = xr.DataArray([], dims='time')

    # === Load "event" segments (always post-1987) ===
    event = cum_to_peak.where(~cum_to_peak.cum_window_is_event.isnull(), drop=True).dropna(dim='time')
    
    # === Start Plot ===
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(111)
    
    # === Plot each group ===
    for i in range(event.size // int(window)):
        event_seg = event[int(window) * i : int(window) * (i+1)]
        #event_no = int(np.unique(event_seg.cum_window_is_event.values).item())
        event_seg_below = event_seg.where(~event_seg[f'cum_window_above_{threshold}'], drop=True)
        event_seg_above = event_seg.where(event_seg[f'cum_window_above_{threshold}'], drop=True)
        ax1.plot(event_seg_below[f'cum_window_{time_interval}'], event_seg_below.values, color='maroon', linestyle='-', markersize=6, label = f'Lake Filling Event - below threshold of {threshold} mm', zorder=1)
        ax1.plot(event_seg_above[f'cum_window_{time_interval}'], event_seg_above.values, color='#005b96', linestyle='-', markersize=6, label = 'Lake Filling Event', zorder=1)
        #ax1.text(int(window)+(event_no/5), event_seg.max(), str(event_no), fontsize=8, color='#005b96')
        
    for i in range(no_event.size // int(window)):
        no_event_seg = no_event[int(window) * i : int(window) * (i+1)]
        no_event_seg_below = no_event_seg.where(~no_event_seg[f'cum_window_above_{threshold}'], drop=True)
        no_event_seg_above = no_event_seg.where(no_event_seg[f'cum_window_above_{threshold}'], drop=True)
        ax1.plot(no_event_seg_below[f'cum_window_{time_interval}'], no_event_seg_below.values, color='#b3cde0', linestyle='--', markersize=6, label = 'No Lake Filling', zorder=1)
        ax1.plot(no_event_seg_above[f'cum_window_{time_interval}'], no_event_seg_above.values, color='maroon', linestyle='--', markersize=6, label = f'No Lake Filling - above threshold of {threshold} mm', zorder=1)

    
    # === Clean Legend (avoid duplicates) ===
    handles = [
        mlines.Line2D([], [], color='#005b96', linestyle='-', label='Lake Filling'),
        mlines.Line2D([], [], color='maroon', linestyle='-', label=f'Lake Filling - below {threshold} mm'),
        mlines.Line2D([], [], color='maroon', linestyle='--', label=f'No Lake Filling - above {threshold} mm'),
        mlines.Line2D([], [], color='#b3cde0', linestyle='--', label='No Lake Filling')
    ]
    ax1.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, 1), fontsize=15)
    
    # === Threshold line ===
    ax1.axhline(y=threshold, color='#03396c', linestyle='--')
    
    # === Formatting ===
    ax1.set_xlabel(f'{time_unit.capitalize()} Leading up to Peak', fontsize=18)
    ax1.set_ylabel('Cumulative Rainfall (mm)', fontsize=18)
    ax1.grid(True)
    xticks = np.arange(0, int(window), 10)
    xtick_labels = [f"{int(x)}" for x in xticks]
    ax1.set_xticks(xticks, xtick_labels, fontsize=13)
    ax1.tick_params(axis='both', labelsize=13)
    
    # === Save Plot ===
    plt.savefig(f'{output_dir}/{Lake}/Cumulative_Rainfall/Cumulative_Rainfall_Rolling_Max/{Lake}_CumRainfall_BeforeRollingMax_{window}{time_unit}.png', bbox_inches='tight')
    plt.close() 


def plot_events_with_enso(ds, enso_index, enso_index_name):
    """
    Plot DEA-derived lake size and flood events together with an ENSO index.

    This function visualizes the relationship between lake-filling events (from DEA)
    and ENSO variability by overlaying the selected ENSO index as a filled 
    positive/negative anomaly plot on a dual y-axis figure.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'dea' (lake surface area) and 'enso' variables.
        'dea' must include a coordinate 'lake_variable' with value 'Size'.
        'enso' must have dimensions ('time', 'enso_indices').
    enso_index : str
        The specific ENSO index to select from the 'enso' variable
        (e.g. 'oni', 'nino34_ersst', 'nino34_hadisst').
    enso_index_name : str
        Readable name of the ENSO index used for axis labeling
        (e.g. "Oceanic Niño Index (ONI)").

    Notes
    -----
    - Requires global variables:
        * Lake (str): lake identifier, used in plot title and file naming.
        * output_dir (str): base directory for figure output.
    - Uses external helper function:
        * `plot_dea_events(ds, axis, zorder_plot, zorder_text)`
          to annotate flood events on the plot.
    - ENSO data are shaded red for positive anomalies and blue for negative anomalies.
    - Lake size is plotted as grey markers with event labels overlaid.

    """
    fig = plt.figure(figsize=(40, 10))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    
    enso=ds['enso'].sel(enso_indices=enso_index).sel(time=slice("1985-01-01", "2025-01-01"))
    dea =ds['dea'].sel(lake_variable='Size')
    
    # Fill the positive and negative areas with different colors
    ax1.fill_between(enso.time, enso.values, where=enso.values >= 0, color='lightcoral', alpha=0.5)
    ax1.fill_between(enso.time, enso.values, where=enso.values < 0, color='lightblue', alpha=0.5)
    
    # === Plot full time series of lake size with event labels ===    
    ax2.plot(dea.time.values, dea.values,
                 color='grey', linestyle='', marker='o', markersize=8, alpha=0.4, zorder=1)
    plot_dea_events(ds, ax2, zorder_plot=5, zorder_text=10)
    
    ax1.grid()
    plt.xlabel('Date', fontsize=20) 
    ax2.set_ylabel('Lake Pixel Count', fontsize=20)
    ax1.set_ylabel(enso_index_name, fontsize=20)
    
    # Set y-ticks and y-tick labels for ax1
    ax1.set_yticks(np.arange(-3, 3.1, 1))
    ax1.set_yticklabels([f"{int(x)}" for x in np.arange(-3, 3.1, 1)], fontsize=20)
    # Set y-ticks and y-tick labels for ax2
    ax2.set_yticks(np.arange(-750,751,250))
    ax2.set_yticklabels(list(np.arange(-750,751,250)), fontsize=20)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax1.get_xticklabels(), rotation=90, ha='center', fontsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    plt.savefig(f'{output_dir}/{Lake}/Timeseries/Drivers/{Lake}_FloodEvents_ENSO_{enso_index}.png', bbox_inches='tight')
    plt.close()    


def plot_events_with_ipo(ds, ipo_index):
    """
    Plot DEA-derived lake size and flood events together with an IPO index.

    This function visualizes the relationship between lake-filling events (from DEA)
    and IPO variability by overlaying the selected IPO index as a filled 
    positive/negative anomaly plot on a dual y-axis figure.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'dea' (lake surface area) and 'ipo' variables.
        'dea' must include a coordinate 'lake_variable' with value 'Size'.
        'ipo' must have dimensions ('time', 'ipo_indices').
    ipo_index : str
        The specific IPO index to select from the 'ipo' variable
        (e.g. 'oni', 'nino34_ersst', 'nino34_hadisst').

    Notes
    -----
    - Requires global variables:
        * Lake (str): lake identifier, used in plot title and file naming.
        * output_dir (str): base directory for figure output.
    - Uses external helper function:
        * `plot_dea_events(ds, axis, zorder_plot, zorder_text)`
          to annotate flood events on the plot.
    - IPO data are shaded red for positive anomalies and blue for negative anomalies.
    - Lake size is plotted as grey markers with event labels overlaid.

    """
    fig = plt.figure(figsize=(40, 10))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    
    ipo=ds['ipo'].sel(ipo_indices=ipo_index).sel(time=slice("1985-01-01", "2025-01-01"))
    dea =ds['dea'].sel(lake_variable='Size')
    
    # Fill the positive and negative areas with different colors
    ax1.fill_between(ipo.time, ipo.values, where=ipo.values >= 0, color='lightcoral', alpha=0.5)
    ax1.fill_between(ipo.time, ipo.values, where=ipo.values < 0, color='lightblue', alpha=0.5)
    
    # === Plot full time series of lake size with event labels ===    
    ax2.plot(dea.time.values, dea.values,
                 color='grey', linestyle='', marker='o', markersize=8, alpha=0.4, zorder=1)
    plot_dea_events(ds, ax2, zorder_plot=5, zorder_text=10)
    
    ax1.grid()
    plt.xlabel('Date', fontsize=20) 
    ax2.set_ylabel('Lake Pixel Count', fontsize=20)
    ax1.set_ylabel(f'IPO ({ipo_index})', fontsize=20)
    
    # Set y-ticks and y-tick labels for ax1
    values = np.arange(-0.6, 0.7, 0.2)
    ax1.set_yticks(values)
    ax1.set_yticklabels([f"{v:.1f}" for v in values], fontsize=20)

    # Set y-ticks and y-tick labels for ax2
    ax2.set_yticks(np.arange(-750,751,250))
    ax2.set_yticklabels(list(np.arange(-750,751,250)), fontsize=20)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax1.get_xticklabels(), rotation=90, ha='center', fontsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    plt.savefig(f'{output_dir}/{Lake}/Timeseries/Drivers/{Lake}_FloodEvents_IPO_{ipo_index}.png', bbox_inches='tight')
    plt.close()  


def plot_rolling_sum_with_enso(ds, start_date, window, time_unit, threshold1, threshold2, enso_index, enso_index_name):
    if start_date=='1900':
        grid_spacing=10
    else:
        grid_spacing=1
    
    #Extract variables
    rainfall_sum = ds['mean_gridded_rainfall_rolling_sum']
    rainfall_sum_above_threshold1 = rainfall_sum.where(rainfall_sum[f'rolling_peak_above_{threshold1}'])
    rainfall_sum_above_threshold2 = rainfall_sum.where(rainfall_sum[f'rolling_peak_above_{threshold2}'])
    dea = ds['dea'].sel(lake_variable='Size')
    enso=ds['enso'].sel(enso_indices=enso_index).sel(time=slice(f"{start_date}-01-01", "2025-01-01"))
    	
    ### PLOT: timeseries of 128 day cumulative rainfall with SOI ###
    fig = plt.figure(figsize=(40, 10))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 120))
    
    # Fill the positive and negative areas with different colors
    ax1.fill_between(enso.time, enso.values, where=enso.values >= 0, color='lightcoral', alpha=0.5)
    ax1.fill_between(enso.time, enso.values, where=enso.values < 0, color='lightblue', alpha=0.5)
    
    # === Plot rolling cumulative rainfall ===
    ax2.plot(rainfall_sum.time, rainfall_sum.values,
             linestyle='-', label=f'{window} {time_unit.capitalize()} Rolling Cumulative Sum of rainfall', alpha=0.5, color='#b3cde0', zorder=2)
    
    ax2.plot(rainfall_sum_above_threshold1.time, rainfall_sum_above_threshold1.values,
             linestyle='-', label=f'Peak above {threshold1} mm', alpha=0.5, color='#011f4b', zorder=3)
    
    ax2.plot(rainfall_sum_above_threshold2.time, rainfall_sum_above_threshold2.values,
             linestyle='-', label=f'Peak above {threshold2} mm', alpha=0.5, color='maroon', zorder=4)
    
    
    # === Plot full time series of lake size with event labels ===    
    ax3.plot(dea.time.values, dea.values,
                 color='grey', linestyle='', marker='o', markersize=8, alpha=0.4, zorder=1)
    
    plot_dea_events(ds, ax3, zorder_plot=5, zorder_text=10)
    
    
    # === Add a horizontal line at threshold levels ===
    ax2.axhline(y=threshold1, color='#03396c', linestyle='--')
    ax2.axhline(y=threshold2, color='maroon', linestyle='--')
    
    # === Axis Formatting ===
    ax1.set_ylabel(enso_index_name, fontsize=20)
    ax2.set_ylabel('128 day Cumulative Precipitation (mm/128 days)', fontsize=20)
    ax3.set_ylabel('Lake Pixel Count', fontsize=20)
    
    # Set y-ticks and y-tick labels for enso
    ax1.set_yticks(np.arange(-3, 3.1, 1))
    ax1.set_yticklabels([f"{int(x)}" for x in np.arange(-3, 3.1, 1)], fontsize=20)
    
    # Set y-ticks and y-tick labels for dea
    ax3.set_yticks(np.arange(-750,751,250))
    ax3.set_yticklabels(list(np.arange(-750,751,250)), fontsize=20)
    
    # Set y-ticks and y-tick labels for rain
    ax2.set_yticks(np.arange(-1500,1501,500))
    ax2.set_yticklabels(list(np.arange(-1500,1501,500)), fontsize=20)
    
    ax1.grid(True, which='major', axis='both') # Add grid to the background
    ax1.xaxis.set_major_locator(mdates.YearLocator(grid_spacing))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax1.get_xticklabels(), rotation=90, ha='center', fontsize=20)
    
    # === Save figure ===
    plt.savefig(f'{output_dir}/{Lake}/Timeseries/Drivers/{Lake}_FloodEvents_{window}{time_unit}_ENSO_{enso_index}_{start_date}.png', bbox_inches='tight')
    plt.close()
    

def plot_rolling_sum_with_ipo(ds, start_date, window, time_unit, threshold1, threshold2, ipo_index):
    if start_date=='1900':
        grid_spacing=10
    else:
        grid_spacing=1

    #Extract variables
    rainfall_sum = ds['mean_gridded_rainfall_rolling_sum']
    rainfall_sum_above_threshold1 = rainfall_sum.where(rainfall_sum[f'rolling_peak_above_{threshold1}'])
    rainfall_sum_above_threshold2 = rainfall_sum.where(rainfall_sum[f'rolling_peak_above_{threshold2}'])
    dea = ds['dea'].sel(lake_variable='Size')
    ipo=lake_daily_1900_ds['ipo'].sel(ipo_indices=ipo_index).sel(time=slice(f"{start_date}-01-01", "2025-01-01"))
    	
    ### PLOT: timeseries of 128 day cumulative rainfall with SOI ###
    fig = plt.figure(figsize=(40, 10))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 120))
    
    # Fill the positive and negative areas with different colors
    ax1.fill_between(ipo.time, ipo.values, where=ipo.values >= 0, color='lightcoral', alpha=0.5)
    ax1.fill_between(ipo.time, ipo.values, where=ipo.values < 0, color='lightblue', alpha=0.5)
    
    # === Plot rolling cumulative rainfall ===
    ax2.plot(rainfall_sum.time, rainfall_sum.values,
             linestyle='-', label=f'{window} {time_unit.capitalize()} Rolling Cumulative Sum of rainfall', alpha=0.5, color='#b3cde0', zorder=2)
    
    ax2.plot(rainfall_sum_above_threshold1.time, rainfall_sum_above_threshold1.values,
             linestyle='-', label=f'Peak above {threshold1} mm', alpha=0.5, color='#011f4b', zorder=3)
    
    ax2.plot(rainfall_sum_above_threshold2.time, rainfall_sum_above_threshold2.values,
             linestyle='-', label=f'Peak above {threshold2} mm', alpha=0.5, color='maroon', zorder=4)
    
    
    # === Plot full time series of lake size with event labels ===    
    ax3.plot(dea.time.values, dea.values,
                 color='grey', linestyle='', marker='o', markersize=8, alpha=0.4, zorder=1)
    
    plot_dea_events(ds, ax3, zorder_plot=5, zorder_text=10)
    
    
    # === Add a horizontal line at threshold levels ===
    ax2.axhline(y=threshold1, color='#03396c', linestyle='--')
    ax2.axhline(y=threshold2, color='maroon', linestyle='--')
    
    # === Axis Formatting ===  	
    ax3.set_ylabel('Lake Pixel Count', fontsize=20)
    ax2.set_ylabel('128 day Cumulative Precipitation (mm/128 days)', fontsize=20)
    ax1.set_ylabel(f'IPO ({ipo_index})', fontsize=20)
    
    # Set y-ticks and y-tick labels for ipo
    values = np.arange(-0.6, 0.7, 0.2)
    labels = [f"{v:.1f}" for v in values]
    ax1.set_yticks(values)
    ax1.set_yticklabels(labels, fontsize=20)
    
    # Set y-ticks and y-tick labels for dea
    ax3.set_yticks(np.arange(-750,751,250))
    ax3.set_yticklabels(list(np.arange(-750,751,250)), fontsize=20)
    
    # Set y-ticks and y-tick labels for rain
    ax2.set_yticks(np.arange(-1500,1501,500))
    ax2.set_yticklabels(list(np.arange(-1500,1501,500)), fontsize=20)
    
    ax1.grid(True, which='major', axis='both') # Add grid to the background
    ax1.xaxis.set_major_locator(mdates.YearLocator(grid_spacing))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax1.get_xticklabels(), rotation=90, ha='center', fontsize=20)
    
    # === Save figure ===
    plt.savefig(f'{output_dir}/{Lake}/Timeseries/Drivers/{Lake}_FloodEvents_{window}{time_unit}_IPO_{ipo_index}_{start_date}.png', bbox_inches='tight')
    plt.close() 



#### ========= Functions: Plot formattting =========
def plot_yearly_xticks(axis_no):
    """
    Set x-axis ticks to yearly intervals with rotated labels.

    Parameters:
        axis_no (matplotlib.axes.Axes): The axis to apply the tick formatting.
    """
    axis_no.xaxis.set_major_locator(mdates.YearLocator())
    axis_no.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(axis_no.get_xticklabels(), rotation=90, ha='center', fontsize=20)
    axis_no.tick_params(axis='y', labelsize=20)


def plot_weekly_xticks(axis_no):
    """
    Set x-axis ticks to weekly intervals with rotated date labels.

    Parameters:
        axis_no (matplotlib.axes.Axes): The axis to apply the tick formatting.
    """
    axis_no.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    axis_no.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axis_no.tick_params(axis='both', labelsize=15)
    for label in axis_no.get_xticklabels():
        label.set_rotation(90)


def plot_offset_xticks(da, axis_no):
    """
    Plot offset-based x-axis ticks with alternating grid and label layout.

    Parameters:
        da (xr.DataArray): The offset data array used for tick limits.
        axis_no (matplotlib.axes.Axes): The axis to format.
    """
    minor_xticks = np.arange(da.min(skipna=True), da.max(skipna=True) + 1, 16)
    major_xticks = minor_xticks + 8
    xtick_labels = [f"{int(x)}" for x in minor_xticks]

    axis_no.set_xticks(minor_xticks, minor=True)
    axis_no.set_xticks(major_xticks, xtick_labels, fontsize=15)
    axis_no.tick_params(axis='y', labelsize=15)

    axis_no.xaxis.grid(which='minor', alpha=1, linestyle='--', linewidth=1)
    axis_no.xaxis.grid(which='major', alpha=0)
    axis_no.yaxis.grid(True, linestyle='--', alpha=0.3)
    axis_no.tick_params(axis='x', which='major', length=0)


def truncate_colormap(cmap, minval=0.2, maxval=1.0, n=256):
    """
    Truncate a colormap to avoid light or dark extremes.

    Parameters:
        cmap (Colormap): A matplotlib colormap instance.
        minval (float): Lower limit (0 to 1).
        maxval (float): Upper limit (0 to 1).
        n (int): Number of discrete color levels in the new colormap.

    Returns:
        LinearSegmentedColormap: A truncated version of the input colormap.
    """
    new_cmap = LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

       
#%% Load Data

# Load Masks
Lake_mask_r001_file = f'AGCD/{Lake}/{Lake}_mask_r001.nc'
Lake_mask_r001_netcdf = xr.open_dataset(f'{input_dir}/{Lake_mask_r001_file}')
Lake_mask_r001 = Lake_mask_r001_netcdf['Mask']

Lake_mask_r005_file = f'AGCD/{Lake}/{Lake}_mask_r005.nc'
Lake_mask_r005_netcdf = xr.open_dataset(f'{input_dir}/{Lake_mask_r005_file}')
Lake_mask_r005 = Lake_mask_r005_netcdf['Mask']


#-----------------------------------------------------------------------------------------------------------------------------------
### Load Gridded Daily
agcd_daily_file = f'AGCD/{Lake}/agcd_v1_precip_total_r005_daily_{Lake}_1900to2024.nc'
agcd_daily_ds = xr.open_dataset(f'{input_dir}/{agcd_daily_file}')
agcd_daily_ds['time'] = agcd_daily_ds.indexes['time'].normalize()

# Catchment rainfall
agcd_daily_ds = agcd_daily_ds.rename({'precip': 'gridded_rainfall'})
agcd_daily_ds['gridded_rainfall'] = agcd_daily_ds['gridded_rainfall'].assign_coords(mask=(('lat', 'lon'), Lake_mask_r005.data))
agcd_daily_ds['gridded_rainfall'] = agcd_daily_ds['gridded_rainfall'].where(agcd_daily_ds['gridded_rainfall'].mask == 1)
agcd_daily_ds['gridded_rainfall'].attrs.update({
    'long_name': 'Daily gridded catchment rainfall',
    'description': f'AGCD daily gridded rainfall masked to the catchment area of {Lake}.',
    "source": "AGCD"
})

#mean
agcd_daily_ds['mean_gridded_rainfall'] = agcd_daily_ds['gridded_rainfall'].mean(dim=('lon', 'lat'))
agcd_daily_ds['mean_gridded_rainfall'].attrs = {
    'units': agcd_daily_ds['gridded_rainfall'].attrs.get('units', ''),
    'long_name': 'Daily mean catchment rainfall',
    'description': f'AGCD Spatial average of daily gridded rainfall masked to the catchment area of {Lake}.',
    "source": "AGCD"
}

lake_daily_1900_ds = agcd_daily_ds.sel(time=slice("1900-01-01", agcd_daily_ds.time[-1]))


#Load Runoff
if Lake != 'LB':
    
    # Read runoff stations data from the netcdf file
    runoff_stations_file = f'{input_dir}/runoff_stations/{Lake}_runoff_stations/{Lake}_daily_station_runoff.nc'
    runoff_stations_ds = xr.open_dataset(runoff_stations_file)
    
    runoff_stations_da = runoff_stations_ds['runoff']
    runoff_stations_da.name = 'station_runoff'  # optional rename
    lake_daily_1900_ds['station_runoff'] = runoff_stations_da
    lake_daily_1900_ds['station_runoff'].attrs.update({
        'long_name': 'Daily runoff at stations in catchment area',
        'description': f'Daily runoff observations from gauge stations located within the catchment area of {Lake}..',
    })

    # Filter runoff stations
    lake_daily_1900_ds = filter_out_empty_stations(lake_daily_1900_ds, 'runoff')


# Load rainfall stations
rainfall_stations_file = f'{input_dir}/rainfall_stations/{Lake}_rainfall_stations/{Lake}_daily_station_rainfall.nc'
rainfall_stations_ds = xr.open_dataset(rainfall_stations_file)
   
rainfall_stations_da = rainfall_stations_ds['rainfall']
rainfall_stations_da.name = 'station_rainfall'
lake_daily_1900_ds['station_rainfall'] = rainfall_stations_da
lake_daily_1900_ds['station_rainfall'].attrs.update({
    'long_name': 'Daily rainfall at stations in catchment area',
    'description': f'Daily rainfall observations from gauge stations located within the catchment area of {Lake}..',
})

# Filter rainfall stations
lake_daily_1900_ds = filter_out_empty_stations(lake_daily_1900_ds, 'rainfall')


### Load NAus
agcd_NT_daily_file = 'AGCD/NT/agcd_v1_precip_total_r005_daily_NAus_1900to2024.nc'
agcd_NT_daily_ds = xr.open_dataset(f'{input_dir}/{agcd_NT_daily_file}')
agcd_NT_daily_ds['time'] = agcd_NT_daily_ds.indexes['time'].normalize()
agcd_NT_daily_da = agcd_NT_daily_ds['precip']


# Load DEA
dea_file = f'{input_dir}/DEA/{Lake}_dea/{Lake}_dea.nc'
dea_ds = xr.open_dataset(dea_file)

dea_da = dea_ds['lake_observations']
dea_da.name = 'dea'  # optional rename
lake_daily_1900_ds['dea'] = dea_da
lake_daily_1900_ds['dea'].attrs['long_name'] = "Daily lake extent and surface area"


#%% Load CESM Data
cesm_file_paths = sorted(glob.glob(f'{input_dir}/CESM_CAM5_LME/precip/b.e11.BLMTRC5CN.f19_g16*.nc'))

cesm_ensemble_list = []
for path in cesm_file_paths:
    cesm_ds = xr.open_dataset(path, decode_times=True)
    cesm_ds = cesm_ds.assign_coords(time=cesm_ds['time_bnds'].isel(nbnd=0))
    cesm_da = cesm_ds.precip
    cesm_lw_da = cesm_da.sel(lat=slice(-18,-16)).sel(lon=slice(132,135))
    cesm_lw_mean_da = cesm_lw_da.mean(dim=('lon', 'lat'))
    cesm_lw_mean_da = cesm_lw_mean_da * 1000 * 86400 * cesm_lw_mean_da['time'].dt.days_in_month
    cesm_lw_mean_da.attrs['units'] = 'mm/month'
    cesm_ensemble_list.append(cesm_lw_mean_da)

cesm_precip = xr.concat(cesm_ensemble_list, dim='ensemble', join='outer')
cesm_precip = cesm_precip.assign_coords(ensemble=range(1, 14))




#%% Load Data: Drivers - ENSO & IPO 

# Read in ENSO data
enso_file_paths = glob.glob(f'{input_dir}/ClimaticDrivers_indices/enso_*.nc')

enso_indices_list = []
for path in enso_file_paths:
    enso_ds = xr.open_dataset(path)
    var_name = list(enso_ds.data_vars)
    enso_da = enso_ds[var_name[0]]    
    enso_indices_list.append(enso_da)

enso_indices_names = [da.name if da.name is not None else f"da{i+1}" for i, da in enumerate(enso_indices_list)]
ENSO_noaa = xr.concat(enso_indices_list, dim='enso_indices', join='outer')
ENSO_noaa = ENSO_noaa.assign_coords(enso_indices=np.array(enso_indices_names, dtype=object))
ENSO_noaa = ENSO_noaa.rename('enso')
ENSO_noaa.attrs = {
    "long_name": "Monthly ENSO indices",
    "description": (
        "Combined ENSO dataset containing multiple sea surface temperature anomaly indices along an 'enso_indices' dimension. "
        "Includes Niño 3.4 (ERSSTv5), Niño 3.4 (HadISST), and the Oceanic Niño Index (ONI) from NOAA CPC."
    ),
    "source": "NOAA",
    "units": "°C",
    "url": "https://psl.noaa.gov/data/timeseries/month/",
}

#monthly
ENSO_noaa_monthly_1900_da = ENSO_noaa.sel(time=slice(lake_daily_1900_ds.time.min(),lake_daily_1900_ds.time.max()))

# daily
extra_time = pd.Timestamp(ENSO_noaa_monthly_1900_da.time.max().values) + pd.DateOffset(months=1)
ENSO_noaa_monthly_1900_da_extended = ENSO_noaa_monthly_1900_da.reindex(time=list(ENSO_noaa_monthly_1900_da.time.values) + [np.datetime64(extra_time)])
ENSO_noaa_daily_1900_da = ENSO_noaa_monthly_1900_da_extended.resample(time='1D').ffill()
ENSO_noaa_daily_1900_da = ENSO_noaa_daily_1900_da.isel(time=slice(0, -1))
lake_daily_1900_ds['enso'] = ENSO_noaa_daily_1900_da
   
#-----------------------------------------------------------------------------------------------------------------------------------
# Read in IPO data
IPOtripole_noaa = read_ipo_file()
IPOtripole_noaa_monthly_1900_da = IPOtripole_noaa.sel(time=slice(lake_daily_1900_ds.time.min(),lake_daily_1900_ds.time.max()))

# daily
extra_time = pd.Timestamp(IPOtripole_noaa_monthly_1900_da.time.max().values) + pd.DateOffset(months=1)
IPOtripole_noaa_monthly_1900_da_extended = IPOtripole_noaa_monthly_1900_da.reindex(time=list(IPOtripole_noaa_monthly_1900_da.time.values) + [np.datetime64(extra_time)])
IPOtripole_noaa_daily_1900_da = IPOtripole_noaa_monthly_1900_da_extended.resample(time='1D').ffill()
IPOtripole_noaa_daily_1900_da = IPOtripole_noaa_daily_1900_da.isel(time=slice(0, -1))
lake_daily_1900_ds['ipo'] = IPOtripole_noaa_daily_1900_da


#%% Analyse Data: Identify Filling Events from DEA

# - Filling events are defined as distinct peaks in lake size, separated by periods where the lake is considered "empty".
# - The 10% threshold was chosen to eliminate small dry-season fluctuations and ensure only substantial filling episodes are captured.
# - Events are only counted if they span more than a single timestep (i.e., not isolated noise).
# - The timestep *before* the event onset is included, to capture full filling dynamics.

# Step 0: Extract the lake size time series and remove missing values (NaNs)
dea_Lake_Size_da = lake_daily_1900_ds['dea'].sel(lake_variable='Size').dropna(dim='time')

# Step 1: Set threshold at 10% of max lake size to define "empty" conditions
# This threshold ensures we only count meaningful wet periods, not minor fluctuations.
threshold = dea_Lake_Size_da.max().item() / 10 #ca.70th percentile

# Step 2: Create a boolean mask of times when the lake is "not empty"
is_event = dea_Lake_Size_da > threshold  # True = part of an event, False = "empty"

# Step 3: Convert the mask to a NumPy boolean array for easier manipulation
event_mask = is_event.values.astype(bool)

# Step 4: Detect event start points by checking for transitions from 0 → 1 (False → True)
group_change = np.diff(event_mask.astype(int), prepend=0)

# Step 5: Assign a preliminary group ID to each event using cumulative sum of event starts
# Example: event_mask → [False, False, True, True, False, True, True]
#           group_id → [ nan ,  nan ,   1 ,   1 ,  nan ,   2 ,   2 ]
group_id = np.cumsum((group_change == 1) & event_mask).astype(float)

# Step 5b: Set non-event (i.e., "empty") days to NaN
group_id[~event_mask] = np.nan

# Step 5c: Remove "events" that consist of only one timestep (likely noise)
counts = Counter(group_id[~np.isnan(group_id)])
singleton_ids = {gid for gid, count in counts.items() if count == 1}
for gid in singleton_ids:
    group_id[group_id == gid] = np.nan

# Step 5d: Reindex remaining group IDs to be consecutive starting from 1
# (e.g., if event 2 and 4 remain after filtering, they become 1 and 2)
unique_ids = np.unique(group_id[~np.isnan(group_id)])
id_map = {old: new for new, old in enumerate(unique_ids, start=1)}
group_id = np.array([id_map[val] if val in id_map else np.nan for val in group_id])

# Step 5e: Include the timestep *before* each event start
# This is important to capture the full rise of each event and enable magnitude calculations.
starts = np.where((group_change == 1) & event_mask)[0]
for start in starts:
    if start > 0:
        group_id[start - 1] = group_id[start]

# Step 6: Convert to xarray DataArray, restoring coordinates
event_id_da = xr.DataArray(group_id, coords=dea_Lake_Size_da.coords, dims=dea_Lake_Size_da.dims)
event_id_1900_da = event_id_da.reindex(time=lake_daily_1900_ds['time'])

# Step 7: Fill gaps resulting from re-indexing (mask=forward/backfill agree → interior gaps only)
ff = event_id_1900_da.ffill(dim='time') # forward-fill the reindexed array
bf = event_id_1900_da.bfill(dim='time') # fback-fill the reindexed arr
mask_enclosed = ff.notnull() & (ff == bf)
event_id_1900_filled_da = event_id_1900_da.where(~mask_enclosed, other=ff) # replace enclosed NaNs with the agreed value (ff)

# Step 8: Add to dataset
assert event_id_1900_filled_da.sizes['time'] == lake_daily_1900_ds.sizes['time']
assert np.all(event_id_1900_filled_da.time.values == lake_daily_1900_ds.time.values)
lake_daily_1900_ds['dea'] = lake_daily_1900_ds['dea'].assign_coords(dea_events=("time", event_id_1900_filled_da.data))
lake_daily_1900_ds["dea"].coords["dea_events"].attrs["long_name"] = "DEA event number"

# Step 9: Identify maximum lake size of each event
lake_daily_1900_ds = lake_daily_1900_ds.assign_coords(dea_event_max=find_lake_event_max_coord(lake_daily_1900_ds))

# Step 10: Resample ds to monthly
lake_monthly_1900_ds = resample_lake_dataset_to_monthly(lake_daily_1900_ds)


### ========= Calculate offset =========
lake_daily_1900_ds['dea'] = lake_daily_1900_ds['dea'].assign_coords(dea_offset=calculate_event_offset_coord(lake_daily_1900_ds, 'daily'))
lake_monthly_1900_ds['dea'] = lake_monthly_1900_ds['dea'].assign_coords(dea_offset=calculate_event_offset_coord(lake_monthly_1900_ds, 'monthly'))


####  === Calculate cumulative rainfall per event ===
# daily gridded
lake_daily_1900_ds['mean_gridded_rainfall_event_cum'] = cumulative_rainfall_per_event(lake_daily_1900_ds, 'mean_gridded_rainfall')
lake_daily_1900_ds['mean_gridded_rainfall_event_cum'].attrs.update({
    'long_name': f'Daily cumulative {Lake} catchment rainfall during events',
    'description': (
        f'Daily cumulative {Lake} catchment rainfall, starting from the first day of each identified event up to the day of peak lake size (maximum DEA). '
        'The coordinates can be used to identify event number, peak day, and offset from the peak.'
    ),
    'units': 'mm'})

#daily station
lake_daily_1900_ds['station_rainfall_event_cum'] = cumulative_rainfall_per_event(lake_daily_1900_ds, 'station_rainfall')
lake_daily_1900_ds['station_rainfall_event_cum'].attrs.update({
    'long_name': 'Daily cumulative station rainfall during events',
    'description': (
        'Daily cumulative station based rainfall, starting from the first day of each identified event up to the day of peak lake size (maximum DEA). '
        'The coordinates can be used to identify event number, peak day, and offset from the peak.'
    ),
    'units': 'mm'})

# monthly gridded
lake_monthly_1900_ds['mean_gridded_rainfall_event_cum'] = cumulative_rainfall_per_event(lake_monthly_1900_ds, 'mean_gridded_rainfall')
lake_monthly_1900_ds['mean_gridded_rainfall_event_cum'].attrs.update({
    'long_name': f'Monthly cumulative {Lake} catchment rainfall during events',
    'description': (
        f'Monthly cumulative {Lake} catchment rainfall, starting from the first day of each identified event up to the day of peak lake size (maximum DEA). '
        'The coordinates can be used to identify event number, peak day, and offset from the peak.'
    ),
    'units': 'mm'})

#monthly station
lake_monthly_1900_ds['station_rainfall_event_cum'] = cumulative_rainfall_per_event(lake_monthly_1900_ds, 'station_rainfall')
lake_monthly_1900_ds['station_rainfall_event_cum'].attrs.update({
    'long_name': 'Monthly cumulative station rainfall during events',
    'description': (
        'Monthly cumulative station based rainfall, starting from the first day of each identified event up to the day of peak lake size (maximum DEA). '
        'The coordinates can be used to identify event number, peak day, and offset from the peak.'
    ),
    'units': 'mm'})


####  === Filter Event Cumulative Precip by Rate of Change ===
lake_daily_1900_ds['mean_gridded_rainfall_event_cum_filtered'] = filter_cumulative_rainfall(lake_daily_1900_ds)
lake_monthly_1900_ds['mean_gridded_rainfall_event_cum_filtered'] = filter_cumulative_rainfall(lake_monthly_1900_ds)
    

#%% Analyse Data: Monthly rolling cumulative dataset

#Define variables
window_size_monthly = 4  # Define the rolling window size in months
lower_threshold_monthly = 600 # Define lower threshold
higher_threshold_monthly = 800 # Define higher threshold

## Loop through each segment defined by two troughs
# Check if the segment of the rolling sum is above the lower or higher threshold
# For each segment, find peak magnitude and examine the window leading up to it (is it an event/ is it above lower threshold)
peak_segments_dict = extract_peak_segments(lake_monthly_1900_ds, 6, 'months', window_size_monthly, lower_threshold_monthly, higher_threshold_monthly)

# Rolling sum
lake_monthly_1900_ds['mean_gridded_rainfall_rolling_sum'] = peak_segments_dict['rolling_sum']
lake_monthly_1900_ds['mean_gridded_rainfall_rolling_sum'].attrs.update({
    'long_name': f'{window_size_monthly}-month rolling sum of mean gridded rainfall',
    'description': (
        f'Rolling {window_size_monthly}-month sum of rainfall from gridded dataset over the {Lake} catchment area. '
        f'The coordinates can be used to identify if the peaks of the rolling sum exceed the defined thresholds ({lower_threshold_monthly}/ {higher_threshold_monthly}) mm.'),
    'units': 'mm'
})


# Cumulative rainfall to peak
lake_monthly_1900_ds['mean_gridded_rainfall_cum_to_peak'] = peak_segments_dict[f'rainfall_cum_{window_size_monthly}_months']

# Then attach boolean coordinates separately
classified_da = classify_rainfall_windows_by_event_and_threshold(
    lake_monthly_1900_ds, peak_segments_dict[f'rainfall_cum_{window_size_monthly}_months'], 
    window_size_monthly, 'months', lower_threshold_monthly)

lake_monthly_1900_ds['mean_gridded_rainfall_cum_to_peak'] = lake_monthly_1900_ds['mean_gridded_rainfall_cum_to_peak'].assign_coords(
    cum_window_is_event=classified_da.cum_window_is_event.reindex(time=lake_monthly_1900_ds.time, fill_value=np.nan),
    cum_window_above_600=classified_da.cum_window_above_600.reindex(time=lake_monthly_1900_ds.time, fill_value=False))

lake_monthly_1900_ds['mean_gridded_rainfall_cum_to_peak'].attrs.update({
    'long_name': f'Cumulative rainfall up to peaks of ({window_size_monthly}-month windows)',
    'description': (
        f'Cumulative rainfall - using a {window_size_monthly}-month window up to peaks identified in rolling sum '
        f'- from rainfall spatially averaged over {Lake} catchment. '
        f'The coordinates can be used to identify if the cumulative rainfall segments occur during an event '
        f'and/or exceed a threshold of {lower_threshold_monthly} mm.'),
    'units': 'mm'
})


#%% Analyse Data: Daily rolling cumulative dataset

#Define variables
window_size_daily = 112  # Define the rolling window size in days
lower_threshold_daily = 600 # Define lower threshold
higher_threshold_daily = 800 # Define higher threshold

## Loop through each segment defined by two troughs
# Check if the segment of the rolling sum is above the lower or higher threshold
# For each segment, find peak magnitude and examine the window leading up to it (is it an event/ is it above lower threshold)
peak_segments_dict = extract_peak_segments(lake_daily_1900_ds, 180, 'days', window_size_daily, lower_threshold_daily, higher_threshold_daily)

# Rolling sum
lake_daily_1900_ds['mean_gridded_rainfall_rolling_sum'] = peak_segments_dict['rolling_sum']
lake_daily_1900_ds['mean_gridded_rainfall_rolling_sum'].attrs.update({
    'long_name': f'{window_size_daily}-day rolling sum of mean gridded rainfall',
    'description': (
        f'Rolling {window_size_daily}-day sum of rainfall from gridded dataset over the {Lake} catchment area. '
        f'The coordinates can be used to identify if the peaks of the rolling sum exceed the defined thresholds ({lower_threshold_daily}/ {higher_threshold_daily}) mm.'),
    'units': 'mm'
})


# Cumulative rainfall to peak
lake_daily_1900_ds['mean_gridded_rainfall_cum_to_peak'] = peak_segments_dict[f'rainfall_cum_{window_size_daily}_days']

# Then attach boolean coordinates separately
classified_da = classify_rainfall_windows_by_event_and_threshold(
    lake_daily_1900_ds, peak_segments_dict[f'rainfall_cum_{window_size_daily}_days'],
    window_size_daily, 'days', lower_threshold_daily)
lake_daily_1900_ds['mean_gridded_rainfall_cum_to_peak'] = lake_daily_1900_ds['mean_gridded_rainfall_cum_to_peak'].assign_coords(
    cum_window_is_event=classified_da.cum_window_is_event.reindex(time=lake_daily_1900_ds.time, fill_value=np.nan),
    cum_window_above_600=classified_da.cum_window_above_600.reindex(time=lake_daily_1900_ds.time, fill_value=False))

lake_daily_1900_ds['mean_gridded_rainfall_cum_to_peak'].attrs.update({
    'long_name': f'Cumulative rainfall up to peaks of ({window_size_daily}-day windows)',
    'description': (
        f'Cumulative rainfall - using a {window_size_daily}-day window up to peaks identified in rolling sum '
        f'- from rainfall spatially averaged over {Lake} catchment. '
        f'The coordinates can be used to identify if the cumulative rainfall segments occur during an event '
        f'and/or exceed a threshold of {lower_threshold_daily} mm.'),
    'units': 'mm'
})


#%% Analyse Data: Cut to 1987

# Cut to 1987
lake_daily_1987_ds = lake_daily_1900_ds.sel(time=slice("1987-01-01", lake_daily_1900_ds.time[-1]))
lake_daily_1987_ds = filter_out_empty_stations(lake_daily_1987_ds, 'rainfall')
lake_daily_1987_ds = filter_out_empty_stations(lake_daily_1987_ds, 'runoff')

lake_monthly_1987_ds = lake_monthly_1900_ds.sel(time=slice("1987-01-01", lake_monthly_1900_ds.time[-1]))
lake_monthly_1987_ds = filter_out_empty_stations(lake_monthly_1987_ds, 'rainfall')
lake_monthly_1987_ds = filter_out_empty_stations(lake_monthly_1987_ds, 'runoff')


#%% 1. Map Plot: 1.1 EA map catchment area

# Create a new figure with specified size and add a subplot with PlateCarree projection and coastlines
fig = plt.figure(figsize=(40, 40))
ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax1.coastlines()  # Add coastlines

# Set longitude and latitude limits (xmin, xmax, ymin, ymax)
ax1.set_extent([130, 155, -40, -10], crs=ccrs.PlateCarree())

# Add land feature to the map
land = cfeature.NaturalEarthFeature(category='physical', name='land', scale='50m', facecolor='white')
ax1.add_feature(land, edgecolor='black')

# Duplicate each value 10 times in each direction for higher resolution
Z = np.repeat(np.repeat(Lake_mask_r005.values, 10, axis=0), 10, axis=1)
X = np.linspace(Lake_mask_r005.coords['lon'].values[0], Lake_mask_r005.coords['lon'].values[-1], Z.shape[1])
Y = np.linspace(Lake_mask_r005.coords['lat'].values[0], Lake_mask_r005.coords['lat'].values[-1], Z.shape[0])
    
# Fill lake mask
ax1.contourf(Lake_mask_r005.coords['lon'].values, Lake_mask_r005.coords['lat'].values, Lake_mask_r005.values, cmap=ListedColormap(['white', 'white']), transform=ccrs.PlateCarree(), zorder=2)

# Plot lake masks as black contour lines
ax1.contour(X, Y, Z, colors='k', linewidths=0.75, transform=ccrs.PlateCarree())

# Set tick positions for latitude and longitude on the plot
#ax1.set_xticks([130, 135, 140, 145, 150, 155], crs=ccrs.PlateCarree())  # Add latitude and longitude values
#ax1.set_yticks([-40, -35, -30, -25, -20, -15, -10], crs=ccrs.PlateCarree())

# Set labels for x and y axes
#ax1.tick_params(axis='both', which='major', labelsize=15)
#ax1.set_xlabel('longitude', fontsize=20)
#ax1.set_ylabel('latitude', fontsize=20)

# Add gridlines
gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
              linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlocator = LongitudeLocator()
gl.ylocator = LatitudeLocator()
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 30, 'color': 'black'}
gl.ylabel_style = {'size': 30, 'color': 'black'}


# Save the plot as a PDF file with specified output directory and close the plot
plt.savefig(f'{output_dir}/{Lake}/Maps/EA_map_{Lake}.png', bbox_inches='tight')
plt.close()



#%% 1. Map Plot: 1.2 Catchment map with all stations

# Create a new figure with specified size and add a subplot with PlateCarree projection and coastlines
fig = plt.figure(figsize=(15, 17))
ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax1.coastlines()  # Add coastlines
ax1.set_facecolor(presentation_bom_colours[4])

# Fill lake mask
ax1.pcolormesh(Lake_mask_r005.coords['lon'].values, Lake_mask_r005.coords['lat'].values, Lake_mask_r005.values, cmap=ListedColormap(['white', 'white']), transform=ccrs.PlateCarree(), zorder=2)

# Plot lake masks as black contour lines
ax1.contour(X, Y, Z, colors='k', linewidths=1, transform=ccrs.PlateCarree())

# Extract station lat/lon and IDs
rainfall_lats = lake_daily_1987_ds['rainfall_station_lat'].values
rainfall_lons = lake_daily_1987_ds['rainfall_station_lon'].values
rainfall_ids = lake_daily_1987_ds['rainfall_station'].values

runoff_lats = lake_daily_1987_ds['runoff_station_lat'].values
runoff_lons = lake_daily_1987_ds['runoff_station_lon'].values
runoff_ids = lake_daily_1987_ds['runoff_station'].values

# Plot rainfall stations (green markers)
ax1.scatter(rainfall_lons, rainfall_lats, marker='X', s=150, color=presentation_bom_colours[1],
            transform=ccrs.PlateCarree(), zorder=3)

# Add rainfall station IDs as text labels
for lon, lat, sid in zip(rainfall_lons, rainfall_lats, rainfall_ids):
    ax1.text(lon - 0.16, lat + 0.02, sid, fontsize=15, color=presentation_bom_colours[1], transform=ccrs.PlateCarree(), zorder=4)

# Plot runoff stations (magenta markers)
ax1.scatter(runoff_lons, runoff_lats, marker='X', s=150, color=presentation_bom_colours[2],
            transform=ccrs.PlateCarree(), zorder=3)

# Add runoff station IDs as text labels
for lon, lat, sid in zip(runoff_lons, runoff_lats, runoff_ids):
    ax1.text(lon + 0.02, lat + 0.02, sid, fontsize=15, color=presentation_bom_colours[2], transform=ccrs.PlateCarree(), zorder=4)

# Set labels for x and y axes
#ax1.set_xlabel('longitude', fontsize=20)
#ax1.set_ylabel('latitude', fontsize=20)

# Add gridlines with labels
gl = ax1.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                   linewidth=0.5, color='white', alpha=0.7, linestyle='--')

# Show only left and bottom labels
gl.top_labels = False
gl.right_labels = False

# Control label font size
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

# =======================
# Add a textbox with station names
# =======================

# Combine station labels

rainfall_info = [
    f"{sid} – {sname}"
    for sid, sname in zip(lake_daily_1987_ds['rainfall_station'].values,
                          lake_daily_1987_ds['rainfall_station_name'].values)
]
runoff_info = [
    f"{sid} – {sname}"
    for sid, sname in zip(lake_daily_1987_ds['runoff_station'].values,
                          lake_daily_1987_ds['runoff_station_name'].values)
]

# Create text block
textbox_content = "Rainfall Stations:\n" + "\n".join(rainfall_info)
textbox_content += "\n\nRunoff Stations:\n" + "\n".join(runoff_info)

# Add text box to top left
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black')
ax1.text(
    0.55, 0.2, textbox_content,
    transform=ax1.transAxes,
    fontsize=15,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=props,
    zorder=5
)

# Save the plot as a PDF file and close the plot
plt.savefig(f'{output_dir}/{Lake}/Maps/LW_map_stations.pdf', bbox_inches='tight')
plt.close()


#%% 1. Map Plot: 1.3 Rainfall Map 

# r001
#fig = plt.figure(figsize=(len(gridded_monthly_timeslice_lake.coords['lon'])/5, len(gridded_monthly_timeslice_lake.coords['lat'])/5))
#ax1 = fig.add_subplot(1, 1, 1)
#im0 = ax1.pcolormesh(gridded_monthly_timeslice_lake.coords['lon'].values, gridded_monthly_timeslice_lake.coords['lat'].values, gridded_monthly_timeslice_lake.mean(dim=('time'), skipna=True), cmap = 'Blues')
#cb1 = fig.colorbar(im0, fraction=0.025, pad=0.09,  ax=ax1, orientation='horizontal')
#cb1.set_label('rainfall (mm/day)')
#cb1.outline.set_color('lightgray')
#plt.savefig(f'{output_dir}/{Lake}/gridded_rainfall_{Lake}_map_r001.pdf', bbox_inches='tight')
#plt.close()


# r005
fig = plt.figure(figsize=(len(lake_daily_1987_ds['gridded_rainfall'].coords['lon']), len(lake_daily_1987_ds['gridded_rainfall'].coords['lat'])))
ax1 = fig.add_subplot(1, 1, 1)

# Plot the average gridded_rainfall
im0 = ax1.pcolormesh(
    lake_daily_1987_ds['gridded_rainfall'].coords['lon'].values,
    lake_daily_1987_ds['gridded_rainfall'].coords['lat'].values,
    lake_daily_1987_ds['gridded_rainfall'].mean(dim='time', skipna=True),
    cmap='Blues'
)

# Colorbar
cb1 = fig.colorbar(im0, fraction=0.025, pad=0.03, ax=ax1, orientation='horizontal')
cb1.set_label(f'gridded_rainfall ({lake_daily_1987_ds['gridded_rainfall'].units}/day)', fontsize=35)
cb1.outline.set_color('lightgray')

# Add gridlines
ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

# Enlarge tick labels
ax1.tick_params(axis='both', labelsize=30)

# Save figure
plt.savefig(f'{output_dir}/{Lake}/Maps/{Lake}_map_r005_rainfall.pdf', bbox_inches='tight')
plt.close()


#%% 2. Timeseries Plot: 2.1 Runoff & Rainfall 

### Runoff
if Lake != 'LB':
    plot_timeseries('daily', 'runoff', mean_data=None, station_data=lake_daily_1987_ds['station_runoff'])


### Rainfall
# Plot daily
plot_timeseries('daily', 'rainfall', plot_type='dots', mean_data=lake_daily_1987_ds['mean_gridded_rainfall'], station_data=lake_daily_1987_ds['station_rainfall'])

# Plot monthly
plot_timeseries('monthly', 'rainfall', mean_data=lake_monthly_1987_ds['mean_gridded_rainfall'], station_data=lake_daily_1987_ds['station_rainfall'])

# Plot monthly no stations
plot_timeseries('monthly', 'rainfall', mean_data=lake_monthly_1987_ds['mean_gridded_rainfall'])


#%% 2. Timeseries Plot: 2.2 DEA Flood Events (optional: with Rainfall / Runoff)

####  Plot DEA Events  
plot_dea_timeseries(lake_daily_1987_ds, time_unit_lake='daily', event_labels=False)
plot_dea_timeseries(lake_daily_1987_ds, time_unit_lake='daily')    
plot_dea_timeseries(lake_daily_1987_ds, time_unit_lake='monthly')    

####  Plot Rainfall
plot_dea_timeseries(lake_daily_1987_ds, time_unit_lake='daily', timeseries_type='rainfall', time_unit_data='daily', plot_mean=True, plot_station=True)
plot_dea_timeseries(lake_daily_1987_ds, time_unit_lake='daily', timeseries_type='rainfall', time_unit_data='daily', plot_mean=True)
plot_dea_timeseries(lake_daily_1987_ds, time_unit_lake='daily', timeseries_type='rainfall', time_unit_data='monthly', plot_mean=True, plot_station=True)
plot_dea_timeseries(lake_daily_1987_ds, time_unit_lake='daily', timeseries_type='rainfall', time_unit_data='monthly', plot_mean=True)

####  Plot Runoff
plot_dea_timeseries(lake_daily_1987_ds, time_unit_lake='daily', timeseries_type='runoff', time_unit_data='monthly', plot_station=True)
plot_dea_timeseries(lake_daily_1987_ds, time_unit_lake='daily', timeseries_type='runoff', time_unit_data='daily', plot_station=True)

# Flood Events with Rainfall Peaks before Event Max
plot_dea_timeseries(lake_daily_1987_ds, time_unit_lake='monthly', timeseries_type='rainfall', time_unit_data='monthly', plot_mean=True, peaks=True)


#### Combined Daily Plot: Lake Size, rainfall & Event Labels ####
# Create a large figure with two y-axes (twin axis)
fig = plt.figure(figsize=(40, 10))
ax1 = fig.add_subplot(111)        # Left y-axis for rainfall
ax2 = ax1.twinx()                 # Right y-axis for lake size

# === Plot mean catchment rainfall ===
ax1.plot(lake_daily_1987_ds['mean_gridded_rainfall'].time.values, lake_daily_1987_ds['mean_gridded_rainfall'].values,
    linestyle='', marker='o', markersize=6, label='Mean rainfall over catchment area', color=presentation_bom_colours[3], zorder=2)

# === Plot rainfall from each individual station ===
for i in range(lake_daily_1987_ds['station_rainfall'].rainfall_station.size):
    station_series = lake_daily_1987_ds['station_rainfall'].isel(rainfall_station=i)

    # Split into low (<10 mm) and high (≥10 mm) rainfall
    low_rain_mask = station_series < 10
    high_rain_mask = station_series >= 10

    # Plot low rainfall in semi-transparent colour
    ax1.plot(station_series.time.values[low_rain_mask], station_series.values[low_rain_mask],
        color=presentation_bom_colours[2], linestyle='', marker='o', markersize=4, alpha=0.5, label=None,
        zorder=1)

    # Plot high rainfall with stronger opacity
    ax1.plot(station_series.time.values[high_rain_mask], station_series.values[high_rain_mask],
        color=presentation_bom_colours[2], linestyle='', marker='o', markersize=4, alpha=0.8, label=None,
        zorder=1)

# === Plot full time series of lake size as grey dots ===
ax2.plot(lake_daily_1987_ds['dea'].sel(lake_variable='Size').time.values, lake_daily_1987_ds['dea'].sel(lake_variable='Size').values,
                 color='grey', linestyle='', marker='o', markersize=6, alpha=0.4, zorder=3)

# === Plot each detected event with label ===
plot_dea_events(lake_daily_1987_ds, ax2, zorder_plot=5, zorder_text=10)

# === Axis Formatting ===
ax1.set_ylabel(f'rainfall ({lake_daily_1987_ds['mean_gridded_rainfall'].units}/day)', fontsize=20)
ax2.set_ylabel(f'Lake Size {lake_daily_1987_ds['dea'].sel(lake_variable='Size').units}', fontsize=20)

ax1.grid(True, which='major', axis='both') # Add grid to the background
plot_yearly_xticks(ax1) # Yearly ticks on x-axis
ax2.tick_params(axis='y', labelsize=20)

# === Save figure ===
plt.savefig(f'{output_dir}/{Lake}/Timeseries/DEA_timeseries/{Lake}_FloodEvents-daily_Rainfall-daily_mean_station_categorised.png', bbox_inches='tight')
plt.close()


#%% 3. Scatter Plot: 3.1 Station vs Gridded Rainfall

rainfall_scatter(lake_daily_1987_ds['station_rainfall'], lake_daily_1987_ds['mean_gridded_rainfall'], 100)     #All data (100 percentile)
rainfall_scatter(lake_daily_1987_ds['station_rainfall'], lake_daily_1987_ds['mean_gridded_rainfall'], 95)      #remove top 5% outliers
rainfall_scatter(lake_daily_1987_ds['station_rainfall'], lake_daily_1987_ds['mean_gridded_rainfall'], 99.95)   #remove top 0.05% outliers


#%% 3. Scatter Plot: 3.2 Flood Events Max vs Rainfall Peaks Scatter

"""
rainfall_peaks_near_lake_peaks = find_rainfall_peaks_near_lake_peaks(lake_daily_1987_ds, percentile=0.8)

# === Extract x and y ===
x = rainfall_peaks_near_lake_peaks.values
y = lake_size_max.values
labels = [f"Event {int(e)}" for e in events]


# === Calculate linear regression and confidence interval ===
slope, intercept = np.polyfit(x, y, 1)
y_model = np.polyval([slope, intercept], x)

x_mean = np.mean(x)
n = x.size
dof = n - 2
t = stats.t.ppf(0.975, dof)

residual = y - y_model
std_error = np.sqrt(np.sum(residual**2) / dof)

x_line = np.linspace(np.min(x), np.max(x), 100)
y_line = np.polyval([slope, intercept], x_line)

ci = t * std_error * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))


# === PLOT ===
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111)

# Scatter points
ax.scatter(x, y, s=100, color=blue_colors[2])

# Trend line
ax.plot(x_line, y_line, color=blue_colors[2], label='Linear trend')

# Confidence interval
ax.fill_between(x_line, y_line - ci, y_line + ci, color='#005b96', alpha=0.4, label='95% confidence interval')

# Annotations (optional)
for i, label in enumerate(labels):
    ax.annotate(f'{label}', (x[i], y[i]), fontsize=15, xytext=(5, 5), textcoords='offset points')

# Labels and legend
ax.set_xlabel(f'rainfall peak ({rainfall_peaks_near_lake_peaks.units})', fontsize=20)
ax.set_ylabel(f'Lake size peak ({lake_size_max.units})', fontsize=20)
ax.legend(fontsize=20)
ax.grid(True)
ax.tick_params(axis='both', labelsize=20)

plt.tight_layout()
plt.savefig(f'{output_dir}/{Lake}/{Lake}_FloodEventPeaks_rainfallPeaks_scatter_{Lake_code}_1.png', bbox_inches='tight')
plt.close()
"""

#%% 4. Single Event Plot: Each Flood Event with (cumulative) Rainfall/ Runoff
        
#### Rainfall #### 
#Dates
plot_rainfall_each_event(lake_daily_1987_ds) #Entire Event
plot_rainfall_each_event(lake_daily_1987_ds, rise=True) #Rise
plot_cumulative_rainfall_each_event(lake_daily_1900_ds, offset=False)      

#Offset
plot_rainfall_each_event(lake_daily_1987_ds, offset=True) #Entire Event
plot_rainfall_each_event(lake_daily_1987_ds, offset=True, rise=True) #Rise
plot_cumulative_rainfall_each_event(lake_daily_1900_ds, offset=True)      


#### Runoff #### 
#Dates
plot_runoff_each_event(lake_daily_1987_ds) #Entire Event
plot_runoff_each_event(lake_daily_1987_ds, rise=True) #Rise

#Offset
plot_runoff_each_event(lake_daily_1987_ds, offset=True) #Entire Event
plot_runoff_each_event(lake_daily_1987_ds, offset=True, rise=True) #Rise
      

#%% 5. Histogram Plot: 5.1 All Flood Events as Offset from Max

# Extract variables
events = np.unique(lake_daily_1900_ds.dea_events.dropna(dim='time').values)
dea = lake_daily_1900_ds['dea'].sel(lake_variable='Size')
dea_rise = dea.where(dea.dea_offset <= 0, drop=True)
    

# Set up the figure and axis
fig = plt.figure(figsize=(40, 10))
ax = fig.add_subplot(111)

step_width = np.round(16/events.max(),2)  # Horizontal shift per event
bar_width = step_width*0.9

for i, event in enumerate(events):
    # Mask the data for this event
    dea_rise_event = dea_rise.where(dea_rise.dea_events == event, drop=True)
     
    # Shift the offset for plotting
    shifted_offset = dea_rise_event.dea_offset + (step_width * i)
    
    # Plot bars
    plt.bar(shifted_offset.values, dea_rise_event.values+1,
        width=bar_width, color=dea_colour_list[i], label=f'Event {int(event)}')

# Define offset major and minor tick positions
plot_offset_xticks(dea_rise.dea_offset, ax)

# Labels and legend
plt.xlabel("Days from Event Peak", fontsize=20)
plt.ylabel(f"Lake Size ({dea.units})", fontsize=20)
plt.title("Lake Surface Area by Day Offset for Each Event", fontsize=15)
plt.legend(fontsize=15, bbox_to_anchor=(0, 1), loc='upper left')

# save plot
plt.savefig(f'{output_dir}/{Lake}/{Lake}_FloodEvents_Offset_Rise.png', bbox_inches='tight')
plt.close()
    
#%% 5. Histogram Plot: 5.1 Frequency of Lake Filling Durations

events = np.unique(lake_daily_1987_ds.dea_events.dropna(dim='time').values)
event_offset_daily_1987 = lake_daily_1987_ds.dea_offset
event_offset_rise_daily_1987 = event_offset_daily_1987.where(event_offset_daily_1987 <= 0, drop=True)

event_offsets_min = []
for event in events:
    offset_event = event_offset_daily_1987.where(event_offset_rise_daily_1987.dea_events == event, drop=True)
    min_offset = np.abs(offset_event.min(dim='time')).item()
    event_offsets_min.append(min_offset)
    
# Define bin edges with np.arange (since your values are already floats)
bin_edges = np.arange(0, max(event_offsets_min) + 16, 16)
bin_centers = bin_edges[:-1] + 8  # midpoint of each bin (16/2 = 8)

# Plot histogram
plt.figure(figsize=(10, 5))
plt.hist(event_offsets_min, bins=bin_edges, color='#005b96', edgecolor='black', rwidth=0.9)

plt.xlabel("Lake Filling Duration (days)")
plt.ylabel("Frequency")
plt.xticks(bin_centers, labels=[str(int(b)) for b in bin_edges[:-1]], rotation=90)  
#plt.xticks(bin_edges, rotation=90)  

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.savefig(f'{output_dir}/{Lake}/Frequency_16dBins_rainfall_DEA_Inrease_filtered.pdf', bbox_inches='tight')
plt.close()


#%% 6. Cumulative Plot: 6.1 Cumulative Rainfall Offset from Event Maxima (Events only)

plot_cumulative_rainfall_all_events(lake_daily_1987_ds)
plot_cumulative_rainfall_all_events(lake_daily_1987_ds, magnitude=True)
plot_cumulative_rainfall_all_events(lake_daily_1987_ds, filtered=True)
plot_cumulative_rainfall_all_events(lake_daily_1987_ds, magnitude=True, filtered=True)

#%% 6. Cumulative Plot: 6.2 Reverse Cumulative Rainfall Before Event Maxima (Events only)

ds= lake_daily_1900_ds
events = np.unique(ds.dea_events.dropna(dim='time').values)
rainfall = lake_daily_1900_ds['mean_gridded_rainfall']
max_no_days = 200


# PLOT
fig, ax = plt.subplots(figsize=(30, 10))

for i, event in enumerate(events):
    event_max = ds.where(ds.dea_events==event, drop=True).where(ds.dea_event_max, drop=True)
    event_max_filtered = ds.where(ds.dea_events==event, drop=True).where(ds.dea_offset_filtered==0, drop=True)
    if event_max_filtered.time.size > 0:
        max_date = event_max_filtered.time
    else:
        max_date = event_max.time

    list_rainfall_sums = []    
    for j in range(max_no_days):
        min_date = max_date - pd.Timedelta(days=j)
        rainfall_slice_sum = rainfall.sel(time=slice(min_date.values[0], max_date.values[0])).sum()
        list_rainfall_sums.append(rainfall_slice_sum)
    
    rainfall_sums = xr.DataArray(
        list_rainfall_sums,
        dims=["days"],
        coords={"days": range(1, max_no_days+1)},  
        name="rainfall_sums"
    )
    
    max_date_str = max_date.values[0].astype('datetime64[s]').astype(object).strftime("%Y-%m-%d")    
    ax.plot(rainfall_sums.days.values, rainfall_sums.values, linestyle='-', label = f'Event {int(event)}: {max_date_str}', color=dea_colour_list[i], zorder=1)
    
    # === Axis Formatting ===
    ax.grid()
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=20, ncol=2)
    ax.set_ylabel(f'Rainfall ({rainfall.units})', fontsize=15)
    ax.tick_params(axis='both', labelsize=13)
    
# === Save figure ===
plt.savefig(f'{output_dir}/{Lake}/Cumulative_Rainfall/{Lake}_ReverseCumRainfall_EventsOnly.png', bbox_inches='tight')
plt.close()


#%% 6. Cumulative Plot: 6.3 Cumulative Rainfall build-up before Rolling Sum Maximum (all years)

plot_cum_rainfall_with_threshold(ds=lake_daily_1900_ds, threshold=lower_threshold_daily, time_interval=f'{window_size_daily}_days')
plot_cum_rainfall_with_threshold(ds=lake_monthly_1900_ds, threshold=lower_threshold_monthly, time_interval=f'{window_size_monthly}_months')


#%% 6. Cumulative Plot: 6.4 Optimal Rolling-Sum Duration and Threshold Analysis (all Years)

def select_closest_per_year(da, month=2, max_days=90):
    """
    For each year in the range of `da.time` select the sample closest to (month, day).
    If the closest sample is more than `max_days` away, the result for that year is NaN.

    Parameters
    ----------
    da : xarray.DataArray
        Input DataArray with a 'time' coordinate (datetime64).
    month : int, optional
        Target month (1-12). Default 2 (February).
    max_days : int, optional
        Maximum allowed distance (in days) from the target date to accept a value.
        If the nearest sample is farther than this, the year will be filled with NaN.

    Returns
    -------
    xr.DataArray
        DataArray indexed by year with the chosen values. Also has a coordinate
        `selected_time` (dtype datetime64[ns]) giving the original timestamp chosen
        for each year (NaT where none chosen).
    """
    # ensure input has time coordinate
    if 'time' not in da.coords:
        raise ValueError("Input DataArray must have a 'time' coordinate.")

    times = da['time'].values  # numpy datetime64 array
    if len(times) == 0:
        raise ValueError("Input DataArray 'time' is empty.")

    # year range
    yrs = np.arange(pd.to_datetime(times.min()).year, pd.to_datetime(times.max()).year + 1)

    values = []
    coords=[]

    for y in yrs:
        # target date for this year
        target = np.datetime64(f"{int(y):04d}-{int(month):02d}-01")
        
        # compute absolute day difference for all available times
        # use numpy broadcasting / vectorized arithmetic -> result in float days
        diffs_days = np.abs((times - target) / np.timedelta64(1, 'D')).astype(float)

        # find index of minimum difference
        idx = np.argmin(diffs_days)  # index into times
        min_days = diffs_days[idx]

        if np.isfinite(min_days) and min_days <= max_days:
            sel_time = times[idx]
            sel_val = da.sel(time=sel_time).item()  # scalar value
            sel_coord = da.dea_events.sel(time=sel_time).item()
            values.append(sel_val)
            coords.append(sel_coord)
        else:
            # no acceptable sample within tolerance
            values.append(np.nan)
            coords.append(np.nan)

    # build output DataArray
    result = xr.DataArray(
	data=np.array(values, dtype=da.dtype),
	coords={
		'year': np.array([np.datetime64(f"{y}", "Y") for y in yrs]),
		'dea_events': ('year', coords)},
    dims=['year']
	)
    
    # If original da has attributes/unit/etc, you might want to copy units
    if hasattr(da, 'attrs'):
        result.attrs.update({
            'units': da.attrs['units'],
            'source_da': f'{getattr(da, 'source', 'input_da')}, variable: {getattr(da, 'name', 'input_da')}',
            'selection_target': f'YEAR-{month:02d}-01',
            'max_days_tolerance': int(max_days),
        })

    return result


ds=lake_daily_1900_ds

distance_val = 221
window_max= 208
threshold=600
time_unit='days'

#Extract variables
dea = ds['dea'].sel(lake_variable='Size')
start_date= ds.time.values[0].astype('datetime64[s]').astype(object).strftime("%Y")

rolling_sums_dict = {}
for window in range(1, window_max+1):

    # Step 1: Calculate rolling cumulative sum of mean rainfall
    rainfall_sum = ds['mean_gridded_rainfall'].rolling(time=window, center=False).sum()

    # Step 2: Find peaks
    peaks_and_troughs = identify_peaks_and_troughs(rainfall_sum, distance_val)

    # Sanity check
    if not (len(peaks_and_troughs['peaks']) == len(peaks_and_troughs['troughs']) or len(peaks_and_troughs['peaks']) == len(peaks_and_troughs['troughs']) - 1):
        raise ValueError("Number of peaks must be equal to or one less than number of troughs.")

    peak_dates = rainfall_sum.time[peaks_and_troughs['peaks']]
    rolling_sum_peaks = rainfall_sum.sel(time=peak_dates)
    
    # Step 3: Filter da to yearly subset (with target date)
    rolling_sums_dict[f'{window} days'] = select_closest_per_year(rolling_sum_peaks)


# Combine into one DataArray with new dimension "var"
combined_da = xr.concat(rolling_sums_dict .values(), dim="window")
combined_da = combined_da.assign_coords(window=list(rolling_sums_dict.keys()))
combined_da = combined_da.sel(year=slice(combined_da.year[1], combined_da.year[-1]))

# Mask event years
event_ids = ds.dea_events.where(ds.dea_event_max, drop=True)
event_years = event_ids.time.values.astype('datetime64[Y]')
mask = np.isin(combined_da['year'].values, event_years.astype('datetime64[ns]'))

combined_events_da = combined_da.isel(year=mask)
combined_excluding_events_da = combined_da.isel(year=~mask)

# === Fill gaps so lines are continuous ===
combined_excluding_events_da_filled = (
    combined_excluding_events_da.ffill(dim='window').bfill(dim='window'))

combined_events_da_filled = (
    combined_events_da.ffill(dim='window').bfill(dim='window'))

# === Plot (use the *_filled arrays) ===
fig, ax = plt.subplots(figsize=(20, 10))

for year in combined_excluding_events_da_filled['year'].values:
    ax.plot(
        combined_excluding_events_da_filled['window'].values,
        combined_excluding_events_da_filled.sel(year=year).values,
        color='#b3cde0'
    )

for i, event_year in enumerate(combined_events_da_filled['year'].values):
    yvals = combined_events_da_filled.sel(year=event_year).values
    if np.nanmax(yvals)<600:
        ax.plot(
            combined_events_da_filled['window'].values,
            yvals,
            color='maroon'
        )
    else:
        ax.plot(
            combined_events_da_filled['window'].values,
            yvals,
            color='#005b96'
        )
    #ax.text(int(window_max)+1, np.nanmax(yvals), f'{int(i+1)}', fontsize=8, color='#005b96')

# === Threshold line ===
ax.axhline(y=threshold, color='#03396c', linestyle='--')
ax.axvline(x=112, color='#03396c', linestyle='--')

# === Formatting ===
ax.set_xlabel(f'Cumulative {time_unit.capitalize()}', fontsize=18)
ax.set_ylabel('Cumulative Rainfall (mm)', fontsize=18)
ax.grid(True)
xticks = np.arange(0, int(window_max)+1, 16)
xtick_labels = [f"{int(x)}" for x in xticks]
ax.set_xticks(xticks, xtick_labels, fontsize=13)
ax.tick_params(axis='both', labelsize=13)

# === Save Plot ===
plt.savefig(f'{output_dir}/{Lake}/Cumulative_Rainfall/Cumulative_Rainfall_Rolling_Max/{Lake}_CumRainfall_OptimalRollingSum_ThresholdAnalysis_{start_date}to2024.png', bbox_inches='tight')
plt.close()


#%% 7. Rolling-Sum Plot: Daily and Monthly Rolling-Sum Rainfall (set/ variable window)

### PLOT: timeseries of daily cumulative rainfall ###
plot_rolling_sum_window(lake_daily_1987_ds, '1987', window_size_daily, 'days', lower_threshold_daily, higher_threshold_daily)
plot_rolling_sum_window(lake_daily_1900_ds, '1900', window_size_daily, 'days', lower_threshold_daily, higher_threshold_daily)

### PLOT: timeseries of 80-224 day cumulative rainfall ###
plot_rolling_sum_variable_window(lake_daily_1900_ds, start_date='1900', time_unit='days', threshold=600, window_min=80, window_max=224)      
plot_rolling_sum_variable_window(lake_daily_1987_ds, start_date='1987', time_unit='days', threshold=600, window_min=80, window_max=224)      


### PLOT: timeseries of 4M cumulative rainfall 1900 to 2024 ###
plot_rolling_sum_window(lake_monthly_1987_ds, '1987', window_size_monthly, 'months', lower_threshold_monthly, higher_threshold_monthly)
plot_rolling_sum_window(lake_monthly_1900_ds, '1900', window_size_monthly, 'months', lower_threshold_monthly, higher_threshold_monthly)

### PLOT: timeseries of 2-8 months cumulative rainfall ###
plot_rolling_sum_variable_window(lake_monthly_1900_ds, start_date='1900', time_unit='months', threshold=600, window_min=2, window_max=8)      
plot_rolling_sum_variable_window(lake_monthly_1987_ds, start_date='1987', time_unit='months', threshold=600, window_min=2, window_max=8)      


#%% 8. Driver Plot: 8.1 ENSO & IPO with Events

# === ENSO ===
enso_indices = lake_daily_1900_ds['enso'].enso_indices.values

plot_events_with_enso(lake_daily_1900_ds, enso_indices[0], 'Nino 3.4 (ERSST)')
plot_events_with_enso(lake_daily_1900_ds, enso_indices[1], 'Nino 3.4 (HadISST)')
plot_events_with_enso(lake_daily_1900_ds, enso_indices[2], 'Oni')

# === IPO ===
ipo_indices = lake_daily_1900_ds['ipo'].ipo_indices.values

plot_events_with_ipo(lake_daily_1900_ds, ipo_indices[0])
plot_events_with_ipo(lake_daily_1900_ds, ipo_indices[1])
plot_events_with_ipo(lake_daily_1900_ds, ipo_indices[2])


#%% 8. Driver Plot: 8.2 ENSO & IPO with Rolling Sum

# === ENSO ===
enso_indices = lake_daily_1900_ds['enso'].enso_indices.values

#Nino 3.4 (ERSST)
plot_rolling_sum_with_enso(lake_daily_1987_ds, '1987', window_size_daily, 'days', lower_threshold_daily, higher_threshold_daily, enso_indices[0], 'Nino 3.4 (ERSST)')
plot_rolling_sum_with_enso(lake_daily_1900_ds, '1900', window_size_daily, 'days', lower_threshold_daily, higher_threshold_daily, enso_indices[0], 'Nino 3.4 (ERSST)')

#Nino 3.4 (HadISST)
plot_rolling_sum_with_enso(lake_daily_1987_ds, '1987', window_size_daily, 'days', lower_threshold_daily, higher_threshold_daily, enso_indices[1], 'Nino 3.4 (HadISST)')
plot_rolling_sum_with_enso(lake_daily_1900_ds, '1900', window_size_daily, 'days', lower_threshold_daily, higher_threshold_daily, enso_indices[1], 'Nino 3.4 (HadISST)')

#Oni
plot_rolling_sum_with_enso(lake_daily_1987_ds, '1987', window_size_daily, 'days', lower_threshold_daily, higher_threshold_daily, enso_indices[2], 'Oni')
plot_rolling_sum_with_enso(lake_daily_1900_ds, '1900', window_size_daily, 'days', lower_threshold_daily, higher_threshold_daily, enso_indices[2], 'Oni')


# === IPO ===
ipo_indices = lake_daily_1900_ds['ipo'].ipo_indices.values

#HadISST
plot_rolling_sum_with_ipo(lake_daily_1987_ds, '1987', window_size_daily, 'days', lower_threshold_daily, higher_threshold_daily, ipo_indices[0])
plot_rolling_sum_with_ipo(lake_daily_1900_ds, '1900', window_size_daily, 'days', lower_threshold_daily, higher_threshold_daily, ipo_indices[0])

#ERSST
plot_rolling_sum_with_ipo(lake_daily_1987_ds, '1987', window_size_daily, 'days', lower_threshold_daily, higher_threshold_daily, ipo_indices[1])
plot_rolling_sum_with_ipo(lake_daily_1900_ds, '1900', window_size_daily, 'days', lower_threshold_daily, higher_threshold_daily, ipo_indices[1])

#COBE
plot_rolling_sum_with_ipo(lake_daily_1987_ds, '1987', window_size_daily, 'days', lower_threshold_daily, higher_threshold_daily, ipo_indices[2])
plot_rolling_sum_with_ipo(lake_daily_1900_ds, '1900', window_size_daily, 'days', lower_threshold_daily, higher_threshold_daily, ipo_indices[2])


#%% Table of Events

ds = lake_daily_1987_ds

dea = ds['dea'].sel(lake_variable='Size')
events = np.unique(ds.dea_events.dropna(dim='time').values)
filtered_events = np.unique(ds.dea_offset_filtered.dropna(dim='time').dea_events)
events_max_dates = dea.where(ds.dea_event_max, drop=True).time

start_list = []
filtered_offset_list = []
filtered_cum_max_list = []
rolling_cum_max_list = []
for event in events:
    start_time = dea.where(dea.dea_events == event, drop=True).idxmin(dim='time')
    start_list.append(start_time)
    if ds.dea_offset_filtered.dropna(dim='time').where(ds.dea_events==event, drop=True).any():
        filtered_offset = ds.dea_offset_filtered.dropna(dim='time').where(ds.dea_events==event, drop=True)[0].reset_coords(drop=True)
        filtered_cum_max = ds.mean_gridded_rainfall_event_cum_filtered.dropna(dim='time').where(ds.dea_events ==event, drop=True).max().reset_coords(drop=True)
        
    else:
        filtered_offset = xr.DataArray(np.nan)
        filtered_cum_max = xr.DataArray(np.nan)
    filtered_offset_list.append(filtered_offset)
    filtered_cum_max_list.append(filtered_cum_max)
    rolling_cum_max = ds.mean_gridded_rainfall_rolling_sum.where(ds.cum_window_is_event==event, drop=True).max()
    rolling_cum_max_list.append(rolling_cum_max)
        
events_min_dates = xr.concat(start_list, dim="time").sortby("time")
filtered_offsets = xr.concat(filtered_offset_list, dim="events")
filtered_cum_maxs = xr.concat(filtered_cum_max_list, dim="events")
rolling_cum_maxs = xr.concat(rolling_cum_max_list, dim="events")



# Convert your pieces to numpy/pandas friendly arrays
index = events.astype(int)  # event IDs
col1 = events_min_dates.values  # min dates
col2 = events_max_dates.values  # max dates
col3 = filtered_offsets.values  
col4 = filtered_cum_maxs.values
col5 = rolling_cum_maxs.values  

# Build a DataFrame
events_table = pd.DataFrame({
    "event_no": index,
    "start_date": col1,
    "end_date": col2,
    "duration in days (filtered)": col3,
    "max cum precip (filtered)": col4,
    "112 day cum max": col5
})

# Optional: set event_id as index
events_table.set_index("event_no", inplace=True)

print(events_table)
# Export to CSV
events_table.to_csv(f"{output_dir}/{Lake}/events_table.csv")


#%% CESM Trial 1
rolling_sum = cesm_precip.rolling(time=4, center=False).sum()

for i in range(13):
    #Extract variables
    rainfall_sum_cesm = rolling_sum.sel(time=slice(CF(1987, 1, 1), None)).sel(ensemble=i+1)
    rainfall_sum_cesm = rainfall_sum_cesm.assign_coords(time=rainfall_sum_cesm.indexes['time'].to_datetimeindex())  # datetime64[ns]
    rainfall_sum_agcd = lake_monthly_1987_ds['mean_gridded_rainfall_rolling_sum']
    dea = lake_daily_1987_ds['dea'].sel(lake_variable='Size')
    		
    # Create a large figure with two y-axes (twin axis)
    fig = plt.figure(figsize=(40, 10))
    ax1 = fig.add_subplot(111)        # Left y-axis for rainfall
    ax2 = ax1.twinx()                 # Right y-axis for lake size
    
    # === Plot rolling cumulative rainfall ===
    ax1.plot(rainfall_sum_cesm.time, rainfall_sum_cesm.values, linestyle='-', label='CESM', alpha=0.5, color='green', zorder=2)
    ax1.plot(rainfall_sum_agcd.time, rainfall_sum_agcd.values, linestyle='-', label='AGCD', alpha=0.5, color='lightblue', zorder=2)
    
    # === Plot full time series of lake size with event labels ===    
    ax2.plot(dea.time.values, dea.values, color='grey', linestyle='', marker='o', markersize=8, alpha=0.4, zorder=1)
    plot_dea_events(lake_daily_1987_ds, ax2, zorder_plot=5, zorder_text=10)
    	
    # === Axis Formatting ===
    ax1.set_ylabel(f'Cumulative rainfall ({rainfall_sum.units}/4 months)', fontsize=20)    	
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=20)
    ax1.grid(True, which='major', axis='both') # Add grid to the background
    ax1.xaxis.set_major_locator(mdates.YearLocator(1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax1.get_xticklabels(), rotation=90, ha='center', fontsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    
    # === Save figure ===
    plt.savefig(f'{output_dir}/{Lake}/CESM/cesm_ensemble{i+1}_1987.pdf', bbox_inches='tight')
    plt.close()
    

#%% CESM Trial 2

#Ensemble Mean
cesm_precip_ensemble_mean = cesm_precip.mean('ensemble')

#Mean over DJF Months
cesm_precip_yearly_djf_mean  = cesm_precip_ensemble_mean.where(cesm_precip['time'].dt.season == 'DJF', drop=True).isel(time=slice(2, None)).coarsen(time=3, boundary='trim').mean()

monthly_precip_5Y_mean = cesm_precip_ensemble_mean.groupby((rolling_sum.time.dt.year // 10) * 10).mean(dim='time')*3
djf_precip_5Y_mean = cesm_precip_yearly_djf_mean.groupby((cesm_precip_yearly_djf_mean.time.dt.year // 10) * 10).mean('time')

fig, ax = plt.subplots(figsize=(40, 10))

ax.plot(djf_precip_5Y_mean.year, djf_precip_5Y_mean.values, linestyle='-', label='Mean DJF Monthly Total', alpha=0.5, color='green', zorder=2)
ax.plot(monthly_precip_5Y_mean.year, monthly_precip_5Y_mean.values, linestyle='-', label='Monthly Total', alpha=0.5, color='lightblue', zorder=2)

# === Yearly ticks and formatting ===
ticks = np.arange(850, 2006, 25)
ax.set_xticks(ticks)
ax.set_xticklabels([str(t) for t in ticks], rotation=90)
ax.set_xlim(850, 2005)

plt.grid()
ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=20)
plt.ylabel("Precipitation (mm)")
plt.savefig(f'{output_dir}/{Lake}/CESM/CESM_Trial.pdf', bbox_inches='tight')
plt.close()


rolling_yearly_max = cesm_precip.rolling(time=12, center=False).max()
rolling_yearly_max_5Y_mean = rolling_yearly_max.groupby((rolling_yearly_max.time.dt.year // 5) * 5).max(dim='time').mean('ensemble')

fig, ax = plt.subplots(figsize=(40, 10))
ax.plot(rolling_yearly_max_5Y_mean.year, rolling_yearly_max_5Y_mean.values, linestyle='-', alpha=0.5, color='green', zorder=2)

plt.grid()
plt.ylabel("Precipitation (mm)")
plt.savefig(f'{output_dir}/{Lake}/CESM/CESM_Trial2.pdf', bbox_inches='tight')
plt.close()  


#%% NT Map Events and Event Mean

filtered = True

if filtered:
    suffix = 'filtered'
    dates_filtered = lake_daily_1900_ds.dea_offset_filtered.dropna(dim='time')
    NT_dates = agcd_NT_daily_da.sel(time=dates_filtered.time)
    NT_events = np.unique(NT_dates.dea_events.dropna(dim='time').values)
else:
    suffix = 'unfiltered'
    dates_unfiltered = lake_daily_1900_ds.dea_offset.dropna(dim='time')
    NT_dates = agcd_NT_daily_da.sel(time=dates_unfiltered.time)
    NT_events = np.unique(NT_dates.dea_events.dropna(dim='time').values)

NT_event_list = []
for event in NT_events:
    NT_event_dates = NT_dates.where(NT_dates.dea_events == event, drop=True)
    NT_event_mean = NT_event_dates.sum(dim='time')
    NT_event_list.append(NT_event_mean)

NT_event_precip = xr.concat(NT_event_list, dim='dea_events', join='outer')
NT_event_precip = NT_event_precip.assign_coords(dea_events=NT_events)


# PLOT Each Event
for event in NT_events:

    event_precip = NT_event_precip.where(NT_event_precip.dea_events == event, drop=True).squeeze(dim='dea_events')

    fig = plt.figure(figsize=(NT_event_precip.coords['lon'].size, NT_event_precip.coords['lat'].size))
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Rainfall
    im0 = ax1.pcolormesh(
        event_precip.coords['lon'].values,
        event_precip.coords['lat'].values,
        event_precip.values,
        cmap='Blues',
        norm=Normalize(vmin=0, vmax=3000),
        transform=ccrs.PlateCarree()
    )

    # Lake mask
    Z = np.repeat(np.repeat(Lake_mask_r005.values, 10, axis=0), 10, axis=1)
    X = np.linspace(Lake_mask_r005.coords['lon'].values[0], Lake_mask_r005.coords['lon'].values[-1], Z.shape[1])
    Y = np.linspace(Lake_mask_r005.coords['lat'].values[0], Lake_mask_r005.coords['lat'].values[-1], Z.shape[0])
    ax1.contour(X, Y, Z, colors='k', linewidths=0.75, transform=ccrs.PlateCarree())

    
    # Colorbar
    cb1 = fig.colorbar(im0, fraction=0.025, pad=0.03, ax=ax1, orientation='horizontal')
    cb1.set_label(f'gridded_rainfall ({lake_daily_1987_ds['gridded_rainfall'].units})', fontsize=180)
    cb1.set_ticks(np.linspace(0, 1000, 5))
    cb1.ax.tick_params(labelsize=100)
    cb1.outline.set_color('lightgray')    
        
    # Add gridlines
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = LongitudeLocator()
    gl.ylocator = LatitudeLocator()
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 150, 'color': 'gray'}
    gl.ylabel_style = {'size': 150, 'color': 'gray'}

    # Save figure
    plt.savefig(f'{output_dir}/{Lake}/Maps/NT_maps/NT_map_{suffix}_event{int(event)}.pdf', bbox_inches='tight')
    plt.close()



# PLOT Event Mean
event_mean_precip = NT_event_precip.mean(dim='dea_events')

fig = plt.figure(figsize=(event_mean_precip.coords['lon'].size, event_mean_precip.coords['lat'].size))
ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Plot Rainfall
im0 = ax1.pcolormesh(
    event_mean_precip.coords['lon'].values,
    event_mean_precip.coords['lat'].values,
    event_mean_precip.values,
    cmap='Blues',
    norm=Normalize(vmin=0, vmax=1000),
    transform=ccrs.PlateCarree()
)

# Lake mask
Z = np.repeat(np.repeat(Lake_mask_r005.values, 10, axis=0), 10, axis=1)
X = np.linspace(Lake_mask_r005.coords['lon'].values[0], Lake_mask_r005.coords['lon'].values[-1], Z.shape[1])
Y = np.linspace(Lake_mask_r005.coords['lat'].values[0], Lake_mask_r005.coords['lat'].values[-1], Z.shape[0])
ax1.contour(X, Y, Z, colors='k', linewidths=0.75, transform=ccrs.PlateCarree())


# Colorbar
cb1 = fig.colorbar(im0, fraction=0.025, pad=0.03, ax=ax1, orientation='horizontal')
cb1.set_label(f'gridded_rainfall ({lake_daily_1987_ds['gridded_rainfall'].units})', fontsize=180)
cb1.set_ticks(np.linspace(0, 1000, 5))
cb1.ax.tick_params(labelsize=100)
cb1.outline.set_color('lightgray')    
    
# Add gridlines
gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
              linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
#gl.xlines = False
gl.xlocator = LongitudeLocator()
gl.ylocator = LatitudeLocator()
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 150, 'color': 'gray'}
gl.ylabel_style = {'size': 150, 'color': 'gray'}

# Save figure
plt.savefig(f'{output_dir}/{Lake}/Maps/NT_maps/NT_map_{suffix}_event_mean.pdf', bbox_inches='tight')
plt.close()

    
#%% NT Map 112 day sum Compostite

#only run oce
# Make Landmask from Global land polygons (Natural Earth ~1:110m)
land = regionmask.defined_regions.natural_earth_v5_0_0.land_50
land_mask = land.mask(agcd_NT_daily_da.isel(time=0)).notnull()
agcd_NT_daily_masked_da = agcd_NT_daily_da.where(land_mask)


event = False
start_date ='1987'
end_date = '2024'

# Dates during event/ non-event -> NT subset
dates_window = lake_daily_1900_ds.cum_window_112_days.dropna(dim='time').time
start_date_1987 = dates_window.where(dates_window['cum_window_112_days'] == 1, drop=True).sel(time=slice('1987', None))[0]
dates_post1987 = dates_window.sel(time=dates_window.time >= start_date_1987)
dates_pre1987  = dates_window.sel(time=dates_window.time <  start_date_1987)

if event:
    event_suffix = 'event'
    dates_subset_post1987 = dates_post1987.where(~dates_post1987.cum_window_is_event.isnull(), drop=True)
    if start_date =='1900':
        dates_subset_pre1987 = dates_pre1987.where(dates_pre1987['rolling_peak_above_600'], drop=True)
        if end_date == '1987':
            dates_subset = dates_subset_pre1987
        elif end_date == '2024':
            dates_subset = xr.concat([dates_subset_pre1987, dates_subset_post1987], dim="time")    
    
    elif start_date =='1987':
        dates_subset = dates_subset_post1987 
        end_date = '2024'

else:    
    event_suffix = 'non_event'
    dates_subset_post1987 = dates_post1987.where(dates_post1987.cum_window_is_event.isnull(), drop=True)
    
    if start_date =='1900':
        dates_subset_pre1987 = dates_pre1987.where(~dates_pre1987['rolling_peak_above_600'], drop=True)
        if end_date == '1987':
            dates_subset = dates_subset_pre1987
        elif end_date == '2024':
            dates_subset = xr.concat([dates_subset_pre1987, dates_subset_post1987], dim="time")
    
    elif start_date =='1987':
        dates_subset = dates_subset_post1987 
        end_date = '2024'

NT_window_subset = agcd_NT_daily_masked_da.sel(time=dates_subset)


# Mean of all events/ non-events
NT_window_list = []
for i in range(int(NT_window_subset.time.size/window_size_daily)):
    NT_window_slice = NT_window_subset.isel(time=slice(window_size_daily*i, window_size_daily*(i+1)))
    NT_window_slice_sum = NT_window_slice.sum(dim='time')
    NT_window_list.append(NT_window_slice_sum)
NT_window_subset_mean = xr.concat(NT_window_list, dim="years").mean(dim="years")


# PLOT
fig = plt.figure(figsize=(40, 40))
ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax1.coastlines()  # Add coastlines

# Plot Rainfall
im0 = ax1.pcolormesh(
    NT_window_subset_mean.coords['lon'].values,
    NT_window_subset_mean.coords['lat'].values,
    NT_window_subset_mean.values,
    cmap='Blues',
    norm=Normalize(vmin=0, vmax=600),
    transform=ccrs.PlateCarree()
)

# Lake mask
Z = np.repeat(np.repeat(Lake_mask_r005.values, 10, axis=0), 10, axis=1)
X = np.linspace(Lake_mask_r005.coords['lon'].values[0], Lake_mask_r005.coords['lon'].values[-1], Z.shape[1])
Y = np.linspace(Lake_mask_r005.coords['lat'].values[0], Lake_mask_r005.coords['lat'].values[-1], Z.shape[0])
ax1.contour(X, Y, Z, colors='k', linewidths=0.75, transform=ccrs.PlateCarree())


# Colorbar
cb1 = fig.colorbar(im0, fraction=0.025, pad=0.03, ax=ax1, orientation='horizontal')
cb1.set_label(f'Rainfall ({lake_daily_1987_ds['gridded_rainfall'].units}/ 112 days)', fontsize=30)
cb1.set_ticks(np.linspace(0, 600, 7))
cb1.ax.tick_params(labelsize=30)
cb1.outline.set_color('lightgray')    
    
# Add gridlines
gl = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
              linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlocator = LongitudeLocator()
gl.ylocator = LatitudeLocator()
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 30, 'color': 'black'}
gl.ylabel_style = {'size': 30, 'color': 'black'}

# Save figure
plt.savefig(f'{output_dir}/{Lake}/Maps/NT_maps/NAus_map_window_{event_suffix}_mean_{start_date}_{end_date}.png', bbox_inches='tight')
plt.close()

    
#%%

ds=lake_daily_1900_ds
threshold=lower_threshold_daily
time_interval=f'{window_size_daily}_days'

    
window, time_unit = time_interval.split("_", 1)
events = np.unique(ds.dea_events.dropna(dim='time').values)

cum_to_peak = ds.mean_gridded_rainfall_cum_to_peak

# === No-Event Segments (after 1987) ===
no_event = cum_to_peak.where(cum_to_peak.cum_window_is_event.isnull(), drop=True).dropna(dim='time')
dates = no_event.where(no_event[f'cum_window_{time_interval}'] == 1, drop=True).sel(time=slice('1987', None))
if dates.size > 0:
    start = dates[0].time
    # Filter only segments after that date
    no_event = no_event.sel(time=slice(start, None))
else:
    no_event = xr.DataArray([], dims='time')

# === Load "event" segments (always post-1987) ===
event = cum_to_peak.where(~cum_to_peak.cum_window_is_event.isnull(), drop=True).dropna(dim='time')


#Rank lake size maximas in descending order
lake_size_maximums = ds['dea'].sel(lake_variable='Size').where(ds.dea_event_max, drop=True).dropna(dim='time')
ranked_lake_size_max = lake_size_maximums.sortby(lake_size_maximums, ascending=True)
event_ranking = ranked_lake_size_max['dea_events'].values.astype(int).tolist()

# Ranked Colourmap
original_cmap = colormaps['Blues']
cmap = truncate_colormap(original_cmap, minval=0.2, maxval=1.0)
norm = plt.Normalize(vmin=1, vmax=len(event_ranking))
colours = [cmap(norm(i + 1)) for i in range(len(event_ranking))]

        
# === Start Plot ===
fig = plt.figure(figsize=(15, 10))
ax1 = fig.add_subplot(111)

# === Plot each group ===
for i, event_rank in enumerate(event_ranking):
    event_seg = event.where(event.cum_window_is_event == event_rank, drop=True)
    #event_no = event_rank
    event_seg_below = event_seg.where(~event_seg[f'cum_window_above_{threshold}'], drop=True)
    event_seg_above = event_seg.where(event_seg[f'cum_window_above_{threshold}'], drop=True)
    ax1.plot(event_seg_below[f'cum_window_{time_interval}'], event_seg_below.values, color=colours[i], linestyle='-', markersize=6, label = f'Lake Filling Event - below threshold of {threshold} mm', zorder=1)
    ax1.plot(event_seg_above[f'cum_window_{time_interval}'], event_seg_above.values, color=colours[i], linestyle='-', markersize=6, label = 'Lake Filling Event', zorder=1)
    #ax1.text(int(window)+(event_no/5), event_seg.max(), str(event_no), fontsize=8, color='#005b96')
    
for i in range(no_event.size // int(window)):
    no_event_seg = no_event[int(window) * i : int(window) * (i+1)]
    no_event_seg_below = no_event_seg.where(~no_event_seg[f'cum_window_above_{threshold}'], drop=True)
    no_event_seg_above = no_event_seg.where(no_event_seg[f'cum_window_above_{threshold}'], drop=True)
    ax1.plot(no_event_seg_below[f'cum_window_{time_interval}'], no_event_seg_below.values, color='lightgrey', linestyle='--', markersize=6, label = 'No Lake Filling', zorder=1)
    ax1.plot(no_event_seg_above[f'cum_window_{time_interval}'], no_event_seg_above.values, color='maroon', linestyle='--', markersize=6, label = f'No Lake Filling - above threshold of {threshold} mm', zorder=1)


# === Clean Legend (avoid duplicates) ===
handles = [
    mlines.Line2D([], [], color='#005b96', linestyle='-', label='Lake Filling'),
    mlines.Line2D([], [], color='lightgrey', linestyle='--', label='No Lake Filling                           '),
    mlines.Line2D([], [], color='white', linestyle='--', label=''),
    mlines.Line2D([], [], color='white', linestyle='--', label='')
    ]
ax1.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, 1), fontsize=15)


#colourbar
cbar_ax = inset_axes(ax1, width=3.1, height=0.15, loc='upper left', bbox_to_anchor=(0.05, 0.835, 1, 0.05), bbox_transform=ax1.transAxes, borderpad=0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_ticks([events.min(), events.max()])
cbar.set_ticklabels(['smallest', 'highest'])
cbar.ax.tick_params(labelsize=15) 


# === Threshold line ===
ax1.axhline(y=threshold, color='#03396c', linestyle='--')

# === Formatting ===
ax1.set_xlabel(f'{time_unit.capitalize()} Leading up to Peak', fontsize=18)
ax1.set_ylabel('Cumulative Rainfall (mm)', fontsize=18)
ax1.grid(True)
xticks = np.arange(0, int(window), 10)
xtick_labels = [f"{int(x)}" for x in xticks]
ax1.set_xticks(xticks, xtick_labels, fontsize=13)
ax1.tick_params(axis='both', labelsize=13)

# === Save Plot ===
plt.savefig(f'{output_dir}/{Lake}/Cumulative_Rainfall/Cumulative_Rainfall_Rolling_Max/{Lake}_CumRainfall_BeforeRollingMax_{window}{time_unit}_Magnitude.png', bbox_inches='tight')
plt.close()

