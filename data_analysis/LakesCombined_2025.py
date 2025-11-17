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
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize  # already available via your imports
from matplotlib import colormaps
import matplotlib.lines as mlines
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LongitudeLocator, LatitudeLocator)
import warnings

Lake = 'LW'
Lake_longname = 'LakeWoods'

#Define variables for rolling sum
window_size_daily = 112  # Define the rolling window size in days
lower_threshold_daily = 600 # Define lower threshold
higher_threshold_daily = 800 # Define higher threshold


input_dir = r'/Users/leasophiegrunau/Documents/Work/Bewerbungen/code-examples-sophie-grunau/data_analysis/Data'
output_dir = r'/Users/leasophiegrunau/Documents/Work/Bewerbungen/code-examples-sophie-grunau/data_analysis/Output'

presentation_bom_colours = ['#000033','#336666', '#99cc33', '#339966','#8EB28E','#336600']
blue_colors = ['#0000FF', '#00008B', '#4169E1', '#6495ED', '#87CEFA', '#4682B4', '#5F9EA0', '#7B68EE', '#87CEEB', '#ADD8E6']
dea_colour_list = ['tomato', 'darkred', 'orange', 'gold', 'olive', 'forestgreen', 'teal', 'aqua', 'steelblue', 'navy', 'purple', 'fuchsia', 'pink', 'maroon', 'lightcoral', 'red', 'sienna', 'tan', '#000033','#336666', '#99cc33', '#339966','#8EB28E','#336600', '#87CEEB', '#ADD8E6']


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


def read_enso_files():
    """
    Reads ENSO time series data files and returns a combined xarray DataArray.

    Returns
    -------
    xr.DataArray
        A DataArray of shape (time, enso_indices) with:
        - `time` coordinate: datetime index corresponding to year and month.
        - `enso_indices` coordinate: string labels from the metadata for each index.
        - Attributes:
            - `units`: '°C'
            - `source`: NOAA
            - `url`: https://psl.noaa.gov/data/timeseries/month/.
        - Name: 'Monthly ENSO indices'
    """
    file_paths = glob.glob(f'{input_dir}/ClimaticDrivers/enso_*.nc')
    
    indices_list = []
    for path in file_paths:
        ds = xr.open_dataset(path)
        var_name = list(ds.data_vars)
        da_var = ds[var_name[0]]    
        indices_list.append(da_var)
    
    enso_indices_names = [da.name if da.name is not None else f"da{i+1}" for i, da in enumerate(indices_list)]
    da_combined = xr.concat(indices_list, dim='enso_indices', join='outer')
    da_combined = da_combined.assign_coords(enso_indices=np.array(enso_indices_names, dtype=object))
    da_combined = da_combined.rename('enso')
    da_combined.attrs = {
        "long_name": "Monthly ENSO indices",
        "description": (
            "Combined ENSO dataset containing multiple sea surface temperature anomaly indices along an 'enso_indices' dimension. "
            "Includes Niño 3.4 (ERSSTv5), Niño 3.4 (HadISST), and the Oceanic Niño Index (ONI) from NOAA CPC."
        ),
        "source": "NOAA",
        "units": "°C",
        "url": "https://psl.noaa.gov/data/timeseries/month/",
    }
    return da_combined


def read_ipo_files():
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
    file_paths = glob.glob(f'{input_dir}/ClimaticDrivers/tpi.timeseries*.txt')
    
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

def threshold_analysis_select_closest_per_year(da, month=2, max_days=90):
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

    
def window_sum_vectorised(ds, da, window_size):

    da = da.assign_coords({
        'cum_window': ds[f'cum_window_{window_size_daily}_days'].reset_coords(drop=True),
        'cum_window_is_event': ds['cum_window_is_event'].reset_coords(drop=True),
        'cum_window_above_threshold': ds[f'cum_window_above_{lower_threshold_daily}'].reset_coords(drop=True)
    })
    da_dropped_na = da.where(~da.cum_window.isnull(), drop=True)
        
    k = window_size
    n = da_dropped_na.sizes['time']
    
    # Create a DataArray to use as group labels
    chunk_labels = xr.DataArray(np.arange(n) // k, dims='time')
    
    # Group by these labels, sum and ass coords
    windowed_sum  = da_dropped_na.groupby(chunk_labels).sum(dim='time')
    windowed_sum_coords_events = da_dropped_na.cum_window_is_event.groupby(chunk_labels).mean(dim='time')
    windowed_sum_coords_above_threshold = da_dropped_na.cum_window_above_threshold.groupby(chunk_labels).sum(dim='time')
    windowed_sum_coords_time = da_dropped_na.time.groupby(chunk_labels).first()
    
    windowed_sum = windowed_sum.assign_coords({
        'time': windowed_sum_coords_time,
        'above_threshold': windowed_sum_coords_above_threshold,
        'dea_events': windowed_sum_coords_events
    })
    windowed_sum = windowed_sum.swap_dims({'group': 'time'})
    return windowed_sum


#### ========= Functions: Plot data =========

def plot_dea_timeseries(ds, time_unit_lake, timeseries_type=None, time_unit_data=None, plot_mean=False, plot_station=False, peaks=False):
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
        - 'find_rainfall_peaks_near_lake_peaks' - indetifies rainfall peaks closest to events
    - Saves the figure to:
      `{output_dir}/{Lake}/{Lake}_FloodEvents-{time_unit_lake}_{timeseries_type.capitalize()}-{time_unit_data}{file_suffix_mean}{file_suffix_station}.png`
    """
    
    ## File naming
    # Only dea data plotted
    if not plot_mean and not plot_station:	
        file_name = f'{Lake}_FloodEvents-{time_unit_lake}'
            
    # Dea data and mean and / or station data is plotted 
    else:
        # Peaks: only highlight if all conditions met. If not, issue warning and disable peaks
        if peaks:
            if plot_mean and time_unit_data == 'monthly' and timeseries_type == 'rainfall':
                file_name_part1 = f'{Lake}_FloodEvents-{time_unit_lake}_RainfallPeaks-monthly'
            else:
                warnings.warn("Peaks can only be highlighted for monthly rainfall; plotting without peak markers.")
        else:
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
    events = np.unique(ds['dea_events'].dropna(dim='time').values)
    # DEA: use input ds if daily, otherwise convert to monthly. Abort if wrong time_unit.
    if time_unit_lake == 'daily': lake_size = ds['dea'].sel(lake_variable='Size')
    elif time_unit_lake == 'monthly':
        monthly_ds = resample_lake_dataset_to_monthly(ds)
        lake_size = monthly_ds['dea'].sel(lake_variable='Size')
    else: 
        print("time_unit_lake ('daily'/'monthly') missing or wrong. Aborting.")
        return
	
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
    for event in events:
        event_sizes = lake_size.where(lake_size.dea_events == event, drop=True).dropna(dim='time')
        axis_dea.plot(event_sizes.time, event_sizes, linestyle='--', marker='o', markersize=6,
                     color=presentation_bom_colours[0], label=f'Event {int(event)}', zorder=5)
    
        label_size = event_sizes.where(event_sizes.dea_event_max, drop=True)
        axis_dea.text(label_size.time, label_size + 3, f'Event {int(event)}',
                     fontsize=20, ha='center', va='bottom',
                     color=presentation_bom_colours[0], zorder=10)

    
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
    ax1.xaxis.set_major_locator(mdates.YearLocator(1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax1.get_xticklabels(), rotation=90, ha='center', fontsize=20)    
    
    # === Save figure ===
    plt.savefig(f'{output_dir}/Fig2_{file_name}.png', bbox_inches='tight')
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
    plt.savefig(f'{output_dir}/Fig3_{Lake}_FloodEvents_Rainfall_Offset_Rise_Cumulative_AllEvents{suffix}{filterd_var}.png', bbox_inches='tight')
    plt.close() 


def plot_cum_rainfall_with_threshold(ds, threshold, time_interval, ranked=False):
    """
    Plot cumulative rainfall segments associated with lake-filling events
    and non-events, categorized by whether they fall below or above a given threshold.
    
    Parameters:
        ds (xr.Dataset): Input dataset containing cumulative rainfall segments.
        threshold (float): rainfall threshold to distinguish event types.
        time_interval (str): Unit of time with window (e.g. '128_days').
    """

    events = np.unique(ds.dea_events.dropna(dim='time').values)
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
    
    if ranked:
        file_suffix = '_Magnitude'
        #Rank lake size maximas in descending order
        lake_size_maximums = ds['dea'].sel(lake_variable='Size').where(ds.dea_event_max, drop=True).dropna(dim='time')
        ranked_lake_size_max = lake_size_maximums.sortby(lake_size_maximums, ascending=True)
        event_ranking = ranked_lake_size_max['dea_events'].values.astype(int).tolist()

        # Ranked Colourmap
        original_cmap = colormaps['Blues']
        cmap = truncate_colormap(original_cmap, minval=0.2, maxval=1.0)
        norm = plt.Normalize(vmin=1, vmax=len(event_ranking))
        colours_above = [cmap(norm(i + 1)) for i in range(len(event_ranking))]
        colours_below = colours_above
    else:
        file_suffix = ''
        event_ranking = events
        colours_above = ['#005b96'] * len(events)
        colours_below = ['maroon'] * len(events)
    
    # === Start Plot ===
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(111)
    
    # === Plot each group ===
    for i, event_rank in enumerate(event_ranking):
        event_seg = event.where(event.cum_window_is_event == event_rank, drop=True)
        event_seg_below = event_seg.where(~event_seg[f'cum_window_above_{threshold}'], drop=True)
        event_seg_above = event_seg.where(event_seg[f'cum_window_above_{threshold}'], drop=True)
        ax1.plot(event_seg_below[f'cum_window_{time_interval}'], event_seg_below.values, color=colours_below[i], linestyle='-', markersize=6, label = f'Lake Filling Event - below threshold of {threshold} mm', zorder=1)
        ax1.plot(event_seg_above[f'cum_window_{time_interval}'], event_seg_above.values, color=colours_above[i], linestyle='-', markersize=6, label = 'Lake Filling Event', zorder=1)
        
    for i in range(no_event.size // int(window)):
        no_event_seg = no_event[int(window) * i : int(window) * (i+1)]
        no_event_seg_below = no_event_seg.where(~no_event_seg[f'cum_window_above_{threshold}'], drop=True)
        no_event_seg_above = no_event_seg.where(no_event_seg[f'cum_window_above_{threshold}'], drop=True)
        ax1.plot(no_event_seg_below[f'cum_window_{time_interval}'], no_event_seg_below.values, color='lightgrey', linestyle='--', markersize=6, label = 'No Lake Filling', zorder=1)
        ax1.plot(no_event_seg_above[f'cum_window_{time_interval}'], no_event_seg_above.values, color='maroon', linestyle='--', markersize=6, label = f'No Lake Filling - above threshold of {threshold} mm', zorder=1)
    
    
    # === Clean Legend (avoid duplicates) ===
    if ranked:
        handles = [
            mlines.Line2D([], [], color='#005b96', linestyle='-', label='Lake Filling'),
            mlines.Line2D([], [], color='lightgrey', linestyle='--', label='No Lake Filling                           '),
            mlines.Line2D([], [], color='white', linestyle='--', label=''),
            mlines.Line2D([], [], color='white', linestyle='--', label='')
            ]
        
        #colourbar
        cbar_ax = inset_axes(ax1, width=3.1, height=0.15, loc='upper left', bbox_to_anchor=(0.05, 0.835, 1, 0.05), bbox_transform=ax1.transAxes, borderpad=0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks([events.min(), events.max()])
        cbar.set_ticklabels(['smallest', 'highest'])
        cbar.ax.tick_params(labelsize=15)
    else:
        handles = [
            mlines.Line2D([], [], color='#005b96', linestyle='-', label='Lake Filling'),
            mlines.Line2D([], [], color='maroon', linestyle='-', label=f'Lake Filling - below {threshold} mm'),
            mlines.Line2D([], [], color='maroon', linestyle='--', label=f'No Lake Filling - above {threshold} mm'),
            mlines.Line2D([], [], color='lightgrey', linestyle='--', label='No Lake Filling')
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
    plt.savefig(f'{output_dir}/Fig4_{Lake}_CumRainfall_BeforeRollingMax_{window}{time_unit}{file_suffix}.png', bbox_inches='tight')
    plt.close()

    
def plot_threshold_analysis(ds, window_max, threshold, time_unit):    
    #Extract variables
    start_date= ds.time.values[0].astype('datetime64[s]').astype(object).strftime("%Y")
    
    if time_unit == 'days':
        distance_val = 221
    elif time_unit == 'months':
        distance_val = 8
    else:
        print("Wrong time_unit. Aborting.")
        return        
    
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
        rolling_sums_dict[f'{window} days'] = threshold_analysis_select_closest_per_year(rolling_sum_peaks)
    
    
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
    plt.savefig(f'{output_dir}/Fig5_{Lake}_CumRainfall_OptimalRollingSum_ThresholdAnalysis_{start_date}to2024.png', bbox_inches='tight')
    plt.close()

        
def plot_rolling_sum_window(ax_left, ax_right, ds, start_date, window, time_unit, threshold1, threshold2, plot_peaks_troughs=False, peaks_troughs=None):
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
    events = np.unique(ds['dea_events'].dropna(dim='time').values)

		
    # === Plot rolling cumulative rainfall ===
    ax_left.plot(rainfall_sum.time, rainfall_sum.values,
             linestyle='-', label=f'{window} {time_unit.capitalize()} Rolling Sum of Rainfall', alpha=0.5, color='#b3cde0', zorder=2)
    
    ax_left.plot(rainfall_sum_above_threshold1.time, rainfall_sum_above_threshold1.values,
             linestyle='-', label=f'Peak above {threshold1} mm', alpha=0.5, color='#011f4b', zorder=3)
    
    ax_left.plot(rainfall_sum_above_threshold2.time, rainfall_sum_above_threshold2.values,
             linestyle='-', label=f'Peak above {threshold2} mm', alpha=0.5, color='maroon', zorder=4)
    
    # === Plot peaks and troughs ===
    if plot_peaks_troughs:
        if peaks_troughs is None:
            raise ValueError("If plot_peaks_troughs is True, you must provide a peaks_troughs dictionary.")

        peaks = peaks_troughs['peaks']
        troughs = peaks_troughs['troughs']

        ax_left.plot(peaks.time, peaks.values, linestyle='', marker='o', label='Peaks', alpha=0.5, color='maroon', zorder=2)
        ax_left.plot(troughs.time, troughs.values,linestyle='', marker='o', label='Troughs', alpha=0.5, color='green', zorder=2)
    
    # === Plot full time series of lake size with event labels ===    
    ax_right.plot(dea.time.values, dea.values,
                 color='grey', linestyle='', marker='o', markersize=8, alpha=0.4, zorder=1)
    
    for event in events:
        event_sizes = dea.where(dea.dea_events == event, drop=True).dropna(dim='time')
        ax_right.plot(event_sizes.time, event_sizes, linestyle='--', marker='o', markersize=6,
                     color=presentation_bom_colours[0], label=f'Event {int(event)}', zorder=5)
    
        label_size = event_sizes.where(event_sizes.dea_event_max, drop=True)
        ax_right.text(label_size.time, label_size + 3, f'Event {int(event)}',
                     fontsize=20, ha='center', va='bottom',
                     color=presentation_bom_colours[0], zorder=10)
        
    # === Add a horizontal line at threshold levels ===
    ax_left.axhline(y=threshold1, color='#03396c', linestyle='--')
    ax_left.axhline(y=threshold2, color='maroon', linestyle='--')
    	
    # === Axis Formatting ===
    #ax_left.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=20)
    ax_left.grid(True, which='major', axis='both') # Add grid to the background
    ax_left.xaxis.set_major_locator(mdates.YearLocator(grid_spacing))
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_left.get_xticklabels(), rotation=90, ha='center', fontsize=20)
    ax_left.set_ylabel(f'Cumulative Rainfall ({rainfall_sum.units}/{window} {time_unit})', fontsize=20)    	    
    ax_left.tick_params(axis='y', labelsize=20)
    ax_right.set_ylabel(f'Lake Size {dea.units}', fontsize=20)    	    
    ax_right.tick_params(axis='y', labelsize=20)


def plot_rolling_sum_variable_window(ax_left, ax_right, ds, start_date, time_unit, threshold, window_min, window_max):
    
    if time_unit == 'days':
        distance_val = 180
        time_step = 16
    elif time_unit == 'months':
        distance_val = 6
        time_step = 1
    else:
        print("Wrong time_unit. Needs to 'days' or 'months'.")
        return
    
    if start_date=='1900':
        grid_spacing=5
    else:
        grid_spacing=1
        
    # Colourmap
    norm = plt.Normalize(vmin=1, vmax=((window_max - window_min)/time_step)+1)
    cmap_blue = truncate_colormap(colormaps['Blues'], minval=0.2, maxval=1.0)
    colours_blue = [cmap_blue(norm(i + 1)) for i in range(int(((window_max - window_min)/time_step)+1))]
    cmap_red = truncate_colormap(colormaps['Reds'], minval=0.2, maxval=1.0)
    colours_red = [cmap_red(norm(i + 1)) for i in range(int(((window_max - window_min)/time_step)+1))]
    
    #Extract variables
    dea = ds['dea'].sel(lake_variable='Size')
    events = np.unique(ds['dea_events'].dropna(dim='time').values)

    # Start figure
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
        ax_left.plot(rainfall_sum.time, rainfall_sum.values,
                 linestyle='-', label=f'{window} {time_unit.capitalize()} Rolling Cumulative Sum of Rainfall', alpha=0.5, color=colours_blue[i], zorder=2)
    
        ax_left.plot(rainfall_sum_above_threshold.time, rainfall_sum_above_threshold.values,
                 linestyle='-', label=f'{window} {time_unit.capitalize()} Rolling Cumulative Sum of Rainfall', alpha=0.5, color=colours_red[i], zorder=2)
        
    # === Plot full time series of lake size with event labels ===    
    ax_right.plot(dea.time.values, dea.values,
                 color='grey', linestyle='', marker='o', markersize=8, alpha=0.4, zorder=1)
        
    for event in events:
        event_sizes = dea.where(dea.dea_events == event, drop=True).dropna(dim='time')
        ax_right.plot(event_sizes.time, event_sizes, linestyle='--', marker='o', markersize=6,
                     color=presentation_bom_colours[0], label=f'Event {int(event)}', zorder=5)
    
        label_size = event_sizes.where(event_sizes.dea_event_max, drop=True)
        ax_right.text(label_size.time, label_size + 3, f'Event {int(event)}',
                     fontsize=20, ha='center', va='bottom',
                     color=presentation_bom_colours[0], zorder=10)
        
    # === Axis Formatting ===
    ax_left.set_ylabel(f'Cumulative Rainfall ({rainfall_sum.units})', fontsize=20)    	
    ax_left.grid(True, which='major', axis='both') # Add grid to the background
    ax_left.xaxis.set_major_locator(mdates.YearLocator(grid_spacing))
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_left.get_xticklabels(), rotation=90, ha='center', fontsize=20)


def plot_NT_event_map(ax_no, da, start_date, end_date, lake_mask, event=False):
    pre1987 = da.sel(time=slice(da.time[0], '1986'))
    post1987 = da.sel(time=slice('1987', da.time[-1]))

    if event:
        pre1987_subset = pre1987.where(pre1987.above_threshold > 0, drop=True)
        post1987_subset = post1987.where(~post1987.dea_events.isnull(), drop=True)
    else:
        pre1987_subset = pre1987.where(pre1987.above_threshold == 0, drop=True)
        post1987_subset = post1987.where(post1987.dea_events.isnull(), drop=True)
    
    if start_date == '1900':
        if end_date == '1987':
            subset = pre1987_subset.mean(dim='time')
        elif end_date == '2024':  
            subset = xr.concat([pre1987_subset, post1987_subset], dim='time').mean(dim='time')
        else:
            print("Wrong end_date. Aborting.")
            return               
    elif start_date =='1987':
        subset = post1987_subset.mean(dim='time')
        end_date = '2024'
    else:
        print("Wrong start_date. Aborting.")
        return       
    
    # PLOT    
    ax_no.coastlines()  # Add coastlines
    
    # Plot Rainfall
    im0 = ax_no.pcolormesh(
        subset.coords['lon'].values,
        subset.coords['lat'].values,
        subset.values,
        cmap='Blues',
        norm=Normalize(vmin=0, vmax=600),
        transform=ccrs.PlateCarree()
    )
    
    # Lake mask
    ax_no.contour(lake_mask.lon, lake_mask.lat, lake_mask.values, colors='k', linewidths=0.75, transform=ccrs.PlateCarree())
        
    # Add gridlines
    gl = ax_no.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = LongitudeLocator()
    gl.ylocator = LatitudeLocator()
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    
    return im0
    

def plot_rolling_sum_with_driver(ax_left, ax_right1, ax_right2, ds, start_date, driver, driver_index, window, time_unit, threshold1, threshold2):
    if start_date=='1900': 
        grid_spacing=5
    else:
        grid_spacing=1
    
    if driver == 'enso':
        driver_da = lake_daily_1900_ds['enso'].sel(enso_indices=driver_index).sel(time=slice(f"{start_date}-01-01", ds.time[-1]))
    elif driver == 'ipo':
        driver_da = lake_daily_1900_ds['ipo'].sel(ipo_indices=driver_index).sel(time=slice(f"{start_date}-01-01", ds.time[-1]))
    else:
        print("Wrong driver. Aborting.")
        return          

    #Extract variables
    rainfall_sum = ds['mean_gridded_rainfall_rolling_sum']
    rainfall_sum_above_threshold1 = rainfall_sum.where(rainfall_sum[f'rolling_peak_above_{threshold1}'])
    rainfall_sum_above_threshold2 = rainfall_sum.where(rainfall_sum[f'rolling_peak_above_{threshold2}'])
    dea = ds['dea'].sel(lake_variable='Size')
    events = np.unique(ds['dea_events'].dropna(dim='time').values)
    	
    ### PLOT: timeseries of 128 day cumulative rainfall with SOI ###
    ax_right2.spines['right'].set_position(('outward', 120))
    
    # Fill the positive and negative areas with different colors
    ax_left.fill_between(driver_da.time, driver_da.values, where=driver_da.values >= 0, color='lightcoral', alpha=0.5)
    ax_left.fill_between(driver_da.time, driver_da.values, where=driver_da.values < 0, color='lightblue', alpha=0.5)
    
    # === Plot rolling cumulative rainfall ===
    ax_right1.plot(rainfall_sum.time, rainfall_sum.values,
             linestyle='-', label=f'{window} {time_unit.capitalize()} Rolling Cumulative Sum of rainfall', alpha=0.5, color='#b3cde0', zorder=2)
    
    ax_right1.plot(rainfall_sum_above_threshold1.time, rainfall_sum_above_threshold1.values,
             linestyle='-', label=f'Peak above {threshold1} mm', alpha=0.5, color='#011f4b', zorder=3)
    
    ax_right1.plot(rainfall_sum_above_threshold2.time, rainfall_sum_above_threshold2.values,
             linestyle='-', label=f'Peak above {threshold2} mm', alpha=0.5, color='maroon', zorder=4)
    
    
    # === Plot full time series of lake size with event labels ===    
    ax_right2.plot(dea.time.values, dea.values,
                 color='grey', linestyle='', marker='o', markersize=8, alpha=0.4, zorder=1)
        
    for event in events:
        event_sizes = dea.where(dea.dea_events == event, drop=True).dropna(dim='time')
        ax_right2.plot(event_sizes.time, event_sizes, linestyle='--', marker='o', markersize=6,
                     color=presentation_bom_colours[0], label=f'Event {int(event)}', zorder=5)
    
        label_size = event_sizes.where(event_sizes.dea_event_max, drop=True)
        ax_right2.text(label_size.time, label_size + 3, f'Event {int(event)}',
                     fontsize=20, ha='center', va='bottom',
                     color=presentation_bom_colours[0], zorder=10)
    
    
    # === Add a horizontal line at threshold levels ===
    ax_right1.axhline(y=threshold1, color='#03396c', linestyle='--')
    ax_right1.axhline(y=threshold2, color='maroon', linestyle='--')
    
    # === Axis Formatting ===
    ax_left.set_ylabel(f'{driver.upper()} ({driver_index})', fontsize=20)
    ax_right1.set_ylabel('128 day Cumulative Precipitation (mm/128 days)', fontsize=20)
    ax_right2.set_ylabel('Lake Pixel Count', fontsize=20)
    
    # Set y-ticks and y-tick labels for driver
    if driver == 'enso':
        values = np.arange(-3, 3.1, 1)
    elif driver == 'ipo':
        values = np.arange(-0.6, 0.7, 0.2)
    
    labels = [f"{v:.1f}" for v in values]
    ax_left.set_yticks(values)
    ax_left.set_yticklabels(labels, fontsize=20)
   
    # Set y-ticks and y-tick labels for rain
    ax_right1.set_yticks(np.arange(-1500,1501,500))
    ax_right1.set_yticklabels(list(np.arange(-1500,1501,500)), fontsize=20)

    # Set y-ticks and y-tick labels for dea
    ax_right2.set_yticks(np.arange(-750,751,250))
    ax_right2.set_yticklabels(list(np.arange(-750,751,250)), fontsize=20)
        
    ax_left.grid(True, which='major', axis='both') # Add grid to the background
    ax_left.xaxis.set_major_locator(mdates.YearLocator(grid_spacing))
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax_left.get_xticklabels(), rotation=90, ha='center', fontsize=20)
    

def plot_rolling_sum_with_driver_subplots(ds_1900, driver, driver_index, window, time_unit, threshold1, threshold2):
    ds_1987 = ds_1900.sel(time=slice("1987-01-01", ds_1900.time[-1]))

    # Figure
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(40, 20))  # 2 rows, 1 col
    ax1_2 = axes[0].twinx()  # secondary y-axis for first row
    ax1_3 = axes[0].twinx()  # secondary y-axis for first row
    ax2_2 = axes[1].twinx()  # secondary y-axis for second row
    ax2_3 = axes[1].twinx()  # secondary y-axis for second row
    
    # Plot using function
    plot_rolling_sum_with_driver(axes[0], ax1_2, ax1_3, ds_1987, start_date='1987', driver=driver,  driver_index=driver_index, window=window, time_unit=time_unit, threshold1=threshold1, threshold2=threshold2)
    plot_rolling_sum_with_driver(axes[1], ax2_2, ax2_3, ds_1900, start_date='1900', driver=driver,  driver_index=driver_index, window=window, time_unit=time_unit, threshold1=threshold1, threshold2=threshold2)
    
    # === Save figure ===
    plt.savefig(f'{output_dir}/Fig8_{Lake}_FloodEvents_{window}days_{driver}_{driver_index}.png', bbox_inches='tight')
    plt.close()


#### ========= Functions: Plot formattting =========
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
Lake_mask_r005_file = f'{Lake}_mask_r005.nc'
Lake_mask_r005_netcdf = xr.open_dataset(f'{input_dir}/{Lake_mask_r005_file}')
Lake_mask_r005 = Lake_mask_r005_netcdf['Mask']

# Finer Mask for plots: Duplicate each value 10 times in each direction for higher resolution
REPEAT = 10
Z = np.repeat(np.repeat(Lake_mask_r005.values, REPEAT, axis=0), REPEAT, axis=1)
new_lon = np.linspace(Lake_mask_r005['lon'].values[0], Lake_mask_r005['lon'].values[-1], Z.shape[1])
new_lat = np.linspace(Lake_mask_r005['lat'].values[0], Lake_mask_r005['lat'].values[-1], Z.shape[0])

# give dims names that won't collide with your main ds dims
lake_mask_fine_da = xr.DataArray(
    Z,
    coords={'lat': new_lat, 'lon': new_lon},
    dims=('lat', 'lon'),
    name='lake_mask_fine'
)


#-----------------------------------------------------------------------------------------------------------------------------------
### Load Gridded Daily
agcd_daily_file = f'agcd_v1_precip_total_r005_daily_{Lake}_1900to2024.nc'
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

#-----------------------------------------------------------------------------------------------------------------------------------
#Load Runoff
    
# Read runoff stations data from the netcdf file
runoff_stations_file = f'{input_dir}/{Lake}_daily_station_runoff.nc'
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

#-----------------------------------------------------------------------------------------------------------------------------------
### Load rainfall stations
rainfall_stations_file = f'{input_dir}/{Lake}_daily_station_rainfall.nc'
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

#-----------------------------------------------------------------------------------------------------------------------------------
### Load NAus
agcd_NT_daily_file = 'agcd_v1_precip_total_r005_daily_NAus_1900to2024_masked.nc'
agcd_NT_daily_ds = xr.open_dataset(f'{input_dir}/{agcd_NT_daily_file}')
agcd_NT_daily_ds['time'] = agcd_NT_daily_ds.indexes['time'].normalize()
agcd_NT_daily_da = agcd_NT_daily_ds['precip']

#-----------------------------------------------------------------------------------------------------------------------------------
### Load DEA
dea_file = f'{input_dir}/{Lake}_dea.nc'
dea_ds = xr.open_dataset(dea_file)

dea_da = dea_ds['lake_observations']
dea_da.name = 'dea'  # optional rename
lake_daily_1900_ds['dea'] = dea_da
lake_daily_1900_ds['dea'].attrs['long_name'] = "Daily lake extent and surface area"

#-----------------------------------------------------------------------------------------------------------------------------------
### Load Drivers
# Read in ENSO data
ENSO_noaa = read_enso_files()
ENSO_noaa_monthly_1900_da = ENSO_noaa.sel(time=slice(lake_daily_1900_ds.time.min(),lake_daily_1900_ds.time.max()))

# daily
extra_time = pd.Timestamp(ENSO_noaa_monthly_1900_da.time.max().values) + pd.DateOffset(months=1)
ENSO_noaa_monthly_1900_da_extended = ENSO_noaa_monthly_1900_da.reindex(time=list(ENSO_noaa_monthly_1900_da.time.values) + [np.datetime64(extra_time)])
ENSO_noaa_daily_1900_da = ENSO_noaa_monthly_1900_da_extended.resample(time='1D').ffill()
ENSO_noaa_daily_1900_da = ENSO_noaa_daily_1900_da.isel(time=slice(0, -1))
lake_daily_1900_ds['enso'] = ENSO_noaa_daily_1900_da
   
# Read in IPO data
IPOtripole_noaa = read_ipo_files()
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


### ========= Calculate offset =========
lake_daily_1900_ds['dea'] = lake_daily_1900_ds['dea'].assign_coords(dea_offset=calculate_event_offset_coord(lake_daily_1900_ds, 'daily'))


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

####  === Filter Event Cumulative Precip by Rate of Change ===
lake_daily_1900_ds['mean_gridded_rainfall_event_cum_filtered'] = filter_cumulative_rainfall(lake_daily_1900_ds)
    

#%% Analyse Data: Daily rolling cumulative dataset

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


#%% Fig 1. Map Plot: Catchment map with all stations

# Create a new figure with specified size and add a subplot with PlateCarree projection and coastlines
fig = plt.figure(figsize=(15, 17))
ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax1.coastlines()  # Add coastlines
ax1.set_facecolor(presentation_bom_colours[4])

# Fill lake mask
ax1.pcolormesh(Lake_mask_r005.coords['lon'].values, Lake_mask_r005.coords['lat'].values, Lake_mask_r005.values, cmap=ListedColormap(['white', 'white']), transform=ccrs.PlateCarree(), zorder=2)

# Plot lake masks as black contour lines
ax1.contour(lake_mask_fine_da.lon, lake_mask_fine_da.lat, lake_mask_fine_da.values, colors='k', linewidths=1, transform=ccrs.PlateCarree())

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
plt.savefig(f'{output_dir}/Fig1_{Lake}_map_stations.png', bbox_inches='tight')
plt.close()


#%% Fig 2. Timeseries Plot: DEA Flood Events (optional: with Rainfall / Runoff)

####  Plot Rainfall
plot_dea_timeseries(lake_daily_1987_ds, time_unit_lake='daily', timeseries_type='rainfall', time_unit_data='monthly', plot_mean=True, plot_station=True)

####  Plot Runoff
plot_dea_timeseries(lake_daily_1987_ds, time_unit_lake='daily', timeseries_type='runoff', time_unit_data='daily', plot_station=True)

# Flood Events with Rainfall Peaks before Event Max
plot_dea_timeseries(lake_daily_1987_ds, time_unit_lake='monthly', timeseries_type='rainfall', time_unit_data='monthly', plot_mean=True, peaks=True)
      

#%% Fig 3. Cumulative Plot: Cumulative Rainfall Offset from Event Maxima (Events only)

plot_cumulative_rainfall_all_events(lake_daily_1987_ds)
plot_cumulative_rainfall_all_events(lake_daily_1987_ds, magnitude=True)
plot_cumulative_rainfall_all_events(lake_daily_1987_ds, magnitude=True, filtered=True)


#%% Fig 4. Cumulative Plot: Cumulative Rainfall build-up before Rolling Sum Maximum (all years)
  
plot_cum_rainfall_with_threshold(ds=lake_daily_1900_ds, threshold=lower_threshold_daily, time_interval=f'{window_size_daily}_days')
plot_cum_rainfall_with_threshold(ds=lake_daily_1900_ds, threshold=lower_threshold_daily, time_interval=f'{window_size_daily}_days', ranked='True')
    

#%% Fig 5. Cumulative Plot: Optimal Rolling-Sum Duration and Threshold Analysis (all Years)

plot_threshold_analysis(lake_daily_1987_ds, window_max= 208, threshold=lower_threshold_daily, time_unit='days')

#%% Fig 6. Rolling-Sum Plot: Daily and Monthly Rolling-Sum Rainfall (set/ variable window)   

### PLOT: timeseries of daily rolling-sum rainfall ###
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(40, 20))  # 2 rows, 1 col

# Twin axes
ax1_2 = axes[0].twinx()  # secondary y-axis for first row
ax2_2 = axes[1].twinx()  # secondary y-axis for second row

# Plot using function
plot_rolling_sum_window(axes[0], ax1_2, lake_daily_1987_ds, start_date='1987', window=window_size_daily, time_unit='days', threshold1=lower_threshold_daily, threshold2=higher_threshold_daily)
plot_rolling_sum_window(axes[1], ax2_2, lake_daily_1900_ds, start_date='1900', window=window_size_daily, time_unit='days', threshold1=lower_threshold_daily, threshold2=higher_threshold_daily)

#Legend
axes[0].legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=20)

# === Save figure ===
plt.savefig(f'{output_dir}/Fig6_{Lake}_RollingSum_window-{window_size_daily}days.png', bbox_inches='tight')
plt.close()



### PLOT: timeseries of 80-224 day rolling-sum rainfall ###
lower_window = 80
higher_window = 224

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(40, 20))  # 2 rows, 1 col

# Twin axes
ax1_2 = axes[0].twinx()  # secondary y-axis for first row
ax2_2 = axes[1].twinx()  # secondary y-axis for second row

# Plot using function
plot_rolling_sum_variable_window(axes[0], ax1_2, lake_daily_1987_ds, start_date='1987', time_unit='days', threshold=lower_threshold_daily, window_min=lower_window, window_max=higher_window)      
plot_rolling_sum_variable_window(axes[1], ax2_2, lake_daily_1900_ds, start_date='1900', time_unit='days', threshold=lower_threshold_daily, window_min=lower_window, window_max=higher_window)      

#Legend
lines = axes[0].get_lines()  # list of all Line2D objects on this axis
last_two_lines = lines[-2:]
axes[0].legend(handles=last_two_lines, labels=[f'{lower_window}-{higher_window} Day Rolling Sum of Rainfall', f'Peak above {lower_threshold_daily} mm'], loc='upper left', bbox_to_anchor=(0, 1), fontsize=20)

# === Save figure ===
plt.savefig(f'{output_dir}/Fig6_{Lake}_RollingSum_window_{lower_window}-{higher_window}_days.png', bbox_inches='tight')
plt.close()


#%% Fig 7. Map Plot: NT Map 112 day sum Compostite
 
NT_daily_rainfall_window_sum_da = window_sum_vectorised(ds=lake_daily_1900_ds, da=agcd_NT_daily_da, window_size=window_size_daily)

#PLOT
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(30, 18), subplot_kw={"projection": ccrs.PlateCarree()})
axes = axes.flatten()

rows = ["1900\n–\n2024", "1900\n–\n1987", "1987\n–\n2024"]
columns = ["Non-Event Years", "Event Years"]

plot_args = [
    ('1900','2024',False),
    ('1900','2024',True),
    ('1900','1987',False),
    ('1900','1987',True),
    ('1987','2024',False),
    ('1987','2024',True)
]

for ax, (start, end, ev) in zip(axes, plot_args):
    im0 = plot_NT_event_map(ax, NT_daily_rainfall_window_sum_da, start_date=start, end_date=end, lake_mask=lake_mask_fine_da, event=ev)
        
# Colorbar
#fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])  # [left, bottom, width, height]
cbar_ax.set_frame_on(False)  # Hide the rectangle
cb1 = fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')
cb1.set_label(f'Rainfall ({lake_daily_1987_ds['gridded_rainfall'].units}/ 112 days)', fontsize=10)
cb1.set_ticks(np.linspace(0, 600, 7))
cb1.ax.tick_params(labelsize=10)
cb1.outline.set_color('lightgray')    
    
# Add row/column labels
for i, ax in enumerate(axes):
    r = i // 2
    c = i % 2
    if c == 0:  # leftmost column → row label
        ax.text(-0.05, 0.5, rows[r], transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='center', ha='right', multialignment='center')
    if r == 0:  # top row → column label
        ax.text(0.5, 1.05, columns[c], transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom', ha='center')

# Save figure
plt.savefig(f'{output_dir}/Fig7_LW_NAus_map.png', bbox_inches='tight')
plt.close()



#%% Fig 8. Driver Plot: 8.2 ENSO & IPO with Rolling Sum
    
# === ENSO ===
#Nino 3.4 (ERSST)
plot_rolling_sum_with_driver_subplots(lake_daily_1900_ds, driver='enso', driver_index='Nino 3.4 (ERSST)', window=window_size_daily, time_unit='days', threshold1=lower_threshold_daily, threshold2=higher_threshold_daily)

#Nino 3.4 (HadISST)
plot_rolling_sum_with_driver_subplots(lake_daily_1900_ds, driver='enso', driver_index='Nino 3.4 (HadISST)', window=window_size_daily, time_unit='days', threshold1=lower_threshold_daily, threshold2=higher_threshold_daily)

#Oni
plot_rolling_sum_with_driver_subplots(lake_daily_1900_ds, driver='enso', driver_index='ONI', window=window_size_daily, time_unit='days', threshold1=lower_threshold_daily, threshold2=higher_threshold_daily)


# === IPO ===
#HadISST
plot_rolling_sum_with_driver_subplots(lake_daily_1900_ds, driver='ipo', driver_index='HadISST 1.1', window=window_size_daily, time_unit='days', threshold1=lower_threshold_daily, threshold2=higher_threshold_daily)

#ERSST
plot_rolling_sum_with_driver_subplots(lake_daily_1900_ds, driver='ipo', driver_index='ERSST V5', window=window_size_daily, time_unit='days', threshold1=lower_threshold_daily, threshold2=higher_threshold_daily)

#COBE
plot_rolling_sum_with_driver_subplots(lake_daily_1900_ds, driver='ipo', driver_index='COBE', window=window_size_daily, time_unit='days', threshold1=lower_threshold_daily, threshold2=higher_threshold_daily)



