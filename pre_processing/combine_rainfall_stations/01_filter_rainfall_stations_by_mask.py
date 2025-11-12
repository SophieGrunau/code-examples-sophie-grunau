#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:      Lea Sophie Grunau  
Created on:  2025-07-14 

Filter BOM rainfall station metadata spatially using a lake mask and open the 
corresponding webpages for manual data download.

This script:
    • Reads all BOM rainfall station metadata from a fixed-width text file or
      from a "missed stations" CSV if it exists (for partial retries).
    • Loads a lake-specific spatial mask from a NetCDF file.
    • Applies a bounding-box and interpolation-based spatial filter to select
      stations within the lake catchment.
    • Checks which stations have valid data pages (based on HTTP response + content).
    • Opens the webpages for valid stations (with batching and optional delay).
    • Saves a CSV file listing valid stations.
    • If any stations fail to open or are invalid, saves them as a separate
      "missed stations" file for retry.

Dependencies:
   - Python 3.x
   - netCDF4 (for xarray backend)
   - python libraries: argparse, sys, pathlib, pandas, xarray, webbrowser, time, requests, datetime
   - files: {Lake}_mask_r001.nc, rainfall_stations.txt
 Note: Paths must be adapted to your system.

Usage:
    python 01_filter_rainfall_stations_by_mask.py --Lake <lake_short_code>
    e.g. 01_python filter_rainfall_stations_by_mask.py --Lake LW

"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import xarray as xr
import webbrowser
import time
import requests
from datetime import datetime


# ========== Parse command-line argument ==========
parser = argparse.ArgumentParser(description="Returns csv of relevant rainfall stations and opens their webpages")
parser.add_argument("--Lake", required=True, help="Lake name short form (e.g. LE, LW, LG, LB)")
args = parser.parse_args()

Lake = args.Lake

# Paths to necessary input files based on lake name
input_dir = "/Users/leasophiegrunau/Documents/Work/Bewerbungen/code-examples-sophie-grunau/pre_processing/combine_rainfall_stations"
output_dir = Path(f"/Users/leasophiegrunau/Documents/Work/Bewerbungen/code-examples-sophie-grunau/pre_processing/combine_rainfall_stations/{Lake}_rainfall_stations")
Lake_mask_path = Path(f"{input_dir}/{Lake}_mask_r001.nc")
rainfall_stations_path = Path(f'{input_dir}/rainfall_stations.txt')
missed_rainfall_stations_path = Path(f'{output_dir}/{Lake}_missed_rainfall_stations.csv')

# Create the output directory (and parent dirs if needed)
output_dir.mkdir(parents=True, exist_ok=True)


# ========== Definitions ==========
def filter_stations_within_mask(stations_df, lake_mask, lat_col="Lat", lon_col="Lon"):
    """
    Filters a DataFrame of station metadata to include only those
    within the True/1 region of an xarray spatial mask.

    Parameters:
        stations_df (pd.DataFrame): DataFrame with station list and lat/lon columns.
        lake_mask (xr.DataArray): Boolean or 0/1 mask with lat/lon dimensions.
        lat_col (str): Name of the latitude column in the DataFrame (default: "Lat").
        lon_col (str): Name of the longitude column in the DataFrame (default: "Lon").

    Returns:
        pd.DataFrame: Filtered DataFrame of stations within the mask.
    """

    # ========== 1. Bounding box filter (cheap coarse filter) ==========
    # Determine grid spacing assuming regular grid
    lat_spacing = abs(lake_mask.lat.values[1] - lake_mask.lat.values[0]).round(2)
    lon_spacing = abs(lake_mask.lon.values[1] - lake_mask.lon.values[0]).round(2)

    # Compute true extent of the mask area (since grid is cell-centered)
    lat_min = float(lake_mask.lat.min() - lat_spacing / 2)
    lat_max = float(lake_mask.lat.max() + lat_spacing / 2)
    lon_min = float(lake_mask.lon.min() - lon_spacing / 2)
    lon_max = float(lake_mask.lon.max() + lon_spacing / 2)

    # Only keep stations that fall within the bounding box of the mask
    spatial_df = stations_df[
        (stations_df[lat_col] >= lat_min) & (stations_df[lat_col] <= lat_max) &
        (stations_df[lon_col] >= lon_min) & (stations_df[lon_col] <= lon_max)
    ].copy()

    # If nothing passed the coarse bounding box filter, return an empty DataFrame early
    if spatial_df.empty:
        return spatial_df

    # ========== 2. Mask value check via interpolation ==========
    # Interpolate mask values at each station's lat/lon position
    # This gives fractional values (e.g. 0.0, 0.5, 1.0) if station lies near grid boundaries
    mask_values = lake_mask.interp(
        lat=xr.DataArray(spatial_df[lat_col].values, dims="points"),
        lon=xr.DataArray(spatial_df[lon_col].values, dims="points")
    )

    # Store interpolated mask values in the DataFrame
    spatial_df["mask_value"] = mask_values.values

    # ========== 3. Final selection ==========
    # Keep only rows where the interpolated mask value is greater than 0.5 (i.e., "inside" the mask)
    # Drop the temporary 'mask_value' column afterwards
    return spatial_df[spatial_df["mask_value"] > 0.5].drop(columns=["mask_value"]).reset_index(drop=True)


def filter_valid_stations(stations_df, site_col="Site"):
    """
    Filter a DataFrame to keep only stations whose BOM daily data page exists.

    Parameters:
        stations_df (pd.DataFrame): DataFrame with station IDs.
        site_col (str): Name of the station ID column in stations_df.

    Returns:
        pd.DataFrame: Filtered DataFrame with only valid stations.
    """
    def page_exists(station_id):
        url = (
            "https://reg.bom.gov.au/jsp/ncc/cdio/weatherData/av"
            "?p_nccObsCode=136"
            "&p_display_type=dailyDataFile"
            "&p_startYear="
            "&p_c="
            f"&p_stn_num={station_id}"
        )
        try:
            # Use GET to fetch the full page content (required to check for "no data available" message)
            response = requests.get(url, allow_redirects=True, timeout=5)
            if response.status_code != 200:
                # Page does not exist or server error
                return False
            
            # Check if the "no data available" message is present in page text (case insensitive)
            if "no data available" in response.text.lower():
                return False
            
            # Page exists and likely contains data
            return True

        except requests.RequestException:
            # Network or request error, treat as invalid
            return False
    
    # Apply the page check to each station ID, build boolean mask
    mask = stations_df[site_col].apply(page_exists)
    
    # Return filtered DataFrame with only valid stations, reset index
    return stations_df[mask].reset_index(drop=True)

        
def open_station_pages(stations_df, site_col="Site", delay=1, batch_size=None, pause_between_batches=5):
    """
    Open BOM daily data pages for stations in your DataFrame.

    Parameters:
        stations_df (pd.DataFrame): DataFrame containing station IDs.
        site_col (str): Name of the column with station IDs (default "Site").
        delay (float): Seconds to wait between opening each tab (default 1 second).
        batch_size (int or None): Number of tabs to open before pausing. If None, no batching.
        pause_between_batches (float): Seconds to pause between batches if batch_size is set.
    """
    total = len(stations_df)
    for i, station_id in enumerate(stations_df[site_col], 1):
        url = (
            "https://reg.bom.gov.au/jsp/ncc/cdio/weatherData/av"
            "?p_nccObsCode=136"
            "&p_display_type=dailyDataFile"
            "&p_startYear="
            "&p_c="
            f"&p_stn_num={station_id}"
        )
        webbrowser.open_new_tab(url)
        time.sleep(delay)

        if batch_size and i % batch_size == 0 and i != total:
            pause_between_batches_minutes = pause_between_batches/60
            print(f"Opened {i} tabs. Pausing for {pause_between_batches_minutes:.1f} min... ({datetime.now().strftime('%H:%M:%S')})")
            time.sleep(pause_between_batches)

    print("✅ Done opening all station pages.")

        

# ========== Main workflow ==========

# Decide which file to use & if they exist
if missed_rainfall_stations_path.is_file():
    print(f"🔁 Using missed stations file: {missed_rainfall_stations_path.name}")
    lake_rainfall_stations = pd.read_csv(missed_rainfall_stations_path, names=["Site"])
    
    # Open the BOM pages for the filtered stations with controlled tab opening
    open_station_pages(lake_rainfall_stations, delay=1, batch_size=50, pause_between_batches=900)
    
    print("✅ Missing station webpages opened.")
    
    # Remove the missed file after successful use
    try:
        missed_rainfall_stations_path.unlink()
        print(f"🗑️ Deleted missed station file: {missed_rainfall_stations_path.name}")
    except Exception as e:
        print(f"⚠️ Could not delete missed file: {e}")
    
else:
    print("📋 Using full station list.")    
    if not Lake_mask_path.is_file():
        print(f"❌ File not found: {Lake_mask_path}")
        sys.exit(1)
    
    if not rainfall_stations_path.is_file():
        print(f"❌ File not found: {rainfall_stations_path}")
        sys.exit(1)
        
    # Read rainfall stations data from fixed-width formatted text file
    rainfall_stations_df = pd.read_fwf(rainfall_stations_path, skiprows=[0, 1, 3], skipfooter=5, engine="python")
    rainfall_stations_df = rainfall_stations_df[["Site", "Site name", "Start", "End", "Lat", "Lon"]]
        
    # Load mask data
    Lake_mask_netcdf = xr.open_dataset(Lake_mask_path)
    Lake_mask = Lake_mask_netcdf['Mask']
    
    # Filter stations spatially within the lake mask
    lake_rainfall_stations = filter_stations_within_mask(rainfall_stations_df, Lake_mask)
    
    # Filter stations to only those with valid BOM daily data pages
    lake_rainfall_stations = filter_valid_stations(lake_rainfall_stations)
    
    # Save filtered stations to CSV for record keeping
    lake_rainfall_stations.to_csv(f'{output_dir}/{Lake}_rainfall_stations.csv', index=False)
    
    # Open the BOM pages for the filtered stations with controlled tab opening
    open_station_pages(lake_rainfall_stations, delay=1, batch_size=50, pause_between_batches=900)
    
    print(f"✅ Station webpages opened and list of rainfall station in the catchment saved: {Lake}_rainfall_stations.csv")
    