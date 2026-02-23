#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:      Lea Sophie Grunau  
Created on:  2025-07-15  
Last updated: 2026-02-23

Combine daily rainfall time series from multiple BOM rainfall stations
(for a single lake) into a single NetCDF file.

This script:
    • Loads metadata for all rainfall stations associated with a lake.
    • Reads individual daily rainfall CSVs (IDCJAC0009 format).
    • Aligns and merges the station data into one time-indexed DataFrame.
    • Converts the data to an xarray Dataset, attaching spatial metadata (lat/lon).
    • Saves the resulting Dataset as a NetCDF file.

Assumptions:
    • All station CSVs are stored in a subfolder named <Lake>_rainfall_stations.
    • Metadata file is named <Lake>_rainfall_stations.csv.
    • Input CSVs follow BOM's daily data format with "Year", "Month", "Day", and
      "Rainfall amount (millimetres)" columns.

Dependencies:
    - Python 3.x
    - Python libraries: argparse, sys, pathlib, pandas, xarray
    - netCDF4 (for xarray backend)
    - Files: {Lake}_rainfall_stations.csv, IDCJAC0009_*_1800_Data.csv

Usage:
    python 04_combine_rainfall_stations_to_netcdf.py --Lake <lake_short_code>
    e.g. python 04_combine_rainfall_stations_to_netcdf.py --Lake LW

Output:
    {Lake}_daily_station_rainfall.nc
"""

import pandas as pd
import xarray as xr
from pathlib import Path
import argparse
import sys


# ========== Parse command-line argument ==========
parser = argparse.ArgumentParser(description="Combine BOM station rainfall data for a lake into NetCDF")
parser.add_argument("--Lake", required=True, help="Lake short code (e.g. LE, LW, LG, LB)")
args = parser.parse_args()

Lake = args.Lake
data_dir = Path(__file__).parent.parent / 'data' / f'{Lake}_rainfall_stations'
stations_info_file = data_dir / f'{Lake}_rainfall_stations.csv'
output_file = data_dir / f'{Lake}_daily_station_rainfall.nc'


# ========== Check existence ==========
if not data_dir.is_dir():
    print(f"❌ Folder not found: {data_dir}")
    sys.exit(1)

if not stations_info_file.is_file():
    print(f"❌ Metadata CSV not found: {stations_info_file}")
    sys.exit(1)


# ========== Definitions ==========
def read_station_rainfall(station_rainfall_filepath, station_id):
    """
    Reads daily rainfall data for a single BOM station from CSV.

    Parameters:
        station_rainfall_filepath (Path): Path to the CSV file for the station.
        station_id (str): Station ID to use as column name in the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with datetime index and one column named after the station ID,
                      containing daily rainfall values in millimetres.
                      Returns None if file cannot be read.
    """
    try:
        df = pd.read_csv(
            station_rainfall_filepath,
            usecols=["Year", "Month", "Day", "Rainfall amount (millimetres)"]
        )
        # Combine Year, Month, Day into a single datetime column
        df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]], errors="coerce")

        # Drop original Y/M/D columns (optional)
        df.drop(columns=["Year", "Month", "Day"], inplace=True)
        
        # Rename rainfall column to the station ID
        df.rename(columns={"Rainfall amount (millimetres)": station_id}, inplace=True)
        
        # Set the datetime column as index
        df.set_index("Date", inplace=True)
        return df

    except Exception as e:
        print(f"⚠️ Failed to load {station_rainfall_filepath.name}: {e}")
        return None


# ========== Main workflow ==========

# Read in metadata for all stations
stations_info_df = pd.read_csv(stations_info_file, dtype={"Site": str})
stations_info_df.set_index("Site", inplace=True)

# Loop through each station to read its rainfall data and join into combined DataFrame
combined_station_rainfall_df = pd.DataFrame()

for station_id in stations_info_df.index:
    station_rainfall_file = Path(f"{data_dir}/IDCJAC0009_{station_id}_1800_Data.csv")
    if not station_rainfall_file.is_file():
        print(f"⚠️ File missing for station {station_id}")
        continue
    
    station_rainfall_df = read_station_rainfall(station_rainfall_file, station_id)
    if station_rainfall_df is not None:
        combined_station_rainfall_df = combined_station_rainfall_df.join(station_rainfall_df, how="outer")


# Convert combined pandas DataFrame to xarray Dataset with coordinates and metadata
combined_station_rainfall_df.index.name = "time"

combined_station_rainfall_da = xr.DataArray(
    data = combined_station_rainfall_df.values,
    dims = ("time", "rainfall_station"),
    coords = {
        "time": combined_station_rainfall_df.index,
        "rainfall_station": combined_station_rainfall_df.columns.astype(str),
        "rainfall_station_lat": ("rainfall_station", stations_info_df.loc[combined_station_rainfall_df.columns.astype(str), "Lat"].astype(float).values),
        "rainfall_station_lon": ("rainfall_station",  stations_info_df.loc[combined_station_rainfall_df.columns.astype(str), "Lon"].astype(float).values),
        "rainfall_station_name": ("rainfall_station", stations_info_df.loc[combined_station_rainfall_df.columns.astype(str), "Site name"].values),
    },
    attrs = {
        "units": "mm",
        "source": "IDCJAC0009 product",
        "description": f"Daily rainfall values for each station of {Lake}",
    }
)

combined_station_rainfall_ds = xr.Dataset(
    data_vars={"rainfall": combined_station_rainfall_da},
    attrs={
        "title": f"Rainfall station dataset for {Lake}",
        "created_by": "combine_rainfall_stations_to_netcdf.py"
    }
)


# ========== Save ==========
combined_station_rainfall_ds.to_netcdf(output_file, engine="netcdf4")
print(f"✅ Processed {len(combined_station_rainfall_df.columns)} stations and saved NetCDF: {output_file.name}")

