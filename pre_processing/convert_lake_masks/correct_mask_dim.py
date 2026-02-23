#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correct Lake Mask Dimensions and Coordinate Order

Author:      Lea Sophie Grunau  
Created on:  2025-07-07
Last updated: 2026-02-23

Description:
    Corrects dimensional inconsistencies in lake mask NetCDF files by ensuring
    coordinates are in increasing order and dimensions follow standard conventions.
    Also converts mask to binary format and rounds coordinates for consistency.

This script:
    • Reverses coordinate axes if they are in decreasing order
    • Reorders dimensions to standard order (time, lat, lon)
    • Converts mask values to binary integer format (0/1)
    • Rounds lat/lon coordinates to 2 decimal places
    • Saves corrected mask with '_reversed' suffix

Dependencies:
    - Python 3.x
    - xarray, numpy
    - netCDF4 (for xarray backend)
    - Input file: {Lake}_mask_r001.nc

Usage:
    python correct_mask_dim.py --Lake <lake_short_code>
    
    Example:
        python correct_mask_dim.py --Lake LW

Output:
    {Lake}_mask_r001_reversed.nc
    
Common Issues Fixed:
    - Southern hemisphere data with lat coordinates in decreasing order
    - Non-standard dimension ordering (e.g., lon-lat-time instead of time-lat-lon)
    - Float precision issues in coordinate values
    - Non-binary mask values
"""

import argparse
import sys
from pathlib import Path
import xarray as xr
import numpy as np

# ========== Parse command-line argument ==========
parser = argparse.ArgumentParser(description="Reverse latitude and convert mask to binary.")
parser.add_argument("--Lake", required=True, help="Lake name short form (e.g. LE, LW, LG, LB)")
args = parser.parse_args()

Lake = args.Lake
input_dir = Path(__file__).parent.parent
data_dir = input_dir / 'data'
Lake_mask_file = f"{Lake}_mask_r001"
Lake_mask_path = data_dir / f"{Lake_mask_file}.nc"


# ========== Check if file exists ==========
if not Lake_mask_path.is_file():
    print(f"❌ File not found: {Lake_mask_path}")
    sys.exit(1)


# ========== Definitions ==========

# Reverse Dimensions if needed
def ensure_dim_increasing(xarray_data):
    """
    Ensure that the dimensions in the given xarray dataset or array
    are in increasing order. If not, reverse them.

    Parameters:
        xarray_data (xr.Dataset or xr.DataArray): Input xarray object.

    Returns:
        xr.Dataset or xr.DataArray: Object with dimensions reversed if needed.
    """

    for dim in xarray_data.dims:
        coord = xarray_data[dim].values
        if coord[0] > coord[-1]:   # decreasing order
            xarray_data = xarray_data.isel(**{dim: slice(None, None, -1)})
    return xarray_data    

# Re-order Dimensions if needed
def ensure_dim_order(xarray_data, target_order=('time', 'lat', 'lon')):
    """
    Reorder dimensions in an xarray object to match a preferred order (if dimensions exist).
    """
    preferred = [dim for dim in target_order if dim in xarray_data.dims]
    others = [dim for dim in xarray_data.dims if dim not in preferred]
    final_order = preferred + others

    if list(xarray_data.dims) != final_order:
        xarray_data = xarray_data.transpose(*final_order)

    return xarray_data    

# ========== Load, process, save ==========
Lake_mask_netcdf = xr.open_dataset(Lake_mask_path)
Lake_mask = Lake_mask_netcdf['Mask']

# Ensure increasing coordinates and standard order
Lake_mask_correctedDim = ensure_dim_increasing(Lake_mask)
Lake_mask_correctedDim_order = ensure_dim_order(Lake_mask_correctedDim)

# Convert to binary and round coordinates
Lake_mask_binary = Lake_mask_correctedDim_order.astype(np.int32)
Lake_mask_binary.coords['lat'] = Lake_mask_binary.coords['lat'].round(2)
Lake_mask_binary.coords['lon'] = Lake_mask_binary.coords['lon'].round(2)

Lake_mask_binary.to_netcdf(data_dir / f"{Lake_mask_file}_reversed.nc")


print(f"✅ Mask reversed and saved: {Lake_mask_file}_reversed.nc")


