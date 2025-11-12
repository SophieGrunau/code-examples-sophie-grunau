#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author:      Lea Sophie Grunau  
Created on:  2025-07-07 

Converts mask r001 mask (grid of 0.01) to r005 mask (grid of 0.05).

This script:
    • ...

Dependencies:
   - Python 3.x
   - netCDF4 (for xarray backend)
   - python libraries: argparse, sys, pathlib, xarray, numpy
   - files: {Lake}_mask_r001.nc
 Note: Paths must be adapted to your system.

Usage:
    python convert_r001_mask_to_r005.py --Lake <lake_short_code>
    e.g. python convert_r001_mask_to_r005.py --Lake LW

"""

import argparse
import sys
from pathlib import Path
import xarray as xr
import numpy as np

# ========== Parse command-line argument ==========
parser = argparse.ArgumentParser(description="Convert mask from r001 to r005")
parser.add_argument("--Lake", required=True, help="Lake name short form (e.g. LE, LW, LG, LB)")
args = parser.parse_args()

Lake = args.Lake
grid = "r001"
input_dir = "/Users/leasophiegrunau/Documents/Work/Bewerbungen/code-examples-sophie-grunau/pre_processing/convert_lake_masks"
Lake_mask_r001_file = f"{Lake}_mask_r001"
Lake_mask_r001_path = Path(f"{input_dir}/{Lake_mask_r001_file}.nc")


# ========== Check if file exists ==========
if not Lake_mask_r001_path.is_file():
    print(f"❌ File not found: {Lake_mask_r001_path}")
    sys.exit(1)


# ========== Definitions ==========
def pad_dimension_until_005_aligned(data, dim):
    """
    Pad an xarray DataArray so that the edges align with the 0.05° r005 grid.

    For r001 data, this function adds rows or columns of zeros to the specified dimension
    (lat or lon) so that when converted to r005 (via 5x5 aggregation), the grid aligns with 
    0.05° increments (e.g., 133.05, -35.10).

    Parameters:
        data (xr.DataArray): The input binary mask.
        dim (str): The dimension to pad ('lat' or 'lon').

    Returns:
        xr.DataArray: The padded array, aligned for r005 grid conversion.
    """

    coords = data.coords[dim].values  # Coordinate values along the specified dimension
    step = round(np.abs(coords[1] - coords[0]),2)  # Grid resolution (spacing between points)
    edge_start = coords[0]               # First coordinate value
    edge_end = coords[-1]                # Last coordinate value

    def needs_padding(val, align=0.05):
        # Returns True if val is NOT divisible evenly by 0.05
        return int(round(val * 100)) % 5 != 0

    padded = data
    padded_start_count = 0
    padded_end_count = 0
    
    # Adjusted checks for alignment (grid point is centered, not edge-aligned, i.e grid centre is at .00 or .05)
    edge_start_corrected = edge_start - 0.03 # → offset needed to align start to lon x.03/x.08 lat -x.02/-x.07
    edge_end_corrected = edge_end - 0.02 #→ offset needed to align start to lon x.02/x.07 lat -x.03/-x.08

    # Pad the beginning (until lon = x.03/ x.08 or lat = -x.02 / -x.07 becomes aligned)
    if needs_padding(edge_start_corrected): 
        while needs_padding(padded.coords[dim].values[0] - 0.03): 
            zeros = xr.zeros_like(padded.isel({dim: 0})) #takes a slice of the dataset and creates a new DataArray of zeros with the same shape + attributes 
            zeros.coords[dim] = padded.coords[dim].values[0] - step #updates the dimension lat/lon (new coordinate is one step before the current first one)
            padded = xr.concat([zeros, padded], dim=dim) #concatenates the new row of zeros in front of the current data array along the given dimension
            padded_start_count += 1
            
    # Pad the end (until lon = x.02/ x.07 or lat = -x.03/ -x.08 becomes aligned)
    if needs_padding(edge_end_corrected): 
        while needs_padding(padded.coords[dim].values[-1] - 0.02):
            zeros = xr.zeros_like(padded.isel({dim: -1}))
            zeros.coords[dim] = padded.coords[dim].values[-1] + step #updates the dimension lat/lon (new coordinate is one step after the current first one)
            padded = xr.concat([padded, zeros], dim=dim) #concatenates the new row of zeros behind the current data array along the given dimension
            padded_end_count += 1

    print(f"Padded {dim}: {padded_start_count} at start, {padded_end_count} at end")
    return padded


def pad_mask_until_005_aligned(data):
    """
    Pad a 2D mask along both latitude and longitude dimensions 
    so that the resulting grid aligns with a 0.05° resolution.

    This function applies padding using the `pad_dimension_until_005_aligned` 
    logic to both the 'lat' and 'lon' dimensions.

    Parameters:
        data (xr.DataArray or xr.Dataset): The input binary mask to be padded.

    Returns:
        xr.DataArray or xr.Dataset: The padded mask aligned to 0.05° grid boundaries.
    """
    latitude_padded = pad_dimension_until_005_aligned(data, 'lat')
    return pad_dimension_until_005_aligned(latitude_padded, 'lon')



def convert_xarray_to_r005(input_xarray):
    """
    Convert an xarray dataset from 0.01° (r001) to 0.05° (r005) resolution 
    by aggregating values in a 5×5 grid.

    This is typically used to coarsen a binary mask (0s and 1s), where
    each 5×5 block is summed to indicate how many high-res cells overlap
    the coarser grid cell.

    Parameters:
        input_xarray (xr.DataArray): Input xarray array on a 0.01° grid (r001).

    Returns:
        xr.DataArray: Aggregated xarray array on a 0.05° grid (r005).

    Notes:
        ⚠️ The input data must be padded on all sides with at least 2 rows/columns 
        so that its shape is divisible by 5 in both lat and lon. This is critical 
        to ensure proper alignment of the resulting r005 grid.
        Use `pad_mask_until_005_aligned()` beforehand to ensure this.
    """

    # Step 1: extract values
    values_new = input_xarray.values # Get the values from the input xarray
    
    # Step 2: aggregate values in 5x5 blocks (sum across lon then lat)
    aggregated_values = np.add.reduceat(values_new, np.arange(0, len(values_new[0]), 5), axis=1) #sums every 5 columns (i.e., across longitude) for each row
    aggregated_values = np.add.reduceat(aggregated_values, np.arange(0, len(aggregated_values), 5), axis=0) #sums every 5 rows (i.e., across latitude) on the already column-reduced data

    # Step 3: define new grid coordinates by selecting every 5th point (skipping the first and last two rows/columns because grid points represent cell centers, not edges)
    xarray_r005 = input_xarray[2:-2:5, 2:-2:5] #creats a “dummy” xarray structure with the right coordinate grid
    
    # Step 4: apply threshold: >12 → 1, ≤12 → 0 (If more than half the r001 pixels in a coarse cell are 1s, then it counts as a 1.)
    xarray_r005.values = (aggregated_values > 12).astype(np.int32)
    
    return xarray_r005



# ========== Load, process, save ==========
Lake_mask_r001_netcdf = xr.open_dataset(Lake_mask_r001_path)
Lake_mask_r001 = Lake_mask_r001_netcdf['Mask']

Lake_mask_aligned_to_r005 = pad_mask_until_005_aligned(Lake_mask_r001)
Lake_mask_r005 = convert_xarray_to_r005(Lake_mask_aligned_to_r005)

Lake_mask_r005.to_netcdf(f"{input_dir}/{Lake}_mask_r005.nc")
#Lake_mask_aligned_to_r005.to_netcdf(f"{input_dir}/{Lake}_mask_r001_padded.nc")

print(f"✅ Mask converted to r005 and saved: {Lake}_mask_r005.nc")




