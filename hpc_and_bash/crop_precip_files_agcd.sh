#!/bin/bash

# ========== Parse and validate arguments ==========
# Parse arguments like grid=r001 freq=daily region=LE
for arg in "$@"; do
  case $arg in
    grid=*)
      grid="${arg#*=}"
      ;;
    freq=*)
      freq="${arg#*=}"
      ;;
    region=*)
      region="${arg#*=}"
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

# ========== Validate input ==========
# Check that both arguments were provided
if [[ -z "$grid" || -z "$region" ]]; then
  echo "Usage: $0 grid=r001|r005 freq=daily|monthly region=LE|NT|..."
  exit 1
fi

# Validate grid value
if [[ ! "$grid" =~ ^(r001|r005)$ ]]; then
  echo "Invalid grid: $grid"
  echo "Must be 'r001' or 'r005'"
  exit 1
fi

# Validate region value
if [[ ! "$region" =~ ^(LE|LB|LW|LG|NT)$ ]]; then
  echo "Invalid region: $region"
  echo "Must be one of: LE, LB, LW, LG, NT"
  exit 1
fi

# Set version, enforce freq=monthly for r001 & Validate freq value for r005
if [[ "$grid" == "r001" ]]; then
  version="v2-0-3"
  freq="monthly"
else
  version="v1-0-3"
  if [[ -z "$freq" ]]; then
    echo "Missing frequency for r005: must be 'daily' or 'monthly'"
    exit 1
  fi
  if [[ ! "$freq" =~ ^(daily|monthly)$ ]]; then
    echo "Invalid frequency: $freq"
    echo "Must be 'daily' or 'monthly'"
    exit 1
  fi
fi

echo "Grid: $grid, Version: $version, Frequency: $freq, Region: $region"


# ========== Load modules ==========
module load nco
module load netcdf


# ========== Set paths ==========
if [ "$freq" == "daily" ]; then
  freq_path="01day"
else
  freq_path="01month"
fi

# File paths and naming
input_dir="/g/data/zv2/agcd/${version}/precip/total/${grid}/${freq_path}"
scratch_dir="/scratch/iz13/sg7350/precip_files/${region}_precip"
output_dir="/g/data/iz13/precip_files/${region}_precip"

file_prefix_in="agcd_${version%%-*}_precip_total_${grid}_${freq}"
file_prefix_crop="agcd_${version%%-*}_precip_total_${grid}_${freq}_${region}"
output_file_year=${scratch_dir}/${file_prefix_crop}_????.nc

echo "Input directory: $input_dir"
echo "Scratch directory: $scratch_dir"
echo "Output directory: $output_dir"

# Create output directory if needed
mkdir -p "$output_dir"
mkdir -p "$scratch_dir"

# ========== Set start & end year ==========
start_year=1900
end_year=2024

# ========== Set region boundaries ==========
case "$region" in
  LE)
    lon_min=133.322; lon_max=146.778
    lat_min=-33.278; lat_max=-18.772
    ;;
  LB)
    lon_min=145.572; lon_max=146.128
    lat_min=-21.928; lat_max=-21.122
    ;;
  LW)
    lon_min=132.922; lon_max=135.428
    lat_min=-19.028; lat_max=-16.372
    ;;
  LG)
    lon_min=149.322; lon_max=149.628
    lat_min=-35.478; lat_max=-34.7727
    ;;
  NT)
    lon_min=129.0; lon_max=138.0
    lat_min=-25.0; lat_max=-16.0
      ;;
esac

echo "Using region boundaries: Longitude: $lon_min to $lon_max"
echo "Using region boundaries: Latitude:  $lat_min to $lat_max"


# ========== Crop loop ==========
echo "Cropping files to boundaries of $region"
for year in $(seq $start_year $end_year); do
  input_file="${input_dir}/${file_prefix_in}_${year}.nc"
  output_file="${scratch_dir}/${file_prefix_crop}_${year}.nc"

  if [ -f "$input_file" ]; then
    ncks -d lon,$lon_min,$lon_max -d lat,$lat_min,$lat_max "$input_file" "$output_file"
  else
    echo "Unable to crop: File not found: $input_file"
  fi
done


# ========== Verify all yearly files are present ==========
echo "Checking for missing files before concatenation..."

missing=0
for year in $(seq $start_year $end_year); do
  file="${scratch_dir}/${file_prefix_crop}_${year}.nc"
  if [ ! -f "$file" ]; then
    echo "Missing: $file"
    missing=$((missing + 1))
  fi
done

if [ "$missing" -ne 0 ]; then
  echo "Found $missing missing file(s). Skipping concatenation."
#  rm $output_file_year
  exit 1
else
  echo "All expected files found (${start_year}–${end_year}). Proceeding to concatenate..."
fi

# ========== Concatenate all yearly files ==========
combined_file="${output_dir}/${file_prefix_crop}_${start_year}to${end_year}.nc"
ncrcat ${scratch_dir}/${file_prefix_crop}_????.nc "$combined_file"

if [ -f "$combined_file" ]; then
  echo "Combined file created: $combined_file"
else
  echo "Combined file not found."
fi

# ========== Delete individual files  ==========
#rm $output_file_year
#echo "Individual files removed."

  
  
  