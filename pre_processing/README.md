
# Rainfall Station Data Processing Pipeline
**Author:** Lea Sophie Grunau

Automated pipeline for filtering, downloading, and combining Bureau of Meteorology rainfall station data into a unified NetCDF file.

## Pipeline Overview

This is a **4-step sequential pipeline** that must be run in order:
```
01_filter → 02_move → 03_identify → 04_combine
```

## Usage

Run each script in sequence:
```bash
# Step 1: Filter stations and open download webpages
python 01_filter_rainfall_stations_by_mask.py --Lake LW

# Step 2: Move downloaded CSV files to data directory
bash 02_move_rainfall_station_files.sh LW

# Step 3: Identify any missing station files
python 03_identify_missing_rainfall_stations.py --Lake LW

# Step 4: Combine all station CSVs into single NetCDF
python 04_combine_rainfall_stations_to_netcdf.py --Lake LW
```

## What Each Step Does

**Step 1: `01_filter_rainfall_stations_by_mask.py`**
- Spatially filters BOM stations to those within the lake catchment
- Opens webpages for manual data download (with batching and delay)
- Validates station data availability via HTTP checks
- Outputs: 
  - `{Lake}_rainfall_stations.csv` (valid stations)
  - `{Lake}_rainfall_stations_missed.csv` (failed stations, if any)

**Step 2: `02_move_rainfall_station_files.sh`**
- Moves downloaded BOM CSV files from Downloads to data directory
- Organizes files for processing
- **Note:** Manual download required in Step 1 before running this

**Step 3: `03_identify_missing_rainfall_stations.py`**
- Checks for missing or incomplete station download
- Quality control step
- Outputs: List of missing station IDs (for retry)

**Step 4: `04_combine_rainfall_stations_to_netcdf.py`**
- Reads individual BOM CSVs (IDCJAC0009 format)
- Aligns all stations to common time axis
- Attaches spatial metadata (lat/lon)
- Outputs: `{Lake}_daily_station_rainfall.nc`

## Requirements

**Python:**
- Python 3.x
- xarray, pandas, numpy, requests, webbrowser
- NetCDF4

**Input Files:**
- `{Lake}_mask_r001.nc` - Lake catchment mask
- `rainfall_stations.txt` - BOM station metadata (fixed-width format)

**System:**
- Web browser for manual data download in Step 1
- Paths in scripts must be adapted to your system

## Notes

- **Step 1** opens BOM webpages for manual download - be prepared to download multiple CSV files
- **Step 2** requires manual completion of downloads before running
- **Step 3** is optional QC - use if downloads incomplete
- All station CSVs must be in IDCJAC0009 format (BOM daily rainfall)

## Output

Final output: `{Lake}_daily_station_rainfall.nc` containing all catchment stations with aligned daily time series.