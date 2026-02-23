
# Pre-Processing Pipeline
**Author:** Lea Sophie Grunau

Pre-processing tools for lake mask correction and rainfall station data processing.
The two subfolders are independent of each other but both use `{Lake}_mask_r001.nc` as input,
which should be placed in the `data/{Lake}_rainfall_stations/` folder.

## Folder Structure
```
pre_processing/
├── combine_rainfall_stations/   # Filter, download, and combine BOM rainfall station data
├── convert_lake_masks/          # Correct lake mask dimensions and resolution
└── data/
    └── {Lake}_rainfall_stations/
```

## convert_lake_masks
Two standalone scripts for correcting lake masks exported from ArcGIS:

**`correct_mask_dim.py`** — Corrects dimensional inconsistencies (e.g. reversed latitude, wrong dimension order). Run this first if your mask has incorrect dimensions.
```bash
python correct_mask_dim.py --Lake LW
```

**`convert_r001_mask_to_r005.py`** — Converts a mask from 0.01° (r001) to 0.05° (r005) resolution.
```bash
python convert_r001_mask_to_r005.py --Lake LW
```

## combine_rainfall_stations
A 4-step sequential pipeline for processing BOM rainfall station data:
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

## Notes
- **Step 1** opens BOM webpages for manual download - be prepared to download multiple CSV files
- **Step 2** requires manual completion of downloads before running
- **Step 3** is optional QC - use if downloads incomplete
- All station CSVs must be in IDCJAC0009 format (BOM daily rainfall)

## Output
Final output: `{Lake}_daily_station_rainfall.nc` containing all catchment stations with aligned daily time series.