# 02_move_rainfall_station_files.sh
# Author: Sophie Grunau
# Dependencies: 
#   - CSV file: ${LAKE}_rainfall_stations.csv"
#   - bash
# Notes:
#   - Paths for Data_dir and Downloads_dir must be adapted to your system.
#   - Make sure the rainfall station CSV exists and has the correct format.
# Usage: ./02_move_rainfall_station_files.sh <LakeShortCode>
# This script moves all BOM rainfall station csv files from the Downloads folder to your data folder.


#!/bin/bash

# ========== Inputs ==========
LAKE=$1
if [ -z "$LAKE" ]; then
    echo "❌ Please provide a lake short code (e.g. LE, LW, LG)"
    exit 1
fi

# Define paths
Data_dir="/Users/leasophiegrunau/Documents/Work/Bewerbungen/code-examples-sophie-grunau/pre_processing/combine_rainfall_stations/${LAKE}_rainfall_stations"
Downloads_dir="/Users/leasophiegrunau/Downloads"
csv_file="$Data_dir/${LAKE}_rainfall_stations.csv"
failed_log="$Data_dir/${LAKE}_missed_rainfall_stations.csv"

# ========== Check for CSV ==========
if [ ! -f "$csv_file" ]; then
    echo "❌ CSV file not found: $csv_file"
    exit 1
fi

# Create destination folder if it doesn't exist
mkdir -p "$Data_dir"
> "$failed_log"  # Empty or create the file before starting

# ========== Get station IDs from CSV ==========
station_ids=$(tail -n +2 "$csv_file" | cut -d',' -f1)

# ========== Loop through and move files ==========
for station_id in $station_ids; do
    # Construct subfolder and filenames using full station ID (no trimming)
    station_folder="$Downloads_dir/IDCJAC0009_${station_id}_1800"
    station_csv_pattern="IDCJAC0009_${station_id}_1800_Data.csv"
    station_txt_pattern="IDCJAC0009_${station_id}_1800_Note.txt"

    # Check if folder exists
    if [ -d "$station_folder" ]; then
        # Move .csv file to output directory
        if [ -f "$station_folder/$station_csv_pattern" ]; then
            mv "$station_folder/$station_csv_pattern" "$Data_dir/"
        else
            echo "$station_id" >> "$failed_log"
        fi

        # Delete .txt file (optional: or move it somewhere if needed)
        if [ -f "$station_folder/$station_txt_pattern" ]; then
            rm "$station_folder/$station_txt_pattern"
        fi

        # Remove folder if empty after moving
        rmdir "$station_folder" 2>/dev/null
    else
        echo "⚠️ Folder not found: $station_folder"
    fi
done

echo "✅ All matching station files moved to $Data_dir"

