#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:      Lea Sophie Grunau  
Created on:  2025-07-15
Last updated: 2026-02-23

Identify BOM rainfall stations that were filtered for a given lake but for which
no corresponding rainfall CSV file has been downloaded yet.

This script:
    • Loads the filtered list of rainfall stations for a given lake.
    • Checks which of these have corresponding downloaded data files in the
      expected format and directory.
    • Identifies stations that are missing.
    • Saves a simple list of missing station IDs as a CSV file (no header).
      This can be used for retrying the download step later.

Usage:
    python 03_identify_missing_rainfall_stations.py --Lake <lake_short_code>
    e.g. python 03_identify_missing_rainfall_stations.py --Lake LW

Output:
    {Lake}_missed_rainfall_stations.csv
"""

import pandas as pd
from pathlib import Path
import argparse


# ========== Parse command-line argument ==========
parser = argparse.ArgumentParser(description="Returns csv of relevant rainfall stations and opens their webpages")
parser.add_argument("--Lake", required=True, help="Lake name short form (e.g. LE, LW, LG, LB)")
args = parser.parse_args()

Lake = args.Lake

# Data directory
data_dir = Path(__file__).parent.parent / 'data' / f'{Lake}_rainfall_stations'

# ========== Main workflow ==========
# 1. Load full list
stations_df = pd.read_csv(data_dir / f"{Lake}_rainfall_stations.csv", dtype={"Site": str})

# 2. Get list of downloaded files
downloaded = set()
for file in data_dir.glob("IDCJAC0009_*_1800_Data.csv"):
    station_id = file.name.split("_")[1]
    downloaded.add(station_id)

# 3. Identify missed stations
missed_df = stations_df[~stations_df["Site"].isin(downloaded)]

# 4. Save or print
missed_df["Site"].to_csv(Path(f"{data_dir}/{Lake}_missed_rainfall_stations.csv"), index=False, header=False)
print(f"🛠️  Recovered {len(missed_df)} missed stations.")
