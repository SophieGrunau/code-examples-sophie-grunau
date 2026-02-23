#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download script for lake-filling event analysis data from Google Drive.

Author:      Lea Sophie Grunau  
Created on:  2026-02-23
Last updated: 2026-02-23

Description:
    Downloads the required data files from Google Drive for use with LakesCombined_2025.py.

Dependencies:
    - Python 3.x
    - gdown (pip install gdown)

Usage:
    python download_data.py
"""

import gdown
from pathlib import Path

if __name__ == "__main__":
    gdrive_folder_url = "https://drive.google.com/drive/folders/1e4Wr0movBXm1gJSMku0IQXN-YwJ-5MSI?usp=sharing"  #google drive link

    try:
        base_dir = Path(__file__).parent
    except NameError:
        base_dir = Path.cwd()

    dest_folder = base_dir / "data"

    print("Downloading data folder from Google Drive...")
    gdown.download_folder(gdrive_folder_url, output=str(dest_folder), quiet=False)
    print("Done!")
    
    