```markdown
# HPC Utilities: AGCD Climate Data Processing

**Author:** Lea Sophie Grunau

Batch processing scripts for extracting regional subsets from the Australian Gridded 
Climate Dataset (AGCD) on the NCI Gadi HPC system. Crops 120+ years of national 
gridded rainfall data to specified lake catchment regions and combines yearly files 
into a single NetCDF.

## Note

These scripts are designed for the NCI Gadi HPC system where the AGCD data are stored. 
They are included here as code examples demonstrating HPC workflow automation — they 
cannot be run without an NCI account and access to the AGCD data library.

## Contents

- `crop_precip_files_agcd.sh` — main processing script: crops yearly rainfall files 
  to a specified region and merges into a single NetCDF
- `crop_precip_files_agcd.pbs` — PBS job script to submit the above to the HPC queue
- `bashrc_functions.sh` — shell functions for convenient job submission

## What the Scripts Do

`crop_precip_files_agcd.sh` takes a grid resolution, frequency, and region as arguments,
copies the corresponding yearly AGCD precipitation files, crops each to the specified 
regional bounding box, and concatenates all years (1900–2024) into a single output file.

`bashrc_functions.sh` defines convenience functions so instead of typing the full `qsub` 
command with all flags, you can simply run:
```bash
crop_precip_submit grid=r005 freq=daily region=LW
crop_temp_submit var=tmax freq=daily region=LW
```

## Supported Options

- **Grid:** `r001` (0.01°), `r005` (0.05°)
- **Frequency:** `daily`, `monthly`
- **Regions:** `LW` (Lake Woods), `LE` (Lake Eyre), `LB` (Lake Buchanan), `LG` (Lake George), `NT` (Northern Territory)

## Dependencies

- NCI Gadi HPC environment
- PBS/Torque job scheduler (`qsub`)
- NCO (NetCDF Operators): `ncks`, `ncrcat`
- Access to AGCD data (`/g/data/zv2/agcd/`)
```