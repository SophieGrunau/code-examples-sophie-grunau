https://github.com/SophieGrunau/code-examples-sophie-grunau
**Author:** Lea Sophie Grunau

# Scientific Computing Portfolio

A collection of code examples demonstrating proficiency in scientific data analysis, workflow automation, and high-performance computing for geoscience applications.

## Repository Overview

This repository contains examples of different coding tasks and methodologies:
```
.
├── data_analysis/              # Ephemeral lake filling event analysis
├── pre-processing/             # Data preparation pipelines
├── reproducing_paper_figures/  # Reproduction of published methods (R)
├── automatisation/             # Workflow automation scripts
└── hpc_scripts/                # HPC batch processing utilities
```

## Projects

### 🌊 Lake Filling Event Analysis (`data_analysis/`)

**Purpose:** Analyze lake-filling events in Australian ephemeral lakes using satellite observations and climate data.

**Demonstrates:**
- Complex geospatial data analysis with xarray and cartopy
- Event detection algorithms and time series analysis
- Integration of multiple data sources (satellite, gridded, station data)
- Professional code structure with comprehensive documentation
- Production of publication-quality figures

**Key Script:** `LakesCombined_2025.py`

---

### 🔧 Data Pre-Processing (`pre-processing/`)

**Purpose:** Automated pipelines for preparing heterogeneous datasets for analysis.

#### Lake Mask Conversion (`convert_lake_masks/`)
**Demonstrates:**
- Geospatial data manipulation and regridding
- Command-line tool development with argparse
- NetCDF file handling

**Scripts:**
- `convert_r001_mask_to_r005.py` - Resolution conversion
- `correct_mask_dim.py` - Dimension standardisation

#### Rainfall Station Processing (`combine_rainfall_stations/`)
**Demonstrates:**
- Multi-step data processing pipelines
- Web scraping and automated data retrieval
- Text-to-NetCDF conversion
- Spatial filtering and quality control

**Pipeline:**
1. `01_filter_rainfall_stations_by_mask.py` - Spatial filtering & opening relevant webpages
2. `02_move_rainfall_station_files.sh` - File organisation
3. `03_identify_missing_rainfall_stations.py` - QC checks
4. `04_combine_rainfall_stations_to_netcdf.py` - Data aggregation

---

### 📊 Reproducing Published Research (`reproducing_paper_figures/`)

**Purpose:** Reproduction of PCA/SVM analysis from published geological research.

**Demonstrates:**
- R programming proficiency
- Working with unfamiliar scientific domains
- Understanding and implementing published methodologies
- Statistical analysis (PCA, Support Vector Machines)

**Key Script:** `S3_PCA_SVM_method_code.R`

---

### ⚙️ Workflow Automation (`automatisation/`)

**Purpose:** Custom automation for LaTeX document management and version control.

**Demonstrates:**
- Shell scripting for workflow optimisation
- Git automation
- Practical problem-solving for daily tasks

**Script:** `ClosingTexifier.sh`

---

### 🖥️ HPC Utilities (`hpc_scripts/`)

**Purpose:** Batch processing of large climate datasets on HPC systems.

**Demonstrates:**
- PBS/Torque job submission
- Parallel processing optimisation
- Large-scale data manipulation (120 years of gridded climate data)
- Shell scripting for computational workflows

**Scripts:**
- `bashrc_functions.sh` - Environment configuration
- `crop_precip_files_agcd.pbs` - PBS job script
- `crop_precip_files_agcd.sh` - Processing script

---

## Technical Skills Demonstrated

**Languages:**
- Python (primary) - data analysis, visualisation, automation
- R - statistical analysis
- Bash/Shell - workflow automation, HPC

**Key Libraries/Tools:**
- **Data manipulation:** xarray, pandas, numpy
- **Geospatial:** cartopy, GDAL
- **Analysis:** scipy, scikit-learn (via R)
- **Visualization:** matplotlib
- **HPC:** PBS/Torque, batch processing
- **Version control:** Git

**Competencies:**
- Complex data pipeline development
- Geospatial data analysis
- Time series analysis and event detection
- Scientific visualization
- Code documentation and reproducibility
- High-performance computing
- Cross-language workflow integration

## Notes

Each script includes detailed documentation in its header. File paths in configuration sections need to be adapted for different systems.

Data sources include:
- Digital Earth Australia (satellite observations)
- Australian Gridded Climate Data (AGCD)
- Bureau of Meteorology station data
- NOAA climate driver indices

## Author

Lea Sophie Grunau