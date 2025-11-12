# crop_precip_files_agcd_pbs.sh
# Author: Sophie Grunau
# Dependencies: 
#   - crop_precip_files_agcd.sh
#   - bash
#   - nco, netcdf
# Note: This script is designed to run on NCI where the AGCD data are stored. Paths for log_files, err_files, and bash_script must be adapted to your account.
# This script cruns the crop_precip_files_agcd.sh on NCI.

# Changes, that need to be made depending on your account/ size of your data:
#   - PBS -P: your project code
#   - PBS -l walltime=: walltime required
#   - PBS -l storage=: data storage you access

# Usage example:
# qsub -o /home/<user>/log_files/crop_precip_{grid}_{freq}_{region}.out \
#      -e /home/<user>/log_files/crop_precip_{grid}_{freq}_{region}.err \
#      /home/<user>/bash_scripts/crop_precip_files_agcd.pbs


#!/bin/bash
#PBS -P iz13
#PBS -q normal
#PBS -l walltime=00:40:00
#PBS -l ncpus=1
#PBS -l mem=8GB
#PBS -l storage="gdata/iz13+gdata/zv2"
#PBS -N crop_precip
#PBS -l jobfs=1GB

# ========== input ==========
# grid options: r001/ r005
# frequency options: monthly/ daily
# region options: LE, LB, LW, LG, NT
precip_grid=r001
precip_freq=monthly
precip_region=LE

# ====== Run the main script ======
/home/574/sg7350/bash_scripts/crop_precip_files_agcd.sh grid=$precip_grid freq=$precip_freq region=$precip_region

#Run file:
# qsub -o /home/574/sg7350/log_files/crop_precip_r001_monthly_LE.out -e /home/574/sg7350/log_files/crop_precip_r001_monthly_LE.err bash_scripts/crop_precip_files_agcd.pbs