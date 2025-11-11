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
# qsub -o /home/574/sg7350/log_files/crop_precip_r005_monthly_LG.out -e /home/574/sg7350/log_files/crop_precip_r005_monthly_LG.err bash_scripts/crop_precip_files_agcd.pbs