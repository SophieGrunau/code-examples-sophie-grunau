# bashrc_functions.sh
# Author: Sophie Grunau
# Note: Designed for the NCI Gadi HPC system. Paths and PBS directives are 
# specific to that environment and would need to be adapted to run elsewhere.
# Dependencies: NCI environment with qsub
#   - crop_precip_files_agcd.pbs, crop_temp_files_agcd.pbs
#   - crop_precip_files_agcd.sh, crop_temp_files_agcd.sh
# Usage: source this file in your shell, then run e.g.
#   crop_precip_submit grid=r005 freq=daily region=LE
#   crop_temp_submit var=temp freq=daily region=LE

crop_temp_submit() {
	for arg in "$@"; do
		case $arg in
			var=*) var="${arg#*=}" ;;
			freq=*) freq="${arg#*=}" ;;
			region=*) region="${arg#*=}" ;;
			*) echo "Unknown argument: $arg"; return 1 ;;
		esac
	done

	if [[ -z "$var" || -z "$freq" || -z "$region" ]]; then
		echo "Usage: crop_temp_submit var=<var> freq=<freq> region=<region>"
		return 1
	fi

	local base="/home/574/sg7350/log_files"
	local script="bash_scripts/crop_temp_files_agcd.pbs"

	qsub -o "${base}/crop_${var}_r005_${freq}_${region}.out" \
		-e "${base}/crop_${var}_r005_${freq}_${region}.err" \
		$script
}

crop_precip_submit() {
	for arg in "$@"; do
		case $arg in
			grid=*) grid="${arg#*=}" ;;
			freq=*) freq="${arg#*=}" ;;
			region=*) region="${arg#*=}" ;;
			*) echo "Unknown argument: $arg"; return 1 ;;
		esac
	done

	if [[ -z "$grid" || -z "$freq" || -z "$region" ]]; then
		echo "Usage: crop_precip_submit grid=<grid> freq=<freq> region=<region>"
		return 1
	fi

	local base="/home/574/sg7350/log_files"
	local script="bash_scripts/crop_precip_files_agcd.pbs"

	qsub -o "${base}/crop_precip_${grid}_${freq}_${region}.out" \
	     -e "${base}/crop_precip_${grid}_${freq}_${region}.err" \
	     $script
}
