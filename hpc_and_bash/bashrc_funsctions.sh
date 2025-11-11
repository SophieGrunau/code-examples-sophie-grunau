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
