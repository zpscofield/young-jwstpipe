#!/bin/bash
START_TIME_TOTAL=$(date +%s)

CONFIG_FILE="config.yaml"

DATA_DIR_FROM_YAML=$(yq '.data_directory // ""' "$CONFIG_FILE" | tr -d '"')
DATA_DIR=${DATA_DIR_FROM_YAML:-$(dirname "$(realpath "$0")")}

OUTPUT_DIR_FROM_YAML=$(yq '.output_directory // ""' "$CONFIG_FILE" | tr -d '"')
OUTPUT_DIR=${OUTPUT_DIR_FROM_YAML:-$(dirname "$(realpath "$0")")}

get_yaml_value() {
    local key=$1
    local file=$2
    yq .$key $file | tr -d '"'
}

should_skip_step() {
    local step=$1
    yq '.skip_steps // [] | .[]' "$CONFIG_FILE" 2>/dev/null | grep -q "$step"
}

delete_directory_if_exists() {
    local dir=$1
    if [ -d "$dir" ]; then
        echo "[Deleting $dir to avoid conflicts.]"
        echo ""
        rm -rf "$dir"
    fi
}

delete_stage3_directory_if_exists() {
    local dir=$1
    if [ -d "$dir" ]; then
        restore_stage2_files "$OBS_DIR/stage3_output" "$OBS_DIR/stage2_output"
        echo "[Deleting $dir to avoid conflicts.]"
        echo ""
        rm -rf "$dir"
    fi
}

restore_stage2_files() {
    local stage3_dir="$1"
    local stage2_dir="$2"

    echo "Restoring and renaming files from $stage3_dir to $stage2_dir..."
    # Find all `cal.fits` files in the filter subdirectories
    find "$stage3_dir" -type f -name '*cal.fits' | while read -r file; do
        # Extract the base filename without the directory
        base_name=$(basename "$file")
        # Add "_final" before the ".fits" extension
        new_name="${base_name%.fits}_final.fits"
        # Move and rename the file to stage2_dir
        mv "$file" "$stage2_dir/$new_name"
    done

    echo "Files restored and renamed successfully."
}


combine_observations=$(get_yaml_value 'combine_observations' "$CONFIG_FILE")
group_by_directory=$(get_yaml_value 'group_by_directory' "$CONFIG_FILE")
custom_name=$(get_yaml_value 'custom_name' "$CONFIG_FILE")
full_exposure_striping=$(get_yaml_value 'full_exposure_striping' "$CONFIG_FILE")

PIPELINE_DIR=$(get_yaml_value 'pipeline_directory' "$CONFIG_FILE")
MY_CRDS_PATH=$(get_yaml_value 'crds_path' "$CONFIG_FILE")
MY_CRDS_SERVER_URL=$(get_yaml_value 'crds_server_url' "$CONFIG_FILE")
WISP_DIR=$(get_yaml_value 'wisp_directory' "$CONFIG_FILE")
STAGE1_NPROC=$(get_yaml_value 'stage1_nproc' "$CONFIG_FILE")
FNOISE_NPROC=$(get_yaml_value 'fnoise_nproc' "$CONFIG_FILE")
STAGE2_NPROC=$(get_yaml_value 'stage2_nproc' "$CONFIG_FILE")
WISP_NPROC=$(get_yaml_value 'wisp_nproc' "$CONFIG_FILE")
CF_NPROC=$(get_yaml_value 'cfnoise_nproc' "$CONFIG_FILE")
BKG_NPROC=$(get_yaml_value 'bkg_nproc' "$CONFIG_FILE")

export CRDS_PATH=$MY_CRDS_PATH
export CRDS_SERVER_URL=$MY_CRDS_SERVER_URL

run_pipeline() {
    START_TIME=$(date +%s)
    local OBS_NAME=$1
    local UNCAL_PATH=$2

    echo ""
    echo "Processing [$OBS_NAME]"
    echo ""

    OBS_DIR="$OUTPUT_DIR/$OBS_NAME"
    mkdir -p "$OBS_DIR/logs"

    LOG_FILE1="$OBS_DIR/logs/pipeline_stage1.log"
    LOG_FILE2="$OBS_DIR/logs/pipeline_stage2.log"
    LOG_FILE3="$OBS_DIR/logs/pipeline_stage3.log"
    LOG_FILEF="$OBS_DIR/logs/pipeline_fnoise.log"
    LOG_FILEW="$OBS_DIR/logs/pipeline_wisp.log"
    LOG_FILEB="$OBS_DIR/logs/pipeline_bkg.log"
    LOG_FILECF="$OBS_DIR/logs/pipeline_cfnoise.log"

    mkdir -p "$OBS_DIR"
    mkdir -p "$OBS_DIR/logs"

    if ! should_skip_step "download_uncal_references"; then
        echo "« Downloading references for uncal.fits files »"
        echo "  ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯  "
        if [[ "$combine_observations" == "true" || "$group_by_directory" == "true" ]]; then
            IFS=',' read -r -a uncal_files <<< "$UNCAL_PATH" # Comma-separated string to array of file paths
            declare -A unique_dirs  # Associative array for uniqueness
            for file in "${uncal_files[@]}"; do
                dir=$(dirname "$file")
                unique_dirs["$dir"]=1
            done
            # Process each unique directory
            for dir in "${!unique_dirs[@]}"; do
                crds bestrefs --files ${dir}/jw*uncal.fits --sync-references=1
            done
        elif [[ "$combine_observations" == "false" ]]; then
            crds bestrefs --files ${UNCAL_PATH} --sync-references=1
        fi
        echo ""
    else
        echo "[Download uncal references skipped]"
        echo ""
    fi

    if ! should_skip_step "stage1"; then
        if [ -f "$LOG_FILE1" ]; then
            echo "[Existing pipeline_stage1.log file deleted]"
            echo ""
            rm "$LOG_FILE1"
        fi
        delete_directory_if_exists "$OBS_DIR/stage1_output"
        echo "==================="
        echo " Pipeline: stage 1 "
        echo "==================="
        if [[ "$combine_observations" == "true" ]]; then
            echo "Running pipeline in combined mode."
        fi
        if [[ "$combine_observations" == "true" || "$group_by_directory" == "true" ]]; then
            python "$PIPELINE_DIR/utils/pipeline_stage1.py" --nproc "$STAGE1_NPROC" --combined_mode --input_dir "$UNCAL_PATH" --output_dir "$OBS_DIR/stage1_output"
        elif [[ "$combine_observations" == "false" ]]; then
            python "$PIPELINE_DIR/utils/pipeline_stage1.py" --nproc "$STAGE1_NPROC" --input_dir "$UNCAL_PATH" --output_dir "$OBS_DIR/stage1_output"
        fi
        echo ""
    else
        echo "[Pipeline Stage 1 skipped]"
        echo ""
    fi

    if ! should_skip_step "fnoise_correction"; then
        if [ -f "$LOG_FILEF" ]; then
            echo "[Existing pipeline_fnoise.log file deleted]"
            echo ""
            rm "$LOG_FILEF"
        fi
        echo "« Correcting 1/f noise »"
        echo "  ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯  "
        echo "Accessing flat files before beginning calibration..."
        if [[ "$full_exposure_striping" == "true" ]]; then
            python "$PIPELINE_DIR/utils/remstriping_update_parallel.py" --runall --nproc "$FNOISE_NPROC" --output_dir "$OBS_DIR/stage1_output" 
        else
            python "$PIPELINE_DIR/utils/remstriping_update_parallel.py" --runall --nproc "$FNOISE_NPROC" --output_dir "$OBS_DIR/stage1_output"
        fi
        echo ""
    else
        echo "[1/f noise correction skipped]"
        echo ""
    fi

    if ! should_skip_step "download_rate_references"; then
        echo "« Downloading references for rate.fits files »"
        echo "  ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯  "
        crds bestrefs --files $OBS_DIR/stage1_output/jw*rate.fits --sync-references=1
        echo ""
    else
        echo "[Download rate references skipped]"
        echo ""
    fi

    if ! should_skip_step "stage2"; then
        if [ -f "$LOG_FILE2" ]; then
            echo "[Existing pipeline_stage2.log file deleted]"
            echo ""
            rm "$LOG_FILE2"
        fi
        delete_directory_if_exists "$OBS_DIR/stage2_output"
        echo "===================="
        echo " Pipeline - stage 2 "
        echo "===================="
        python "$PIPELINE_DIR/utils/pipeline_stage2.py" --input_dir "$OBS_DIR/stage1_output" --nproc "$STAGE2_NPROC" --output_dir "$OBS_DIR/stage2_output"
        echo ""
    else
        echo "[Pipeline Stage 2 skipped]"
        echo ""
    fi

    if ! should_skip_step "wisp_subtraction"; then
        if [ -f "$LOG_FILEW" ]; then
            echo "[Existing pipeline_wisp.log file deleted]"
            echo ""
            rm "$LOG_FILEW"
        fi
        echo "« Subtracting wisps from exposures »"
        echo "  ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯  "
        python "$PIPELINE_DIR/utils/subtract_wisp.py" --files $OBS_DIR/stage2_output/jw*cal.fits --wisp_dir "$WISP_DIR" --output_dir "$OBS_DIR/stage2_output" --suffix "_wisp" --nproc "$WISP_NPROC"
        echo ""
    else
        echo "[Wisp subtraction skipped]"
        echo ""
    fi

    if ! should_skip_step "cal_fnoise_reduction"; then
        if [ -f "$LOG_FILECF" ]; then
            echo "[Existing pipeline_cfnoise.log file deleted]"
            echo ""
            rm "$LOG_FILECF"
        fi
        echo "« Reducing 1/f noise in exposures »"
        echo "  ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯  "
        if should_skip_step "wisp_subtraction"; then
            python "$PIPELINE_DIR/utils/fnoise_reduction.py" --files $OBS_DIR/stage2_output/jw*cal.fits --output_dir "$OBS_DIR/stage2_output" --suffix "_cfnoise" --nproc "$CF_NPROC"
        else
            python "$PIPELINE_DIR/utils/fnoise_reduction.py" --files $OBS_DIR/stage2_output/jw*cal_wisp.fits --output_dir "$OBS_DIR/stage2_output" --suffix "_cfnoise" --nproc "$CF_NPROC"
        fi
        echo ""
    
    else
        echo "[Cal fnoise reductuction skipped]"
        echo ""
    fi

    if ! should_skip_step "background_subtraction"; then
        if [ -f "$LOG_FILEB" ]; then
            echo "[Existing pipeline_bkg.log file deleted]"
            echo ""
            rm "$LOG_FILEB"
        fi
        echo "« Subtracting background from exposures »"
        echo "  ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯  "
        if should_skip_step "wisp_subtraction"; then
            if should_skip_step "cal_fnoise_reduction"; then
                python "$PIPELINE_DIR/utils/bkg_sub_parallel.py" --input_dir "$OBS_DIR/stage2_output" --nproc "$BKG_NPROC" --output_dir "$OBS_DIR/stage2_output" --files $OBS_DIR/stage2_output/jw*cal.fits --suffix "_cal"
            else
                python "$PIPELINE_DIR/utils/bkg_sub_parallel.py" --input_dir "$OBS_DIR/stage2_output" --nproc "$BKG_NPROC" --output_dir "$OBS_DIR/stage2_output" --files $OBS_DIR/stage2_output/jw*cal_cfnoise.fits --suffix "_cfnoise"
            fi
        elif should_skip_step "cal_fnoise_reduction"; then
            python "$PIPELINE_DIR/utils/bkg_sub_parallel.py" --input_dir "$OBS_DIR/stage2_output" --nproc "$BKG_NPROC" --output_dir "$OBS_DIR/stage2_output" --files $OBS_DIR/stage2_output/jw*cal_wisp.fits --suffix "_wisp"
        else
            python "$PIPELINE_DIR/utils/bkg_sub_parallel.py" --input_dir "$OBS_DIR/stage2_output" --nproc "$BKG_NPROC" --output_dir "$OBS_DIR/stage2_output" --files $OBS_DIR/stage2_output/jw*cal_cfnoise.fits --suffix "_cfnoise"
        echo ""
        fi
    else
        echo "[Background subtraction skipped]"
        echo ""
    fi

    if ! should_skip_step "download_cal_references"; then
        echo "« Downloading references for cal.fits files »"
        echo "  ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯  "
        crds bestrefs --files $OBS_DIR/stage2_output/jw*cal_cfnoise.fits --sync-references=1
        echo ""
    else
        echo "[Download cal references skipped]"
        echo ""
    fi

    if ! should_skip_step "stage3"; then
        if [ -f "$LOG_FILE3" ]; then
            echo "[Existing pipeline_stage3.log file deleted]"
            echo ""
            rm "$LOG_FILE3"
        fi
        delete_stage3_directory_if_exists "$OBS_DIR/stage3_output"
        echo "===================="
        echo " Pipeline - stage 3"
        echo "===================="
        python "$PIPELINE_DIR/utils/pipeline_stage3.py" --input_dir "$OBS_DIR/stage2_output" --target "$OBS_NAME" --output_dir "$OBS_DIR/stage3_output"
        echo ""
    else
        echo "[Pipeline Stage 3 skipped]"
        echo ""
    fi

    echo "===================="
    echo " Pipeline completed "
    echo "===================="
    echo ""

    END_TIME=$(date +%s)
    ELAPSED_TIME=$(( END_TIME - START_TIME ))

    echo "Time elapsed for [$OBS_NAME]: $((ELAPSED_TIME / 3600))h $(((ELAPSED_TIME % 3600) / 60))m $((ELAPSED_TIME % 60))s"
    echo ""
}

echo ""
echo "################################"
echo "#                              #"
echo "# JWST data reduction pipeline #"
echo "#                              #"
echo "################################"

OBSERVATIONS=$(python "$PIPELINE_DIR/utils/get_obs_info.py" "$PIPELINE_DIR")

echo ""
echo "« Observations found »"
echo "  ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯  "

IFS=$'\n'
target_names=""
for line in $OBSERVATIONS; do
    if [[ "$line" == TARGET_NAMES:* ]]; then
        target_names=$(echo "$line" | cut -d':' -f2)
    elif [[ "$line" == OBS:* ]]; then
        target_name=$(echo "$line" | cut -d':' -f2)
        target_dir=$(echo "$line" | cut -d':' -f3)
    fi
done

# Print all target names
echo "All target names:"
IFS=',' read -r -a targets <<< "$target_names"
for target in "${targets[@]}"; do
    echo "- $target"
done

IFS=$'\n'
for line in $OBSERVATIONS; do
    if [[ "$line" == OBS:* ]]; then
        target_name=$(echo "$line" | cut -d':' -f2)
        target_dir=$(echo "$line" | cut -d':' -f3)
        run_pipeline "$target_name" "$target_dir"
    fi
done

END_TIME_TOTAL=$(date +%s)
ELAPSED_TIME=$(( END_TIME_TOTAL - START_TIME_TOTAL ))

echo "Total time elapsed for all observations: $((ELAPSED_TIME / 3600))h $(((ELAPSED_TIME % 3600) / 60))m $((ELAPSED_TIME % 60))s"
echo ""
