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

detect_cal_suffix() {
    # Find the most-processed flavor of cal files in stage2_output and set
    # file_pattern/suffix2 accordingly. Every step that consumes cal files
    # (background subtraction, reference download, stage 3) calls this
    # itself, so skipping one step never leaves suffix2 unset for the next.
    if compgen -G "$OBS_DIR/stage2_output/jw*cal_cfnoise.fits" > /dev/null; then
        echo "[Detected *_cal_cfnoise.fits files]"
        file_pattern="$OBS_DIR/stage2_output/jw*cal_cfnoise.fits"
        suffix2="_cfnoise"
    elif compgen -G "$OBS_DIR/stage2_output/jw*cal_wisp.fits" > /dev/null; then
        echo "[Detected *_cal_wisp.fits files]"
        file_pattern="$OBS_DIR/stage2_output/jw*cal_wisp.fits"
        suffix2="_wisp"
    elif compgen -G "$OBS_DIR/stage2_output/jw*cal.fits" > /dev/null; then
        echo "[Detected *_cal.fits files]"
        file_pattern="$OBS_DIR/stage2_output/jw*cal.fits"
        suffix2=""
    else
        return 1
    fi
}

delete_directory_if_exists() {
    local dir=$1
    if [ -d "$dir" ]; then
        echo "[Deleting $dir to avoid conflicts.]"
        echo ""
        rm -rf "$dir"
    fi
}

is_comma_list() {
    [[ "$1" == *","* ]]
}

crds_bestrefs_for_uncal_input() {
    local uncal_input="$1"
    local uncal_files=()

    # Globs need to be expanded by the shell (not passed as literal strings)
    # so CRDS sees a list of real files. nullglob avoids handing CRDS the
    # raw pattern when nothing matches.
    shopt -s nullglob

    if is_comma_list "$uncal_input"; then
        # Comma-separated list of files; gather unique parent directories
        # and glob each one for jw*uncal.fits.
        IFS=',' read -r -a files <<< "$uncal_input"
        declare -A uniq_dirs
        for f in "${files[@]}"; do
            [ -n "$f" ] || continue
            uniq_dirs[$(dirname "$f")]=1
        done
        for d in "${!uniq_dirs[@]}"; do
            uncal_files+=( "$d"/jw*uncal.fits )
        done
    elif [ -d "$uncal_input" ]; then
        uncal_files=( "$uncal_input"/jw*uncal.fits )
    elif [ -f "$uncal_input" ]; then
        uncal_files=( "$uncal_input" )
    fi

    shopt -u nullglob

    if [ ${#uncal_files[@]} -eq 0 ]; then
        echo "[Error] No uncal files found at: $uncal_input"
        return 1
    fi

    crds bestrefs --files "${uncal_files[@]}" --sync-references=1
}

delete_stage3_directory_if_exists() {
    local dir="$1"

    if [ -d "$dir" ]; then
        echo "[Deleting $dir to avoid conflicts.]"
        echo ""
        rm -rf "$dir"
    fi
}


combine_observations=$(get_yaml_value 'combine_observations' "$CONFIG_FILE")
group_by_directory=$(get_yaml_value 'group_by_directory' "$CONFIG_FILE")
custom_name=$(get_yaml_value 'custom_name' "$CONFIG_FILE")

PIPELINE_DIR=$(get_yaml_value 'pipeline_directory' "$CONFIG_FILE")
MY_CRDS_PATH=$(get_yaml_value 'crds_path' "$CONFIG_FILE")
MY_CRDS_SERVER_URL=$(get_yaml_value 'crds_server_url' "$CONFIG_FILE")
WISP_DIR=$(get_yaml_value 'wisp_directory' "$CONFIG_FILE")
STAGE1_NPROC=$(get_yaml_value 'stage1_nproc' "$CONFIG_FILE")
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
    LOG_FILEW="$OBS_DIR/logs/pipeline_wisp.log"
    LOG_FILEB="$OBS_DIR/logs/pipeline_bkg.log"
    LOG_FILECF="$OBS_DIR/logs/pipeline_cfnoise.log"

    mkdir -p "$OBS_DIR"
    mkdir -p "$OBS_DIR/logs"

    if ! should_skip_step "download_uncal_references"; then
        echo "« Downloading references for uncal.fits files »"
        echo "  ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯  "
        crds_bestrefs_for_uncal_input "$UNCAL_PATH" || exit 1
        echo ""
    else
        echo "[Download uncal references skipped]"
        echo ""
    fi


    if ! should_skip_step "stage1"; then
        delete_directory_if_exists "$OBS_DIR/stage1_output"
        echo "==================="
        echo " Pipeline: stage 1 "
        echo "==================="

        if is_comma_list "$UNCAL_PATH"; then
            python "$PIPELINE_DIR/utils/pipeline_stage1.py" \
                --nproc "$STAGE1_NPROC" \
                --combined_mode \
                --input_dir "$UNCAL_PATH" \
                --output_dir "$OBS_DIR/stage1_output"
        else
            python "$PIPELINE_DIR/utils/pipeline_stage1.py" \
                --nproc "$STAGE1_NPROC" \
                --input_dir "$UNCAL_PATH" \
                --output_dir "$OBS_DIR/stage1_output"
        fi
        echo ""
    else
        echo "[Pipeline Stage 1 skipped]"
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
        echo "« Subtracting wisps from exposures »"
        echo "  ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯  "
        python "$PIPELINE_DIR/utils/subtract_wisp.py" --files $OBS_DIR/stage2_output/jw*cal.fits --wisp_dir "$WISP_DIR" --output_dir "$OBS_DIR/stage2_output" --suffix "_wisp" --nproc "$WISP_NPROC"
        echo ""
    else
        echo "[Wisp subtraction skipped]"
        echo ""
    fi

    if ! should_skip_step "cal_fnoise_reduction"; then
        echo "« Reducing 1/f noise in exposures »"
        echo "  ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯  "

        # Collect cal files of both flavors in one shot so multiprocessing
        # has the full work list. fnoise_reduction.py derives the output
        # filename from each input's actual name.
        shopt -s nullglob
        CFNOISE_INPUTS=( "$OBS_DIR"/stage2_output/jw*cal.fits "$OBS_DIR"/stage2_output/jw*cal_wisp.fits )
        shopt -u nullglob

        if [ ${#CFNOISE_INPUTS[@]} -gt 0 ]; then
            echo "[Processing ${#CFNOISE_INPUTS[@]} files]"
            python "$PIPELINE_DIR/utils/fnoise_reduction.py" \
                --files "${CFNOISE_INPUTS[@]}" \
                --output_dir "$OBS_DIR/stage2_output" \
                --nproc "$CF_NPROC"
        else
            echo "[No cal files found in $OBS_DIR/stage2_output]"
        fi
        echo ""

    else
        echo "[Cal fnoise reduction skipped]"
        echo ""
    fi

    if ! should_skip_step "background_subtraction"; then
        echo "« Subtracting background from exposures »"
        echo "  ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯  "

        if ! detect_cal_suffix; then
            echo "[Error: No suitable input files found for background subtraction!]"
            exit 1
        fi

        python "$PIPELINE_DIR/utils/bkg_sub_parallel.py" \
            --input_dir "$OBS_DIR/stage2_output" \
            --nproc "$BKG_NPROC" \
            --output_dir "$OBS_DIR/stage2_output" \
            --files $file_pattern \
            --suffix "$suffix2"
        
        echo ""

    else
        echo "[Background subtraction skipped]"
        echo ""
    fi

    if ! should_skip_step "download_cal_references"; then
        echo "« Downloading references for cal.fits files »"
        echo "  ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯  "
        if ! detect_cal_suffix; then
            echo "[Error: No cal files found for reference download!]"
            exit 1
        fi
        crds bestrefs --files $file_pattern --sync-references=1
        echo ""
    else
        echo "[Download cal references skipped]"
        echo ""
    fi

    if ! should_skip_step "stage3"; then
        delete_stage3_directory_if_exists "$OBS_DIR/stage3_output"
        echo "===================="
        echo " Pipeline - stage 3"
        echo "===================="
        if ! detect_cal_suffix; then
            echo "[Error: No cal files found for stage 3!]"
            exit 1
        fi
        python "$PIPELINE_DIR/utils/pipeline_stage3.py" --input_dir "$OBS_DIR/stage2_output" --target "$OBS_NAME" --output_dir "$OBS_DIR/stage3_output" --input_suffix "$suffix2"
        echo ""
    else
        echo "[Pipeline Stage 3 skipped]"
        echo ""
    fi

    COLOR_IMAGE_ENABLED=$(get_yaml_value 'color_image_enabled' "$CONFIG_FILE")
    if [[ "$COLOR_IMAGE_ENABLED" == "true" ]]; then
        echo "« Creating color image »"
        echo "  ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯  "
        COLOR_MIN_LEVEL=$(get_yaml_value 'color_image_min_level' "$CONFIG_FILE")
        COLOR_MAX_QUANTILE=$(get_yaml_value 'color_image_max_quantile' "$CONFIG_FILE")
        COLOR_GAMMA=$(get_yaml_value 'color_image_gamma' "$CONFIG_FILE")
        COLOR_HUES=$(get_yaml_value 'color_image_filter_hues' "$CONFIG_FILE")
        COLOR_SUBTRACT_SKY=$(get_yaml_value 'color_image_subtract_sky' "$CONFIG_FILE")
        if [[ "$COLOR_SUBTRACT_SKY" == "false" ]]; then
            SKY_FLAG="--no-subtract-sky"
        else
            SKY_FLAG="--subtract-sky"
        fi
        python "$PIPELINE_DIR/utils/color_image.py" \
            --obs-dir "$OBS_DIR" \
            --target "$OBS_NAME" \
            --min-level "${COLOR_MIN_LEVEL:-0.001}" \
            --max-quantile "${COLOR_MAX_QUANTILE:-0.99999}" \
            --gamma "${COLOR_GAMMA:-2.2}" \
            --filter-hues "${COLOR_HUES:-\{\}}" \
            $SKY_FLAG
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
