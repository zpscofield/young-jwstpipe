"""Streamlit front-end for the YOUNG JWST calibration pipeline.

Local usage:
    streamlit run app.py
    # then open http://localhost:8501

Running on a remote machine over SSH (the common case):
    # On the remote machine:
    streamlit run app.py
    # On your laptop, in a separate terminal:
    ssh -L 8501:localhost:8501 you@remote
    # then open http://localhost:8501 in your laptop's browser

The included .streamlit/config.toml sets headless = true so Streamlit
will not try to launch a browser on the remote machine.

This skeleton handles the config side: it reads config.yaml, shows every
setting as a form widget, and writes the chosen values back when you
click Save. The data-source section (MAST lookup, program-ID download,
existing directory) and the Save & Run button are added in later
commits.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))
from mast_lookup import (
    search_by_target,
    search_by_coordinates,
    search_by_proposal,
    summarize,
    filter_by_summary_rows,
    resolve_target,
)
from mast_download import download_uncal


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.yaml"
PIPELINE_SCRIPT = REPO_ROOT / "young_pipeline.sh"
MAX_LOG_LINES = 500

PIPELINE_STEPS = [
    "download_uncal_references",
    "stage1",
    "fnoise_correction",
    "download_rate_references",
    "stage2",
    "wisp_subtraction",
    "cal_fnoise_reduction",
    "background_subtraction",
    "download_cal_references",
    "stage3",
]


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f) or {}


def save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)


def validate_config(config: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for the given config. Errors block running."""
    errors: list[str] = []
    warnings: list[str] = []

    data_dir = Path(str(config.get("data_directory", "")).strip()).expanduser()
    if not str(data_dir):
        errors.append("Data directory is empty.")
    elif not data_dir.exists():
        errors.append(f"Data directory does not exist: {data_dir}")
    else:
        has_uncal = any(data_dir.rglob("*_uncal.fits"))
        if not has_uncal:
            warnings.append(
                f"No *_uncal.fits files found in {data_dir} (search is recursive). "
                "Download some first or point to a different directory."
            )

    skip = set(config.get("skip_steps") or [])
    if "wisp_subtraction" not in skip:
        wisp = str(config.get("wisp_directory", "")).strip()
        if not wisp:
            warnings.append(
                "WISP templates directory is empty but wisp_subtraction is not skipped. "
                "Either set a path or add 'wisp_subtraction' to the skipped steps."
            )
        elif not Path(wisp).expanduser().exists():
            warnings.append(f"WISP templates directory does not exist: {wisp}")

    crds = str(config.get("crds_path", "")).strip()
    if crds and not Path(crds).expanduser().exists():
        warnings.append(
            f"CRDS cache path does not exist yet: {crds}. The pipeline will create it on first use."
        )

    return errors, warnings


def run_pipeline_streaming() -> int:
    """Run young_pipeline.sh and stream its output into the UI. Returns exit code."""
    with st.status("Running pipeline…", expanded=True, state="running") as status:
        log_placeholder = st.empty()
        lines: list[str] = []

        process = subprocess.Popen(
            ["bash", str(PIPELINE_SCRIPT)],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line.rstrip("\n"))
            log_placeholder.code(
                "\n".join(lines[-MAX_LOG_LINES:]),
                language=None,
            )

        return_code = process.wait()

        if return_code == 0:
            status.update(label="Pipeline finished successfully.", state="complete")
        else:
            status.update(
                label=f"Pipeline exited with code {return_code}. See log above.",
                state="error",
            )
        return return_code


def _get(config: dict, key: str, default):
    value = config.get(key)
    return value if value is not None else default


st.set_page_config(page_title="YOUNG JWST Pipeline", layout="wide")
st.title("YOUNG JWST Calibration Pipeline")
st.caption(f"Editing {CONFIG_PATH}")

current = load_config()
new_config = dict(current)


# 1. Data source
st.header("1. Data source")
data_source_mode = st.radio(
    "How do you want to provide data?",
    [
        "MAST lookup by target name",
        "MAST lookup by RA / Dec",
        "MAST lookup by program ID",
        "Use existing directory",
    ],
    horizontal=True,
)


def _render_aladin(ra_deg: float, dec_deg: float, radius_arcsec: float, label: str = ""):
    """Render an Aladin Lite viewer centered on (ra, dec) with a circle for the radius."""
    # Field of view roughly 4x the search radius, with sensible bounds.
    fov_deg = max(min(4 * radius_arcsec / 3600.0, 5.0), 0.05)
    safe_label = label.replace("'", "").replace('"', "")
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js"></script>
  <style>
    html, body {{ margin: 0; padding: 0; background: #000; }}
    #aladin-lite-div {{ width: 100%; height: 420px; }}
  </style>
</head>
<body>
  <div id="aladin-lite-div"></div>
  <script>
    A.init.then(() => {{
      const aladin = A.aladin('#aladin-lite-div', {{
        target: '{ra_deg} {dec_deg}',
        fov: {fov_deg},
        survey: 'P/DSS2/color',
        showLayersControl: true,
        showGotoControl: true,
        showZoomControl: true,
        showFullscreenControl: true,
        showCooGrid: false
      }});
      const overlay = A.graphicOverlay({{color: 'cyan', lineWidth: 2}});
      aladin.addOverlay(overlay);
      overlay.add(A.circle({ra_deg}, {dec_deg}, {radius_arcsec / 3600.0}));
      const cat = A.catalog({{name: 'Search center', sourceSize: 18, color: 'magenta'}});
      aladin.addCatalog(cat);
      cat.addSources([A.source({ra_deg}, {dec_deg}, {{name: '{safe_label}'}})]);
    }});
  </script>
</body>
</html>"""
    components.html(html, height=440)


def _show_search_results(observations, download_dir: str, session_key: str):
    """Display a summary table with selectable rows and a Download button."""
    summary_rows = summarize(observations)
    total_frames = sum(row["n_frames"] for row in summary_rows)
    st.markdown(
        f"**Found {len(observations)} observations in "
        f"{len({row['program'] for row in summary_rows})} programs ({total_frames} frames).**"
    )
    st.caption("Tick rows to download just those program/filter combinations. Leave nothing ticked to download everything.")

    event = st.dataframe(
        summary_rows,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="multi-row",
        key=f"select_{session_key}",
    )
    selected_indices = list(event.selection.rows) if event and event.selection else []

    if selected_indices:
        selected_rows = [summary_rows[i] for i in selected_indices]
        to_download = filter_by_summary_rows(observations, selected_rows)
        selected_frames = sum(summary_rows[i]["n_frames"] for i in selected_indices)
        button_label = f"Download {selected_frames} selected frames"
    else:
        to_download = observations
        button_label = f"Download all {total_frames} frames"

    if st.button(button_label, key=f"download_{session_key}", type="primary"):
        progress_bar = st.progress(0.0, text="Starting…")
        status_text = st.empty()

        def on_progress(i, total, filename, status):
            progress_bar.progress(i / total, text=f"[{i}/{total}] {filename} ({status})")
            if status == "failed":
                status_text.warning(f"Failed: {filename}")

        try:
            result = download_uncal(to_download, download_dir, progress=on_progress)
        except Exception as exc:
            st.error(f"Download failed: {exc}")
            return None

        progress_bar.empty()
        n_ok = len(result["downloaded"])
        n_fail = len(result["failed"])
        if n_fail:
            st.warning(f"Downloaded {n_ok} files to {result['path']}, {n_fail} failed.")
            with st.expander("Failed files"):
                st.write(result["failed"])
        else:
            st.success(f"Downloaded {n_ok} files to {result['path']}")
        return str(result["path"])
    return None


if data_source_mode == "MAST lookup by target name":
    col_t, col_r, col_d = st.columns([2, 1, 2])
    with col_t:
        target_name = st.text_input("Target name", placeholder="e.g. Abell 2744")
    with col_r:
        target_radius = st.number_input(
            "Search radius (arcsec)",
            min_value=1.0,
            max_value=3600.0,
            value=60.0,
            step=10.0,
            key="target_radius",
        )
    with col_d:
        target_dest = st.text_input(
            "Download to",
            value=_get(current, "data_directory", "./data"),
            key="target_dest",
        )

    if st.button("Search MAST", key="search_target"):
        clean = target_name.strip()
        if not clean:
            st.error("Enter a target name first.")
        else:
            with st.spinner(f"Searching MAST for '{clean}'…"):
                try:
                    st.session_state["mast_obs_target"] = search_by_target(
                        clean, radius_arcsec=float(target_radius)
                    )
                    st.session_state["mast_target_name"] = clean
                    st.session_state["mast_target_radius"] = float(target_radius)
                    coords = resolve_target(clean)
                    if coords is not None:
                        st.session_state["mast_target_coords"] = coords
                    else:
                        st.session_state.pop("mast_target_coords", None)
                except Exception as exc:
                    st.session_state.pop("mast_obs_target", None)
                    st.error(f"MAST search failed: {exc}")

    if "mast_target_coords" in st.session_state:
        ra, dec = st.session_state["mast_target_coords"]
        _render_aladin(
            ra,
            dec,
            st.session_state.get("mast_target_radius", float(target_radius)),
            label=st.session_state.get("mast_target_name", ""),
        )

    if "mast_obs_target" in st.session_state:
        downloaded_path = _show_search_results(
            st.session_state["mast_obs_target"], target_dest, "target"
        )
        if downloaded_path:
            st.session_state["resolved_data_directory"] = downloaded_path

    new_config["data_directory"] = st.session_state.get("resolved_data_directory", target_dest)

elif data_source_mode == "MAST lookup by RA / Dec":
    col_ra, col_dec, col_r, col_d = st.columns([1, 1, 1, 2])
    with col_ra:
        ra_deg = st.number_input(
            "RA (deg)",
            min_value=0.0,
            max_value=360.0,
            value=0.0,
            step=0.001,
            format="%.6f",
            key="coord_ra",
        )
    with col_dec:
        dec_deg = st.number_input(
            "Dec (deg)",
            min_value=-90.0,
            max_value=90.0,
            value=0.0,
            step=0.001,
            format="%.6f",
            key="coord_dec",
        )
    with col_r:
        coord_radius = st.number_input(
            "Search radius (arcsec)",
            min_value=1.0,
            max_value=3600.0,
            value=60.0,
            step=10.0,
            key="coord_radius",
        )
    with col_d:
        coord_dest = st.text_input(
            "Download to",
            value=_get(current, "data_directory", "./data"),
            key="coord_dest",
        )

    _render_aladin(ra_deg, dec_deg, float(coord_radius), label=f"RA={ra_deg:.4f}, Dec={dec_deg:.4f}")

    if st.button("Search MAST", key="search_coord"):
        with st.spinner(f"Searching MAST at RA={ra_deg:.6f}, Dec={dec_deg:.6f}…"):
            try:
                st.session_state["mast_obs_coord"] = search_by_coordinates(
                    ra_deg, dec_deg, radius_arcsec=float(coord_radius)
                )
            except Exception as exc:
                st.session_state.pop("mast_obs_coord", None)
                st.error(f"MAST search failed: {exc}")

    if "mast_obs_coord" in st.session_state:
        downloaded_path = _show_search_results(
            st.session_state["mast_obs_coord"], coord_dest, "coord"
        )
        if downloaded_path:
            st.session_state["resolved_data_directory"] = downloaded_path

    new_config["data_directory"] = st.session_state.get("resolved_data_directory", coord_dest)

elif data_source_mode == "MAST lookup by program ID":
    col_p, col_d = st.columns([2, 3])
    with col_p:
        proposal_input = st.text_input(
            "Program ID(s)",
            placeholder="e.g. 2756, or 2756, 1837 for multiple",
            help="One or more program IDs, separated by commas or spaces.",
        )
    with col_d:
        prop_dest = st.text_input(
            "Download to",
            value=_get(current, "data_directory", "./data"),
            key="prop_dest",
        )

    if st.button("Search MAST", key="search_proposal"):
        if not proposal_input.strip():
            st.error("Enter at least one program ID first.")
        else:
            with st.spinner(f"Searching MAST for proposal(s) {proposal_input}…"):
                try:
                    st.session_state["mast_obs_proposal"] = search_by_proposal(proposal_input.strip())
                except Exception as exc:
                    st.session_state.pop("mast_obs_proposal", None)
                    st.error(f"MAST search failed: {exc}")

    if "mast_obs_proposal" in st.session_state:
        downloaded_path = _show_search_results(
            st.session_state["mast_obs_proposal"], prop_dest, "proposal"
        )
        if downloaded_path:
            st.session_state["resolved_data_directory"] = downloaded_path

    new_config["data_directory"] = st.session_state.get("resolved_data_directory", prop_dest)

else:  # Use existing directory
    new_config["data_directory"] = st.text_input(
        "Data directory (where uncal.fits files live)",
        value=_get(current, "data_directory", "./data"),
        help="The pipeline searches this directory recursively for *_uncal.fits files.",
    )


# 2. Output & grouping
st.header("2. Output & grouping")
col1, col2 = st.columns(2)
with col1:
    new_config["output_directory"] = st.text_input(
        "Output directory",
        value=_get(current, "output_directory", "."),
    )
    new_config["custom_name"] = st.text_input(
        "Custom name (used when 'combine all' is selected)",
        value=_get(current, "custom_name", "Combined_Observation"),
    )
with col2:
    grouping_modes = ["By program ID", "By subdirectory", "Combine all"]
    if _get(current, "group_by_directory", False):
        default_mode = "By subdirectory"
    elif _get(current, "combine_observations", False):
        default_mode = "Combine all"
    else:
        default_mode = "By program ID"
    grouping_mode = st.radio(
        "Grouping",
        grouping_modes,
        index=grouping_modes.index(default_mode),
        help="How to split your uncal files into independent pipeline runs.",
    )
    new_config["group_by_directory"] = grouping_mode == "By subdirectory"
    new_config["combine_observations"] = grouping_mode == "Combine all"


# 3. Calibration steps (skip toggles)
st.header("3. Calibration steps")
st.caption("Check a step to skip it. Unchecked steps run normally.")
current_skip = set(_get(current, "skip_steps", []) or [])
skip_cols = st.columns(2)
new_skip = []
for i, step in enumerate(PIPELINE_STEPS):
    with skip_cols[i % 2]:
        if st.checkbox(f"Skip {step}", value=step in current_skip, key=f"skip_{step}"):
            new_skip.append(step)
new_config["skip_steps"] = new_skip

new_config["wisp_directory"] = st.text_input(
    "WISP templates directory (required for wisp_subtraction)",
    value=_get(current, "wisp_directory", ""),
    help="Download v3 templates from https://stsci.box.com/s/1bymvf1lkrqbdn9rnkluzqk30e8o2bne",
)


# 4. Performance
st.header("4. Performance (parallel workers per stage)")
nproc_cols = st.columns(3)
nproc_fields = [
    ("stage1_nproc", "Stage 1"),
    ("stage2_nproc", "Stage 2"),
    ("fnoise_nproc", "1/f noise (rate)"),
    ("wisp_nproc", "WISP subtraction"),
    ("cfnoise_nproc", "1/f noise (cal)"),
    ("bkg_nproc", "Background subtraction"),
]
for i, (key, label) in enumerate(nproc_fields):
    with nproc_cols[i % 3]:
        new_config[key] = st.number_input(
            f"{label} (nproc)",
            min_value=1,
            max_value=128,
            value=int(_get(current, key, 8)),
            step=1,
        )

with st.expander("Advanced stage 3 options"):
    new_config["stage3_use_multiprocessing"] = st.checkbox(
        "Use multiprocessing across filters in stage 3",
        value=bool(_get(current, "stage3_use_multiprocessing", True)),
    )
    new_config["min_processes"] = st.number_input(
        "Minimum parallel processes for stage 3",
        min_value=1,
        max_value=32,
        value=int(_get(current, "min_processes", 8)),
    )
    new_config["outlier_in_memory"] = st.checkbox(
        "Outlier detection in memory",
        value=bool(_get(current, "outlier_in_memory", True)),
    )
    new_config["resample_in_memory"] = st.checkbox(
        "Resample in memory",
        value=bool(_get(current, "resample_in_memory", True)),
    )
    col_a, col_b = st.columns(2)
    with col_a:
        new_config["pixel_scale"] = st.number_input(
            "Pixel scale (arcsec)",
            min_value=0.001,
            max_value=1.0,
            value=float(_get(current, "pixel_scale", 0.02)),
            step=0.005,
            format="%.4f",
        )
        new_config["pixfrac"] = st.number_input(
            "pixfrac",
            min_value=0.01,
            max_value=1.0,
            value=float(_get(current, "pixfrac", 0.75)),
            step=0.05,
            format="%.2f",
        )
    with col_b:
        new_config["rotation"] = st.number_input(
            "Rotation (degrees; 0 = North up)",
            value=float(_get(current, "rotation", 0.0)),
            step=1.0,
            format="%.2f",
        )
        new_config["res_kernel"] = st.selectbox(
            "Resample kernel",
            ["square", "gaussian", "point", "turbo", "lanczos2", "lanczos3"],
            index=["square", "gaussian", "point", "turbo", "lanczos2", "lanczos3"].index(
                _get(current, "res_kernel", "square")
            ),
        )

with st.expander("Tweakreg and skymatch"):
    new_config["external_reference"] = st.text_input(
        "External reference catalog (CSV with RA,DEC, or 'GAIADR3')",
        value=_get(current, "external_reference", ""),
    )
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        new_config["starfinder"] = st.selectbox(
            "Starfinder",
            ["segmentation", "iraf", "dao"],
            index=["segmentation", "iraf", "dao"].index(
                _get(current, "starfinder", "segmentation")
            ),
        )
        new_config["snr_threshold"] = st.number_input(
            "SNR threshold",
            min_value=0.1,
            max_value=100.0,
            value=float(_get(current, "snr_threshold", 5.0)),
            step=0.5,
        )
    with col_t2:
        new_config["abs_fitgeometry"] = st.selectbox(
            "abs_fitgeometry",
            ["rshift", "shift", "rscale", "general"],
            index=["rshift", "shift", "rscale", "general"].index(
                _get(current, "abs_fitgeometry", "rshift")
            ),
        )
        new_config["fitgeometry"] = st.selectbox(
            "fitgeometry",
            ["rshift", "shift", "rscale", "general"],
            index=["rshift", "shift", "rscale", "general"].index(
                _get(current, "fitgeometry", "rshift")
            ),
        )
    new_config["skymethod"] = st.selectbox(
        "skymethod",
        ["match", "globalmin", "localmin", "globalmin+match"],
        index=["match", "globalmin", "localmin", "globalmin+match"].index(
            _get(current, "skymethod", "match")
        ),
    )

with st.expander("Extract i2d extensions"):
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        new_config["extract_sci"] = st.checkbox("Extract SCI", value=bool(_get(current, "extract_sci", True)))
        new_config["extract_err"] = st.checkbox("Extract ERR", value=bool(_get(current, "extract_err", True)))
        new_config["extract_wht"] = st.checkbox("Extract WHT", value=bool(_get(current, "extract_wht", True)))
        new_config["extract_con"] = st.checkbox("Extract CON", value=bool(_get(current, "extract_con", False)))
    with col_e2:
        new_config["extract_var_poisson"] = st.checkbox("Extract VAR_POISSON", value=bool(_get(current, "extract_var_poisson", False)))
        new_config["extract_var_rnoise"] = st.checkbox("Extract VAR_RNOISE", value=bool(_get(current, "extract_var_rnoise", False)))
        new_config["extract_var_flat"] = st.checkbox("Extract VAR_FLAT", value=bool(_get(current, "extract_var_flat", False)))


# 5. CRDS
st.header("5. CRDS")
new_config["crds_path"] = st.text_input(
    "CRDS cache path",
    value=_get(current, "crds_path", "~/crds_cache"),
)
new_config["crds_server_url"] = st.text_input(
    "CRDS server URL",
    value=_get(current, "crds_server_url", "https://jwst-crds.stsci.edu"),
)


# Preserve any keys we didn't render so we don't accidentally drop them.
for key, value in current.items():
    if key not in new_config:
        new_config[key] = value


st.divider()

errors, warnings = validate_config(new_config)
for message in warnings:
    st.warning(message)
for message in errors:
    st.error(message)

left, middle, right = st.columns([1, 1, 3])
with left:
    if st.button("Save config.yaml"):
        save_config(new_config)
        st.success(f"Saved {CONFIG_PATH}")
with middle:
    run_clicked = st.button("Save & Run pipeline ▶", type="primary", disabled=bool(errors))
    if run_clicked:
        save_config(new_config)
        st.info(f"Saved {CONFIG_PATH}. Starting pipeline…")
        run_pipeline_streaming()
with right:
    st.caption(
        "Save & Run writes config.yaml, then runs young_pipeline.sh and streams "
        "its output above. Per-stage detail still lands in <output>/<obs>/logs/."
    )
