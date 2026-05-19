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

import base64
import re
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
    filter_products_by_summary_rows,
    resolve_target,
)
from mast_download import get_uncal_products, download_uncal_products
from color_image import (
    default_hues_for_filters,
    find_i2d_files,
    make_color_image,
    sort_filters_by_wavelength,
)
from nircam_filters import FILTER_PIVOT_WAVELENGTHS_UM


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.yaml"
PIPELINE_SCRIPT = REPO_ROOT / "young_pipeline.sh"
RESOURCES = REPO_ROOT / "resources"
MAX_LOG_LINES = 500


def _resolve_data_dir(text_dest: str) -> str:
    """Return the data_directory value for a given Data-directory text input.

    Prefers the path returned by the most recent successful download
    (stored as 'resolved_data_directory' in session state, absolute) when
    it still matches what the user has in the text box. If the user has
    edited the text box away from that path, drop the resolved entry so
    the new typed value wins.
    """
    resolved = st.session_state.get("resolved_data_directory")
    if resolved:
        try:
            typed = Path(text_dest).expanduser().resolve()
            cached = Path(resolved).expanduser().resolve()
            if typed == cached:
                return str(cached)
        except OSError:
            pass
        # The user edited the field; forget the resolved entry.
        st.session_state.pop("resolved_data_directory", None)
    return text_dest


@st.cache_data
def _data_uri(relative_path: str, mime: str) -> str:
    data = (RESOURCES / relative_path).read_bytes()
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


@st.cache_data
def _banner_html() -> str:
    bg = _data_uri("xlssc_parallel_jpeg.jpg", "image/jpeg")
    young_logo = _data_uri("younglogowhite.png", "image/png")
    yonsei_logo = _data_uri("transparent_yonsei.png", "image/png")
    jwst_logo = _data_uri("500px-JWST_decal.svg.png", "image/png")
    # 100vw + -50vw / 50% left is the standard "full bleed" trick that
    # breaks out of Streamlit's block-container padding so the banner spans
    # the entire browser width on wide screens. mask-image fades the image
    # itself to transparent at the bottom, which works in any theme
    # (light/dark/system) because the page background shows through.
    return f"""
    <div style="
        position: relative;
        width: 100vw;
        left: 50%;
        right: 50%;
        margin-left: -50vw;
        margin-right: -50vw;
        margin-top: -4rem;
        margin-bottom: -1.5rem;
        height: 460px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.18);
    ">
        <div style="
            position: absolute;
            inset: 0;
            background-image: url('{bg}');
            background-size: cover;
            background-position: center 70%;
        "></div>
        <div style="
            position: absolute;
            inset: 0;
        "></div>
        <div style="
            position: absolute;
            top: 62px;
            left: 36px;
            right: 200px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 24px;
            z-index: 2;
        ">
            <div style="color: #ffffff; max-width: 55%; text-shadow: 0 2px 10px rgba(0,0,0,0.75);">
                <h1 style="font-size: 2.5rem; margin: 0; font-weight: 700; line-height: 1.1;">
                    YOUNG JWST Calibration Pipeline
                </h1>
                <p style="font-size: 1.05rem; margin-top: 10px; opacity: 0.95;">
                    Search MAST for JWST NIRCam data, configure the calibration pipeline, and run it — all from this page.
                </p>
            </div>
            <div style="
                display: flex;
                flex-direction: row;
                gap: 22px;
                align-items: center;
                flex-shrink: 0;
                margin-top: -10px;
            ">
                <img src="{young_logo}"  style="height: 56px; object-fit: contain; filter: drop-shadow(0 2px 6px rgba(0,0,0,0.6));" alt="YOUNG">
                <img src="{yonsei_logo}" style="height: 56px; object-fit: contain; filter: drop-shadow(0 2px 6px rgba(0,0,0,0.6));" alt="Yonsei">
                <img src="{jwst_logo}"   style="height: 56px; object-fit: contain; filter: drop-shadow(0 2px 6px rgba(0,0,0,0.6));" alt="JWST">
            </div>
        </div>
    </div>
    """

PIPELINE_STEPS = [
    "download_uncal_references",
    "stage1",
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


PATH_FIELDS = (
    "crds_path",
    "data_directory",
    "output_directory",
    "wisp_directory",
    "pipeline_directory",
)


def _expand_path_str(value):
    """Expand a leading ~ in a string. Leave non-strings and empty values alone."""
    if isinstance(value, str) and value:
        return str(Path(value).expanduser())
    return value


def save_config(config: dict) -> None:
    # Expand ~ in known path fields so the on-disk config has concrete paths.
    # If we did not do this, '~/crds_cache' would be passed through to the
    # pipeline scripts and CRDS would create a literal directory called '~'.
    expanded = dict(config)
    for field in PATH_FIELDS:
        if field in expanded:
            expanded[field] = _expand_path_str(expanded[field])
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(expanded, f, sort_keys=False, default_flow_style=False)


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


LOG_PANEL_HEIGHT_PX = 500


def _log_panel_html(lines: list[str], max_height_px: int = LOG_PANEL_HEIGHT_PX) -> str:
    body = (
        "\n".join(lines)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body {{ margin: 0; padding: 0; }}
    #log {{
      height: {max_height_px}px;
      max-height: {max_height_px}px;
      overflow-y: auto;
      font-family: ui-monospace, Menlo, Consolas, monospace;
      font-size: 0.85rem;
      line-height: 1.45;
      padding: 12px;
      background: #0d1117;
      color: #d1d9e0;
      border-radius: 6px;
      white-space: pre;
      box-sizing: border-box;
    }}
  </style>
</head>
<body>
  <pre id="log">{body}</pre>
  <script>
    // Auto-scroll to the bottom so the newest line is always visible.
    const el = document.getElementById('log');
    if (el) el.scrollTop = el.scrollHeight;
  </script>
</body>
</html>"""


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

        # def _render_log():
        #     with log_placeholder.container():
        #         components.html(
        #             _log_panel_html(lines[-MAX_LOG_LINES:]),
        #             height=LOG_PANEL_HEIGHT_PX + 20,
        #         )
        def _render_log():
            with log_placeholder.container():
                st.iframe(
                    _log_panel_html(lines[-MAX_LOG_LINES:]),
                    height=LOG_PANEL_HEIGHT_PX + 20,
                )

        # Detect tqdm-style progress lines (e.g. "  3%|▎         | 1/32 [...]"
        # or "Processing Filters:  25%|██▌      | 1/4 [...]") so consecutive
        # updates overwrite each other in the log instead of piling up as
        # separate lines, the way they would in a real terminal. The `%|`
        # marker can appear after an optional `desc:` prefix, so we search
        # anywhere in the line rather than anchoring to the start.
        tqdm_line = re.compile(r"\d+%\|")

        assert process.stdout is not None
        for line in process.stdout:
            stripped = line.rstrip("\n")
            if lines and tqdm_line.search(stripped) and tqdm_line.search(lines[-1]):
                lines[-1] = stripped
            else:
                lines.append(stripped)
            _render_log()

        return_code = process.wait()

        if return_code == 0:
            status.update(label="Pipeline finished successfully.", state="complete")
        else:
            status.update(
                label=f"Pipeline exited with code {return_code}. See log above.",
                state="error",
            )
        return return_code


def _filters_in_observation(output_dir: str, obs_name: str) -> list[str]:
    """Return filters present in <output_dir>/<obs_name>, sorted by wavelength."""
    obs_dir = Path(output_dir).expanduser() / obs_name
    filter_paths = find_i2d_files(obs_dir, obs_name)
    return sort_filters_by_wavelength(filter_paths.keys())


@st.cache_data(show_spinner=False)
def _filters_in_uncal_dir(data_dir: str, dir_mtime: float, n_files: int) -> list[str]:
    """Return filters found in *_uncal.fits primary headers, sorted by wavelength.

    The dir_mtime and n_files args are cache-invalidation signals only — they
    let Streamlit reuse the result until files are added, removed, or modified.
    """
    from astropy.io import fits  # local import: only paid when this helper runs

    path = Path(data_dir).expanduser()
    if not path.is_dir():
        return []
    filters: set[str] = set()
    for fits_path in path.glob("*_uncal.fits"):
        try:
            hdr = fits.getheader(str(fits_path), ext=0)
        except Exception:
            continue
        filt = hdr.get("FILTER")
        if filt:
            filters.add(str(filt))
    return sort_filters_by_wavelength(filters)


def _scan_uncal_filters(data_dir: str) -> list[str]:
    """Cached uncal-header scan; returns [] if data_dir is empty/missing."""
    if not data_dir:
        return []
    path = Path(data_dir).expanduser()
    if not path.is_dir():
        return []
    n_files = sum(1 for _ in path.glob("*_uncal.fits"))
    if n_files == 0:
        return []
    return _filters_in_uncal_dir(str(path), path.stat().st_mtime, n_files)


def _expected_obs_names(config: dict, data_dir: str) -> list[str]:
    """Predict the observation directory names the pipeline would create.

    Mirrors the logic in utils/get_obs_info.py:
      - group_by_directory=True : one obs per immediate subdir with uncal files
      - combine_observations=True (overridden by above) : one obs = custom_name
      - otherwise : one obs per PROGRAM id read from the uncal headers
    """
    if not data_dir:
        return []
    path = Path(data_dir).expanduser()
    if not path.is_dir():
        return []

    if config.get("group_by_directory"):
        names = []
        for sub in sorted(path.iterdir()):
            if not sub.is_dir():
                continue
            if any(sub.rglob("*_uncal.fits")):
                cleaned = sub.name.replace(" ", "_").replace(".", "_")
                names.append(f"Output_{cleaned}")
        return names

    if config.get("combine_observations"):
        custom = str(config.get("custom_name") or "").strip()
        return [custom or "Combined_Observation"]

    # Default: group by PROGRAM id — requires reading primary headers.
    from astropy.io import fits

    programs: set[str] = set()
    for f in path.glob("*_uncal.fits"):
        try:
            hdr = fits.getheader(str(f), ext=0)
        except Exception:
            continue
        pid = str(hdr.get("PROGRAM", "00000")).strip() or "00000"
        programs.add(pid)
    return sorted(programs)


def _expected_obs_with_i2d(config: dict, data_dir: str, output_dir: str) -> list[str]:
    """Of the obs names this config would create, which already have i2d files?"""
    expected = _expected_obs_names(config, data_dir)
    if not expected:
        return []
    base = Path(output_dir).expanduser()
    if not base.is_dir():
        return []
    matches: list[str] = []
    for name in expected:
        obs_dir = base / name
        if obs_dir.is_dir() and any(
            obs_dir.glob("stage3_output/*/output_files/*_i2d.fits")
        ):
            matches.append(name)
    return matches


def _get(config: dict, key: str, default):
    value = config.get(key)
    return value if value is not None else default


st.set_page_config(page_title="YOUNG JWST Pipeline", page_icon="🔭", layout="wide")

# Make Streamlit's own toolbar transparent and let pointer-events pass
# through the header bar to the banner underneath. The toolbar buttons
# themselves stay clickable. Drop every flavor of top padding the main
# block-container might have so the banner can start at the very top.
st.markdown(
    """
    <style>
    [data-testid="stHeader"],
    .stApp > header {
        background: transparent !important;
        pointer-events: none !important;
    }
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stHeader"] button {
        pointer-events: auto !important;
    }
    [data-testid="stMain"] > div.block-container,
    section.main > div.block-container,
    .main .block-container,
    [data-testid="stMainBlockContainer"] {
        padding-top: 0 !important;
    }
    .stApp { padding-top: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(_banner_html(), unsafe_allow_html=True)

current = load_config()
new_config = dict(current)


# st.divider()
# st.caption(f"Editing {CONFIG_PATH}")
st.markdown(
    f'<p style="text-align: center; font-size: 0.8rem; color: gray;">Editing {CONFIG_PATH}</p>',
    unsafe_allow_html=True,
)


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


def _render_aladin(
    ra_deg: float,
    dec_deg: float,
    radius_arcsec: float,
    label: str = "",
    height: int = 620,
    session_key: str = "aladin",
):
    """Render an Aladin Lite viewer centered on (ra, dec) with a circle for the radius.

    Shows a 'Loading sky view...' overlay until Aladin finishes initializing,
    and an error message if loading the script or initializing the viewer
    fails (or takes more than ~15 seconds). A 'Reload sky view' button
    below the viewer lets the user retry by forcing a fresh render.
    """
    retry_token = st.session_state.get(f"{session_key}_retry", 0)
    fov_deg = max(min(4 * radius_arcsec / 3600.0, 5.0), 0.05)
    safe_label = label.replace("'", "").replace('"', "")
    inner_height = max(height - 20, 200)

    st.markdown("##### Sky View")
    html = f"""<!doctype html>
<html data-retry-token="{retry_token}">
<head>
  <meta charset="utf-8" />
  <script src="https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js"></script>
  <style>
    html, body {{ margin: 0; padding: 0; background: #000; color: #ddd; font-family: system-ui, sans-serif; }}
    #aladin-lite-div {{ width: 100%; height: {inner_height}px; }}
    #aladin-status {{
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      gap: 12px;
      background: rgba(0,0,0,0.85);
      z-index: 10;
      text-align: center;
      padding: 20px;
      font-size: 0.95rem;
    }}
    #aladin-status.error {{ background: rgba(70,15,15,0.92); color: #ffd9d9; }}
    .spinner {{
      width: 32px; height: 32px; border-radius: 50%;
      border: 3px solid rgba(255,255,255,0.2);
      border-top-color: #6ec3ff;
      animation: spin 0.9s linear infinite;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  </style>
</head>
<body>
  <div id="aladin-lite-div"></div>
  <div id="aladin-status">
    <div class="spinner"></div>
    <div id="aladin-status-text">Loading sky view…</div>
  </div>
  <script>
    const statusEl = document.getElementById('aladin-status');
    const statusText = document.getElementById('aladin-status-text');
    const showError = (msg) => {{
      statusEl.classList.add('error');
      statusEl.innerHTML = '<div>⚠️ ' + msg + '</div><div style="font-size:0.85rem;opacity:0.8;">Click \\'Reload sky view\\' below to retry.</div>';
    }};
    const loadTimeout = setTimeout(() => {{
      if (statusEl && statusEl.style.display !== 'none') {{
        showError('Sky view took too long to load.');
      }}
    }}, 15000);

    if (typeof A === 'undefined') {{
      clearTimeout(loadTimeout);
      showError('Could not reach the Aladin Lite CDN.');
    }} else {{
      A.init.then(() => {{
        try {{
          const aladin = A.aladin('#aladin-lite-div', {{
            target: '{ra_deg} {dec_deg}',
            fov: {fov_deg},
            survey: 'P/PanSTARRS/DR1/color-z-zg-g',
            showLayersControl: true,
            showGotoControl: true,
            showZoomControl: true,
            showFullscreenControl: true,
            showCooGrid: false
          }});
          const overlay = A.graphicOverlay({{color: 'cyan', lineWidth: 2}});
          aladin.addOverlay(overlay);
          overlay.add(A.circle({ra_deg}, {dec_deg}, {radius_arcsec / 3600.0}));
          const cat = A.catalog({{name: 'Search center', sourceSize: 36, color: 'lime'}});
          aladin.addCatalog(cat);
          cat.addSources([A.source({ra_deg}, {dec_deg}, {{name: '{safe_label}'}})]);
          clearTimeout(loadTimeout);
          statusEl.style.display = 'none';
        }} catch (err) {{
          clearTimeout(loadTimeout);
          showError('Aladin initialization failed: ' + err.message);
        }}
      }}).catch(err => {{
        clearTimeout(loadTimeout);
        showError('Aladin initialization failed: ' + (err && err.message ? err.message : err));
      }});
    }}
  </script>
</body>
</html>"""
    # components.html(html, height=height)
    st.iframe(html, height=height)
    
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button(
            "↻ Reload sky view",
            key=f"{session_key}_reload",
            help="Force the viewer to re-fetch and re-initialize.",
            use_container_width=True,
        ):
            st.session_state[f"{session_key}_retry"] = retry_token + 1
            st.rerun()


def _show_search_results(observations, products, download_dir: str, session_key: str):
    """Display a summary table with selectable rows and a Download button.

    products is the pre-fetched UNCAL products table (from
    get_uncal_products). It's used for accurate file counts and as the
    source for downloading.
    """
    summary_rows = summarize(observations, uncal_products=products)
    total_frames = sum(row["n_frames"] for row in summary_rows)
    st.markdown(
        f"**Found {len(observations)} observations in "
        f"{len({row['program'] for row in summary_rows})} programs ({total_frames} uncal files).**"
    )
    st.caption("Tick rows to download just those program/filter combinations. Leave nothing ticked to download everything.")

    event = st.dataframe(
        summary_rows,
        hide_index=True,
        # use_container_width=True,
        width='stretch',
        on_select="rerun",
        selection_mode="multi-row",
        key=f"select_{session_key}",
    )
    selected_indices = list(event.selection.rows) if event and event.selection else []

    if selected_indices:
        selected_rows = [summary_rows[i] for i in selected_indices]
        to_download = filter_products_by_summary_rows(observations, products, selected_rows)
        selected_frames = sum(summary_rows[i]["n_frames"] for i in selected_indices)
        button_label = f"Download {selected_frames} selected files"
    else:
        to_download = products
        button_label = f"Download all {total_frames} files"

    if st.button(button_label, key=f"download_{session_key}", type="primary"):
        progress_bar = st.progress(0.0, text="Starting…")
        status_text = st.empty()

        def on_progress(i, total, filename, status):
            progress_bar.progress(i / total, text=f"[{i}/{total}] {filename} ({status})")
            if status == "failed":
                status_text.warning(f"Failed: {filename}")

        try:
            result = download_uncal_products(to_download, download_dir, progress=on_progress)
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
            "Data directory",
            value=_get(current, "data_directory", "./data"),
            key="target_dest",
            help="Files download here AND the pipeline reads from here. Change this to point at a new dataset.",
        )

    if st.button("Search MAST", key="search_target"):
        clean = target_name.strip()
        if not clean:
            st.error("Enter a target name first.")
        else:
            try:
                with st.spinner(f"Searching MAST for '{clean}'…"):
                    obs = search_by_target(clean, radius_arcsec=float(target_radius))
                with st.spinner(f"Counting uncal files for {len(obs)} observations…"):
                    products = get_uncal_products(obs)
                st.session_state["mast_obs_target"] = obs
                st.session_state["mast_products_target"] = products
                st.session_state["mast_target_name"] = clean
                st.session_state["mast_target_radius"] = float(target_radius)
                coords = resolve_target(clean)
                if coords is not None:
                    st.session_state["mast_target_coords"] = coords
                else:
                    st.session_state.pop("mast_target_coords", None)
            except Exception as exc:
                st.session_state.pop("mast_obs_target", None)
                st.session_state.pop("mast_products_target", None)
                st.error(f"MAST search failed: {exc}")

    has_results = "mast_obs_target" in st.session_state
    has_coords = "mast_target_coords" in st.session_state

    if has_results or has_coords:
        col_table, col_view = st.columns([1, 1])
        with col_table:
            if has_results:
                downloaded_path = _show_search_results(
                    st.session_state["mast_obs_target"],
                    st.session_state["mast_products_target"],
                    target_dest,
                    "target",
                )
                if downloaded_path:
                    st.session_state["resolved_data_directory"] = downloaded_path
        with col_view:
            if has_coords:
                ra, dec = st.session_state["mast_target_coords"]
                _render_aladin(
                    ra,
                    dec,
                    st.session_state.get("mast_target_radius", float(target_radius)),
                    label=st.session_state.get("mast_target_name", ""),
                    height=640,
                    session_key="aladin_target",
                )

    new_config["data_directory"] = _resolve_data_dir(target_dest)
    st.caption(f"Pipeline will read uncal files from: `{new_config['data_directory']}`")

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
            "Data directory",
            value=_get(current, "data_directory", "./data"),
            key="coord_dest",
            help="Files download here AND the pipeline reads from here. Change this to point at a new dataset.",
        )

    if st.button("Search MAST", key="search_coord"):
        try:
            with st.spinner(f"Searching MAST at RA={ra_deg:.6f}, Dec={dec_deg:.6f}…"):
                obs = search_by_coordinates(ra_deg, dec_deg, radius_arcsec=float(coord_radius))
            with st.spinner(f"Counting uncal files for {len(obs)} observations…"):
                products = get_uncal_products(obs)
            st.session_state["mast_obs_coord"] = obs
            st.session_state["mast_products_coord"] = products
        except Exception as exc:
            st.session_state.pop("mast_obs_coord", None)
            st.session_state.pop("mast_products_coord", None)
            st.error(f"MAST search failed: {exc}")

    col_table, col_view = st.columns([1, 1])
    with col_table:
        if "mast_obs_coord" in st.session_state:
            downloaded_path = _show_search_results(
                st.session_state["mast_obs_coord"],
                st.session_state["mast_products_coord"],
                coord_dest,
                "coord",
            )
            if downloaded_path:
                st.session_state["resolved_data_directory"] = downloaded_path
        else:
            st.info("Enter coordinates and click Search MAST to see what is available.")
    with col_view:
        _render_aladin(
            ra_deg,
            dec_deg,
            float(coord_radius),
            label=f"RA={ra_deg:.4f}, Dec={dec_deg:.4f}",
            height=640,
            session_key="aladin_coord",
        )

    new_config["data_directory"] = _resolve_data_dir(coord_dest)
    st.caption(f"Pipeline will read uncal files from: `{new_config['data_directory']}`")

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
            "Data directory",
            value=_get(current, "data_directory", "./data"),
            key="prop_dest",
            help="Files download here AND the pipeline reads from here. Change this to point at a new dataset.",
        )

    if st.button("Search MAST", key="search_proposal"):
        if not proposal_input.strip():
            st.error("Enter at least one program ID first.")
        else:
            try:
                with st.spinner(f"Searching MAST for proposal(s) {proposal_input}…"):
                    obs = search_by_proposal(proposal_input.strip())
                with st.spinner(f"Counting uncal files for {len(obs)} observations…"):
                    products = get_uncal_products(obs)
                st.session_state["mast_obs_proposal"] = obs
                st.session_state["mast_products_proposal"] = products
            except Exception as exc:
                st.session_state.pop("mast_obs_proposal", None)
                st.session_state.pop("mast_products_proposal", None)
                st.error(f"MAST search failed: {exc}")

    if "mast_obs_proposal" in st.session_state:
        downloaded_path = _show_search_results(
            st.session_state["mast_obs_proposal"],
            st.session_state["mast_products_proposal"],
            prop_dest,
            "proposal",
        )
        if downloaded_path:
            st.session_state["resolved_data_directory"] = downloaded_path

    new_config["data_directory"] = _resolve_data_dir(prop_dest)
    st.caption(f"Pipeline will read uncal files from: `{new_config['data_directory']}`")

else:  # Use existing directory
    new_config["data_directory"] = st.text_input(
        "Data directory (where uncal.fits files live)",
        value=_get(current, "data_directory", "./data"),
        help="The pipeline searches this directory recursively for *_uncal.fits files.",
    )
    st.caption(f"Pipeline will read uncal files from: `{new_config['data_directory']}`")


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
)
st.caption(
    "Download templates from "
    "[stsci.app.box.com](https://stsci.app.box.com/s/1bymvf1lkrqbdn9rnkluzqk30e8o2bne) "
    "and point the path above at the unzipped folder."
)


# 4. Performance
st.header("4. Performance (parallel workers per stage)")
nproc_cols = st.columns(3)
nproc_fields = [
    ("stage1_nproc", "Stage 1"),
    ("stage2_nproc", "Stage 2"),
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
        "External reference catalog",
        value=_get(current, "external_reference", ""),
        help=(
            "Either a path to a CSV with RA,DEC columns, or a catalog name "
            "the jwst pipeline knows about (e.g. GAIADR3). Type the name "
            "plain, without quotes."
        ),
        placeholder="GAIADR3   or   /path/to/refcat.csv",
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


# 6. Color image
st.header("6. Color image")
st.caption(
    "Build a single color TIFF from your stage 3 i2d mosaics. Each filter is "
    "also saved as a stretched grayscale TIFF so you can edit them in Photoshop."
)

new_config["color_image_enabled"] = st.checkbox(
    "Create color image after pipeline run",
    value=bool(_get(current, "color_image_enabled", False)),
    help="When checked, the color image is generated automatically for each observation at the end of the pipeline.",
)

new_config["color_image_subtract_sky"] = st.checkbox(
    "Auto-subtract residual sky per filter (recommended)",
    value=bool(_get(current, "color_image_subtract_sky", True)),
    help=(
        "Each filter's stage-3 mosaic still has a small residual sky pedestal "
        "after background subtraction. When this is on, a sigma-clipped median "
        "is subtracted from each filter so that 'sky' maps to the same black "
        "across all filters, removing color hue in empty regions."
    ),
)

ci_left, ci_right = st.columns(2)
with ci_left:
    new_config["color_image_min_level"] = st.number_input(
        "Stretch min level",
        min_value=0.0,
        max_value=10.0,
        value=float(_get(current, "color_image_min_level", 0.001)),
        step=0.001,
        format="%.4f",
        help="Pixel floor for the asinh stretch. Values at or below this become black.",
    )
    new_config["color_image_max_quantile"] = st.slider(
        "Stretch max quantile",
        min_value=0.95,
        max_value=1.0,
        value=float(_get(current, "color_image_max_quantile", 0.99999)),
        step=0.0001,
        format="%.5f",
        help="The pixel value at this quantile defines the stretch ceiling.",
    )
with ci_right:
    new_config["color_image_gamma"] = st.slider(
        "Gamma",
        min_value=0.5,
        max_value=4.0,
        value=float(_get(current, "color_image_gamma", 2.2)),
        step=0.1,
        help="Final gamma correction. Higher = darker midtones.",
    )

# Decide whether this section is in "pre-run" (configure hues for the
# upcoming pipeline) or "regenerate" (rebuild color image for an obs that
# already finished) mode.
#
# The trigger is whether the obs directory the current config WOULD create
# (custom_name, program ID, or per-subdir name) already exists with stage-3
# i2d files. That way, changing data_directory or custom_name immediately
# refocuses the section — old unrelated outputs in output_directory no
# longer pollute the dropdown.
ci_output_dir = new_config.get("output_directory") or "."
data_dir = new_config.get("data_directory") or ""
uncal_filters = _scan_uncal_filters(data_dir)
expected_with_i2d = _expected_obs_with_i2d(new_config, data_dir, ci_output_dir)

# Start from whatever hues are in the saved config; we'll layer the user's
# new picks on top before save.
saved_hues = dict(_get(current, "color_image_filter_hues", {}) or {})
new_hues = dict(saved_hues)

selected_obs: str | None = None
filters_present: list[str] = []
filters_source: str = ""  # one of "uncal", "i2d", or ""

if expected_with_i2d:
    # The pipeline's target output for this config already exists — switch
    # to regenerate mode using the authoritative i2d filter set.
    if len(expected_with_i2d) == 1:
        selected_obs = expected_with_i2d[0]
        st.caption(f"Observation: `{selected_obs}`")
    else:
        selected_obs = st.selectbox(
            "Observation to (re)generate",
            expected_with_i2d,
            key="color_image_obs_choice",
        )
    filters_present = _filters_in_observation(ci_output_dir, selected_obs)
    filters_source = "i2d"
elif uncal_filters:
    filters_present = uncal_filters
    filters_source = "uncal"

n_filters = len(filters_present)

if filters_source == "uncal":
    st.caption(
        f"{n_filters} filter(s) detected from uncal files in "
        f"`{data_dir}`: {', '.join(filters_present)}. Configure hues now — "
        "they'll be saved to config and applied automatically at the end "
        "of the pipeline run."
    )
elif filters_source == "i2d" and n_filters == 0:
    st.warning(f"No i2d.fits files found under {selected_obs}/stage3_output.")
elif not filters_present:
    st.info(
        "Point `data_directory` at uncal files (or run the pipeline first) "
        "to configure per-filter hues."
    )

if n_filters == 1:
    st.info(
        f"Only one filter ({filters_present[0]}) is available. "
        "The per-filter stretched TIFF will still be saved; no combined color image."
    )
elif n_filters == 2:
    st.caption(
        f"Two filters detected ({filters_present[0]} and {filters_present[1]}). "
        "Combined image will use the luminance-preserving recipe: "
        f"blue = {filters_present[0]}, red = {filters_present[1]}, green = mean."
    )
elif n_filters >= 3:
    st.caption(
        f"Set a hue (0°–240°) for each filter — defaults follow a wavelength "
        "ramp (240° = blue at the shortest wavelength, 0° = red at the longest)."
    )
    defaults = default_hues_for_filters(filters_present)
    # Render hue inputs in up-to-3 columns.
    cols = st.columns(min(3, n_filters))
    for i, filt in enumerate(filters_present):
        with cols[i % len(cols)]:
            wl = FILTER_PIVOT_WAVELENGTHS_UM.get(filt)
            label = f"{filt}" + (f" ({wl} µm)" if wl else "")
            default_value = float(saved_hues.get(filt, defaults.get(filt, 120.0)))
            new_hues[filt] = float(
                st.number_input(
                    label,
                    min_value=0.0,
                    max_value=240.0,
                    value=default_value,
                    step=5.0,
                    key=f"hue_{filt}",
                )
            )

new_config["color_image_filter_hues"] = new_hues

# Generate button + preview.
if selected_obs is not None and _filters_in_observation(ci_output_dir, selected_obs):
    if st.button("Generate color image now", key="color_image_generate"):
        save_config(new_config)
        obs_dir_path = Path(ci_output_dir).expanduser() / selected_obs
        with st.spinner(f"Generating color image for {selected_obs}…"):
            try:
                result = make_color_image(
                    obs_dir=obs_dir_path,
                    target=selected_obs,
                    min_level=float(new_config["color_image_min_level"]),
                    max_quantile=float(new_config["color_image_max_quantile"]),
                    gamma=float(new_config["color_image_gamma"]),
                    filter_hues=new_hues,
                    subtract_sky_per_filter=bool(new_config["color_image_subtract_sky"]),
                )
            except Exception as exc:
                st.error(f"Color image generation failed: {exc}")
                result = None

        if result is not None:
            preview = result.get("preview")
            if preview and Path(preview).exists():
                st.session_state[f"color_preview_{selected_obs}"] = str(preview)
            if result.get("tiff"):
                st.success(f"Saved color TIFF to `{result['tiff']}`")
            per_filter = result.get("per_filter_tiffs", {})
            sky_levels = result.get("sky_levels", {}) or {}
            if per_filter:
                with st.expander(f"Per-filter stretched TIFFs ({len(per_filter)})"):
                    for filt, path in per_filter.items():
                        if filt in sky_levels:
                            st.code(
                                f"{filt}  sky={sky_levels[filt]:+.5f}  {path}",
                                language=None,
                            )
                        else:
                            st.code(f"{filt}: {path}", language=None)

    preview_path = st.session_state.get(f"color_preview_{selected_obs}")
    if preview_path and Path(preview_path).exists():
        st.image(
            preview_path,
            caption=f"Color image preview — {selected_obs}",
            width=600,
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
with right:
    st.caption(
        "Save & Run writes config.yaml, then runs young_pipeline.sh and streams "
        "its output below. Per-stage detail still lands in <output>/<obs>/logs/."
    )

# Pipeline output renders here, OUTSIDE the column layout, so it uses the full page width.
if run_clicked:
    save_config(new_config)
    st.info(f"Saved {CONFIG_PATH}. Starting pipeline…")
    run_pipeline_streaming()
