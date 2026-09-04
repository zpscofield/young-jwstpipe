"""Serialize config.yaml into commented, grouped sections.

The Streamlit UI saves config.yaml on every run. Plain ``yaml.safe_dump``
would write 100+ keys in a flat list and drop any organization, which is
unfriendly to anyone editing the file by hand instead of using the UI.

This module defines a fixed section layout and regenerates the section
header comments every time the file is written, so the organization is
preserved across UI saves. Comments a user adds by hand are not round-tripped
(the layout is regenerated), but the grouping is always present.
"""

from __future__ import annotations

import yaml


# Ordered (section title, [keys]) groups. Keys absent from a given config are
# simply skipped; any key not listed here lands in a trailing "Other" section
# so nothing is ever lost.
CONFIG_LAYOUT = [
    ("CRDS", ["crds_path", "crds_server_url"]),
    ("Paths", ["pipeline_directory", "data_directory", "output_directory"]),
    ("Output and grouping", ["combine_observations", "custom_name", "group_by_directory"]),
    ("Calibration steps to skip", ["skip_steps"]),
    ("Performance (parallel workers per stage)", [
        "stage1_nproc", "stage2_nproc", "wisp_nproc", "cfnoise_nproc",
        "bkg_nproc", "stage3_use_multiprocessing", "min_processes",
    ]),
    ("WISP correction", [
        "wisp_directory", "wisp_create_segmap", "wisp_seg_from_lw", "wisp_sigma",
        "wisp_npixels", "wisp_dilate_segmap", "wisp_save_segmap", "wisp_sub_wisp",
        "wisp_gauss_smooth_wisp", "wisp_gauss_stddev", "wisp_scale_wisp",
        "wisp_scale_method", "wisp_poly_degree", "wisp_factor_min",
        "wisp_factor_max", "wisp_factor_step", "wisp_min_wisp",
        "wisp_flag_wisp_thresh", "wisp_dq_val", "wisp_correct_rows",
        "wisp_correct_cols", "wisp_save_model", "wisp_plot", "wisp_show_plot",
    ]),
    ("1/f noise correction", [
        "cfnoise_whole_image", "cfnoise_threshold1", "cfnoise_threshold2",
        "cfnoise_npixels", "cfnoise_mask_size", "cfnoise_interp_step",
    ]),
    ("Background subtraction", [
        "plot_sky", "bkg_tier_nsigma", "bkg_tier_npixels", "bkg_tier_kernel_size",
        "bkg_tier_dilate_size", "bkg_faint_tiers", "bkg_ring_radius_in",
        "bkg_ring_width", "bkg_ring_clip_max_sigma", "bkg_ring_clip_box_size",
        "bkg_ring_clip_filter_size", "bkg_bg_box_size", "bkg_bg_filter_size",
        "bkg_bg_exclude_percentile", "bkg_bg_sigma", "bkg_plot_smooth",
        "bkg_interpolator", "bkg_dq_flags_to_mask",
    ]),
    ("Mosaic creation (stage 3)", [
        "reference_filter", "mosaic_footprint",
        "pixel_scale", "pixfrac", "rotation", "res_kernel", "resample_in_memory",
        "outlier_in_memory", "external_reference", "starfinder", "snr_threshold",
        "abs_fitgeometry", "fitgeometry", "skymethod",
    ]),
    ("Pipeline step overrides", [
        "stage1_step_overrides", "stage2_step_overrides", "stage3_step_overrides",
    ]),
    ("Output extensions (i2d)", [
        "extract_sci", "extract_err", "extract_con", "extract_wht",
        "extract_var_poisson", "extract_var_rnoise", "extract_var_flat",
    ]),
    ("Color image", [
        "color_image_enabled", "color_image_subtract_sky", "color_image_min_level",
        "color_image_max_quantile", "color_image_gamma", "color_image_filter_hues",
    ]),
]


def _dump_key(key: str, value) -> str:
    """Dump a single key/value to YAML text without a trailing newline."""
    return yaml.safe_dump(
        {key: value}, default_flow_style=False, sort_keys=False
    ).rstrip("\n")


def serialize_config(config: dict) -> str:
    """Return config.yaml text with grouped, commented sections."""
    lines = [
        "# YOUNG JWST Pipeline configuration",
        "# Edit values below, or configure everything from the UI (./run.sh).",
        "# Section headers are regenerated automatically when the file is saved.",
        "",
    ]
    written: set[str] = set()
    for title, keys in CONFIG_LAYOUT:
        present = [k for k in keys if k in config]
        if not present:
            continue
        lines.append(f"# --- {title} ---")
        for key in present:
            lines.append(_dump_key(key, config[key]))
            written.add(key)
        lines.append("")

    leftover = [k for k in config if k not in written]
    if leftover:
        lines.append("# --- Other settings ---")
        for key in leftover:
            lines.append(_dump_key(key, config[key]))
        lines.append("")

    return "\n".join(lines) + "\n"
