"""Shared NIRCam filter metadata.

Pivot wavelengths in micrometers. Single source of truth shared between
pipeline_stage3.py (used for sorting filters longest-first) and
color_image.py (used for default hue assignments).
"""

FILTER_PIVOT_WAVELENGTHS_UM = {
    "F070W": 0.7,  "F090W": 0.9,  "F115W": 1.15, "F140M": 1.41, "F150W": 1.5,
    "F162M": 1.63, "F164N": 1.65, "F150W2": 1.69, "F182M": 1.85, "F187N": 1.87,
    "F200W": 2.0,  "F210M": 2.1,  "F212N": 2.12, "F250M": 2.5,  "F277W": 2.78,
    "F300M": 3.0,  "F323N": 3.24, "F322W2": 3.25, "F335M": 3.36, "F356W": 3.57,
    "F360M": 3.62, "F405N": 4.05, "F410M": 4.08, "F430M": 4.28, "F444W": 4.40,
    "F460M": 4.63, "F466N": 4.65, "F470N": 4.71, "F480M": 4.81,
}
