"""Introspect JWST pipeline step parameters for the installed jwst version.

The Streamlit UI uses this to offer guided per-step parameter overrides
(dropdowns for step / parameter / typed value) and a read-only reference,
without hardcoding any parameter list. Everything is derived at runtime
from each step's configobj ``spec``, so it tracks whatever jwst is
installed.
"""

from __future__ import annotations

import re
from typing import Any


# A spec line looks like:
#   rejection_threshold = float(default=4.0,min=0) # CR sigma rejection threshold
#   maximum_cores = string(default='1') # cores for multiprocessing...
#   flag_4_neighbors = boolean(default=True) # flag the four neighbors
#   weighting = option('optimal','unweighted',default='optimal') # ...
_SPEC_LINE = re.compile(
    r"^\s*(?P<name>\w+)\s*=\s*(?P<type>\w+)\((?P<args>.*?)\)\s*(?:#\s*(?P<desc>.*))?$"
)

# Steps that are pure plumbing / already exposed elsewhere, or that would be
# dangerous to override blindly. We still list their params in the reference
# but they are otherwise harmless to expose.
_PIPELINE_IMPORTS = {
    "stage1": ("jwst.pipeline", "Detector1Pipeline"),
    "stage2": ("jwst.pipeline", "Image2Pipeline"),
    "stage3": ("jwst.pipeline", "Image3Pipeline"),
}


def _coerce_default(type_name: str, raw: str) -> Any:
    """Convert a spec default string to a Python value."""
    if raw is None:
        return None
    raw = raw.strip()
    if raw in ("None", ""):
        return None
    if type_name in ("float",):
        try:
            return float(raw)
        except ValueError:
            return None
    if type_name in ("integer", "int"):
        try:
            return int(raw)
        except ValueError:
            return None
    if type_name in ("boolean", "bool"):
        return raw.lower() in ("true", "1", "yes")
    # string / option / anything else: strip surrounding quotes
    return raw.strip("'\"")


def _parse_args(type_name: str, args: str) -> dict:
    """Parse the inside of a spec type call into default/options/min/max."""
    info: dict[str, Any] = {"default": None, "options": None, "min": None, "max": None}

    # default=...
    m = re.search(r"default\s*=\s*([^,]+?)\s*(?:,|$)", args)
    if m:
        info["default"] = _coerce_default(type_name, m.group(1))

    # min / max (numeric constraints)
    for bound in ("min", "max"):
        mb = re.search(rf"{bound}\s*=\s*([-\d.eE]+)", args)
        if mb:
            try:
                info[bound] = float(mb.group(1))
            except ValueError:
                pass

    # option('a','b',default='b') -> the positional quoted values are choices
    if type_name == "option":
        choices = re.findall(r"'([^']*)'|\"([^\"]*)\"", args)
        opts = [a or b for a, b in choices]
        # The default may also appear quoted; keep all distinct quoted tokens
        # that are not the default keyword value.
        default_val = info["default"]
        opts = [o for o in opts if o != default_val] if default_val is not None else opts
        if default_val is not None:
            opts = [default_val] + opts
        info["options"] = opts or None

    return info


def parse_spec(spec: str) -> dict:
    """Parse a step's configobj spec string into {param: {type, default, ...}}."""
    params: dict[str, dict] = {}
    if not spec:
        return params
    for line in spec.splitlines():
        m = _SPEC_LINE.match(line)
        if not m:
            continue
        name = m.group("name")
        type_name = m.group("type")
        parsed = _parse_args(type_name, m.group("args") or "")
        params[name] = {
            "type": type_name,
            "default": parsed["default"],
            "options": parsed["options"],
            "min": parsed["min"],
            "max": parsed["max"],
            "desc": (m.group("desc") or "").strip(),
        }
    return params


def introspect_stage(stage: str) -> dict:
    """Return {step_name: {param: {type, default, options, min, max, desc}}}.

    stage is one of 'stage1', 'stage2', 'stage3'. Returns {} if the pipeline
    cannot be imported (e.g. jwst not installed in this environment).
    """
    if stage not in _PIPELINE_IMPORTS:
        return {}
    module_name, class_name = _PIPELINE_IMPORTS[stage]
    try:
        module = __import__(module_name, fromlist=[class_name])
        pipeline_cls = getattr(module, class_name)
    except Exception:
        return {}

    steps: dict[str, dict] = {}
    for step_name, step_cls in getattr(pipeline_cls, "step_defs", {}).items():
        spec = getattr(step_cls, "spec", "") or ""
        params = parse_spec(spec)
        # `skip` is a universal Step parameter inherited from the base class,
        # so it is not in each step's own spec. Surface it so users can skip
        # an individual sub-step. We don't claim a default here (the pipeline
        # sets per-step skip defaults); the override only applies if the user
        # explicitly adds it.
        params.setdefault(
            "skip",
            {"type": "boolean", "default": False, "options": None,
             "min": None, "max": None, "desc": "Skip this sub-step."},
        )
        steps[step_name] = params
    return steps


def jwst_version() -> str:
    try:
        import jwst

        return str(jwst.__version__)
    except Exception:
        return "unknown"
