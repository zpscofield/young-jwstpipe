#!/usr/bin/env python3
"""Print one value from config.yaml for young_pipeline.sh.

Replaces the yq/jq dependency. Output is shaped for shell consumption:

  strings          printed as-is, no quotes
  numbers/bools    JSON style (16, 0.75, true, false)
  null / missing   empty line
  lists            one item per line
  dicts            single-line JSON

Usage: python config_get.py <config.yaml> <key>
"""
import json
import sys

import yaml


def render_scalar(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: config_get.py <config.yaml> <key>", file=sys.stderr)
        return 2

    config_file, key = sys.argv[1], sys.argv[2]
    with open(config_file, "r") as f:
        config = yaml.safe_load(f) or {}

    value = config.get(key)
    if isinstance(value, list):
        for item in value:
            print(render_scalar(item))
    else:
        print(render_scalar(value))
    return 0


if __name__ == "__main__":
    sys.exit(main())
