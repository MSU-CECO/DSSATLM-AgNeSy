"""
packages/dssatlm/src/dssatlm/utils.py
Shared utility functions.
"""

import json
import time


def get_current_time() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def dict_to_json_file(data: dict, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)

