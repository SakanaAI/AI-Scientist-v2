import os

import numpy as np


MAX_NESTED_KEYS = 50


def collect_npy_paths(value):
    """Collect unique experiment data paths from nested summary data."""
    paths = set()

    def visit(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key == "exp_results_npy_files" and isinstance(child, list):
                    paths.update(path for path in child if isinstance(path, str))
                else:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return sorted(paths)


def _describe_value(value):
    if isinstance(value, dict):
        keys = sorted(str(key) for key in value)
        description = {
            "type": "dict",
            "length": len(value),
            "keys": keys[:MAX_NESTED_KEYS],
        }
        if len(keys) > MAX_NESTED_KEYS:
            description["keys_truncated"] = len(keys) - MAX_NESTED_KEYS
        return description
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value)}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        description = {"type": type(value).__name__}
        if isinstance(value, str):
            description["value"] = value[:200]
        elif value is None or isinstance(value, (int, float, bool)):
            description["value"] = value
        return description
    return {"type": type(value).__name__}


def _resolved_relative_path(base_folder, path):
    root = os.path.realpath(base_folder)
    resolved = os.path.realpath(os.path.join(root, path))
    try:
        if os.path.commonpath([root, resolved]) != root:
            return None
    except ValueError:
        return None
    return resolved


def describe_npy_file(base_folder, path):
    """Return a compact description of an experiment data file."""
    resolved = _resolved_relative_path(base_folder, path)
    if resolved is None:
        return {"error": "path is outside the experiment directory"}
    if not os.path.isfile(resolved):
        return {"error": "file does not exist"}

    try:
        data = np.load(resolved, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == () and data.dtype == object:
            data = data.item()
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}

    if isinstance(data, dict):
        return {
            "top_level_type": "dict",
            "top_level_keys": {
                str(key): _describe_value(value)
                for key, value in sorted(data.items(), key=lambda item: str(item[0]))
            },
        }
    return {"top_level_type": type(data).__name__, "value": _describe_value(data)}


def build_numpy_data_manifest(base_folder, summaries):
    """Describe every referenced .npy file so the plotting model can inspect its schema."""
    return {
        path: describe_npy_file(base_folder, path)
        for path in collect_npy_paths(summaries)
    }
