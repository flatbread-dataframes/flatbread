import functools
from typing import Any, Callable

import pandas as pd

from flatbread import DEFAULTS
from flatbread.types import PandasObj, Axis


# region helpers
def set_nested_key(data: dict, keys: list[str], value: Any) -> None:
    """
    Set a value in a nested dictionary, creating intermediate dicts as needed.

    Parameters
    ----------
    data : dict
        The root dictionary to mutate.
    keys : list[str]
        Path of keys leading to the target location.
    value : Any
        Value to set at the final key.
    """
    if len(keys) == 1:
        data[keys[0]] = value
    else:
        key = keys[0]
        if key not in data:
            data[key] = {}
        set_nested_key(data[key], keys[1:], value)


def get_nested_key(data: dict, keys: list[str], default: Any = None) -> Any:
    """
    Get a value from a nested dictionary.

    Parameters
    ----------
    data : dict
        The root dictionary to traverse.
    keys : list[str]
        Path of keys to follow.
    default : Any
        Value to return if the path does not exist. Default is None.

    Returns
    -------
    Any
        The value at the key path, or ``default`` if not found.
    """
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data


# region mask
def get_data_mask(index: pd.Index, ignore_keys: list[str] | None):
    """
    Create a mask used for separating data from results of flatbread operations. The keys in `ignore_keys` determine which rows/columns need to be ignored. This can be used when chaining multiple flatbread operations.

    Parameters
    ----------
    index (pd.Index):
        The index used for determining if a row/column contains data or not.
    ignore_keys (list[str]):
        List of index keys indicating that a row/column is *not* a data column. If the index is a MultiIndex then a row/column will be ignored if the key is in the keys of the index, else a row/column will be ignored if it is equal to or a prefix of the key in the index.

    Returns
    -------
    pd.Index:
        Boolean index indicating which rows/columns refer to data.
    """
    if ignore_keys is None:
        return pd.Series(True, index=index)

    # Convert single string to list
    if isinstance(ignore_keys, str):
        ignore_keys = [ignore_keys]

    def should_keep(value):
        # direct match
        if value in ignore_keys:
            return False

        # check for prefix
        if isinstance(value, str):
            for key in ignore_keys:
                if isinstance(key, str) and value.startswith(key):
                    return False
        return True

    if isinstance(index, pd.MultiIndex):
        result = [all(should_keep(el) for el in idx) for idx in index]
    else:
        result = [should_keep(idx) for idx in index]

    result = pd.Series(result, index=index)
    if not result.any():
        raise ValueError(
            "All data has been masked. Check ignore_keys or prior transforms."
        )
    return result


# region ignore
def resolve_ignored_keys(data, transform_name, ignore_keys=None):
    """
    Collect keys to ignore for a transform based on config and tracked labels.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        Data with potential flatbread labels in attrs.
    transform_name : str
        Name of the current transform (e.g. 'totals', 'differences').
    ignore_keys : str | list[str] | None
        Additional keys to ignore, passed by the user.

    Returns
    -------
    list[str]
        Combined list of keys to ignore.
    """
    keys_to_ignore = []
    if isinstance(ignore_keys, str):
        keys_to_ignore.append(ignore_keys)
    elif isinstance(ignore_keys, list):
        keys_to_ignore.extend(ignore_keys)

    to_ignore = DEFAULTS['transforms'].get(transform_name, {}).get('ignore_transforms', [])

    # margin labels (totals, subtotals)
    tracked = data.attrs.get('flatbread', {}).get('labels', {})
    for transform in to_ignore:
        keys_to_ignore.extend(tracked.get(transform, []))

    # panel labels
    panels = get_nested_key(data.attrs, ['flatbread', 'panels']) or {}
    for label, meta in panels.items():
        if label != 'n' and meta.get('type') in to_ignore:
            keys_to_ignore.append(label)

    return keys_to_ignore


# region tag labels
def track_margin_labels(transform: str) -> Callable:
    """
    Tag labels produced by flatbread operations for tracking in chained operations.

    This decorator identifies which labels are produced by a transform so that
    future operations can make informed decisions about what to ignore. The labels
    to track are determined by the 'key_labels' configuration for the transform.

    Parameters
    ----------
    transform : str
        The transform name that corresponds to a section in the flatbread config
        (e.g., 'totals', 'percentages', 'differences').

    Returns
    -------
    Callable
        Decorated function that tracks its key labels in df.attrs.

    Notes
    -----
    Labels are stored in df.attrs under the structure:
    ```python
    {'flatbread': {
        'labels': {
            'percentages': ['pct'],
            'totals': ['Totals'],
            'differences': ['diff']
        }
    }}
    ```
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(data: PandasObj, *args, **kwargs) -> PandasObj:
            # Get transform configuration from transforms section
            transform_config = DEFAULTS.get('transforms', {}).get(transform, {})
            key_label_params = transform_config.get('key_labels', [])

            # Extract the actual label values from function parameters
            labels_to_track = []
            for param_name in key_label_params:
                if param_name in kwargs and kwargs[param_name] is not None:
                    labels_to_track.append(kwargs[param_name])

            # Execute the original function
            result = func(data, *args, **kwargs)

            # Get existing tracked labels for this transform
            existing_labels = get_nested_key(
                result.attrs,
                ['flatbread', 'labels', transform]
            ) or set()

            # Combine existing and new labels
            all_labels = existing_labels.union(labels_to_track)

            # Store updated labels in result attrs
            if not hasattr(result, 'attrs'):
                result.attrs = {}

            set_nested_key(
                result.attrs,
                ['flatbread', 'labels', transform],
                all_labels
            )

            return result

        return wrapper
    return decorator
