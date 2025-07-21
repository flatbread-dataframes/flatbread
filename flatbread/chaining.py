import functools
from typing import Callable

import pandas as pd

from flatbread import DEFAULTS


def get_data_mask(index, ignore_keys):
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

    return pd.Series(result, index=index)


def tag_labels(transform: str) -> Callable:
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
    def set_nested_key(data: dict, keys: list[str], value) -> None:
        """Set a value in a nested dictionary structure."""
        if len(keys) == 1:
            data[keys[0]] = value
        else:
            key = keys[0]
            if key not in data:
                data[key] = {}
            set_nested_key(data[key], keys[1:], value)

    def get_nested_key(data: dict, keys: list[str]) -> set:
        """Get a value from nested dictionary, returning empty set if not found."""
        for key in keys:
            if key in data:
                data = data[key]
            else:
                return set()
        return data if isinstance(data, set) else set(data) if data else set()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
            # Get transform configuration from transforms section
            transform_config = DEFAULTS.get('transforms', {}).get(transform, {})
            key_label_params = transform_config.get('key_labels', [])

            # Extract the actual label values from function parameters
            labels_to_track = []
            for param_name in key_label_params:
                if param_name in kwargs and kwargs[param_name] is not None:
                    labels_to_track.append(kwargs[param_name])

            # Execute the original function
            result = func(df, *args, **kwargs)

            # Get existing tracked labels for this transform
            existing_labels = get_nested_key(
                result.attrs,
                ['flatbread', 'labels', transform]
            )

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
