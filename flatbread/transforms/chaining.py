"""Utilities for tracking flatbread labels across chained operations."""

from typing import Any

import pandas as pd

from flatbread import DEFAULTS
from flatbread.types import PandasObj


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
    return result


# region margins
def resolve_margin_labels(attrs: dict | None = None) -> set[str]:
    """
    Collect all margin labels from config defaults and tracked attrs.

    Parameters
    ----------
    attrs : dict or None
        The DataFrame's ``.attrs`` dict containing potential flatbread
        labels. If None, only default labels are returned.

    Returns
    -------
    set[str]
        All margin labels (default and custom).
    """
    tracked = (attrs or {}).get('flatbread', {}).get('labels', {})
    margin_labels = set()
    for labels in tracked.values():
        margin_labels.update(labels)
    return margin_labels


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
def track_labels(data: PandasObj, transform: str, labels: list[str]) -> None:
    """
    Store margin labels in ``data.attrs`` for a given transform.

    Transforms call this to record which labels they produced, so that
    subsequent chained operations can exclude them via
    ``resolve_ignored_keys``.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        Object whose ``.attrs`` will be mutated in place.
    transform : str
        Transform key (e.g. ``'totals'``, ``'aggregation'``).
    labels : list[str]
        Labels to track.
    """
    existing = get_nested_key(data.attrs, ['flatbread', 'labels', transform]) or set()
    set_nested_key(
        data.attrs,
        keys = ['flatbread', 'labels', transform],
        value = existing.union(labels),
    )