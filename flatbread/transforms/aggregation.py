from functools import singledispatch
from typing import Any, Callable
import warnings

import pandas as pd

import flatbread.chaining as chaining
import flatbread.tooling as tooling
from flatbread.types import Axis, Level


# Ignore lexsort warning as agg is meant to keep the original order of the df in tact
warnings.filterwarnings(
    'ignore',
    category=pd.errors.PerformanceWarning,
    message='indexing past lexsort depth may impact performance.',
    module='flatbread.transforms.aggregation',
)


# region helpers
def get_label(label, aggfunc):
    if label is not None:
        return label
    if isinstance(aggfunc, str):
        return aggfunc
    if hasattr(aggfunc, '__name__') and aggfunc.__name__ != '<lambda>':
        return aggfunc.__name__
    return 'aggregation'


def get_levels(levels, names):
    find_level = lambda lvl: lvl if isinstance(lvl, int) else names.index(lvl)
    if isinstance(levels, (int, str)):
        return [find_level(levels)]
    return [find_level(level) for level in levels]


def create_agg_row(
    agged_data: pd.Series,
    label: str,
    original_index: pd.Index,
    _fill: str = '',
    group_levels: tuple|None = None
) -> pd.DataFrame:
    """Create a properly indexed row for aggregation results."""
    if isinstance(original_index, pd.MultiIndex):
        key = build_multiindex_key(label, original_index, _fill, group_levels)
        validate_index_key(original_index, key)
        return create_multiindex_row(agged_data, key, original_index)
    else:
        validate_index_key(original_index, label)
        return create_single_index_row(agged_data, label, original_index)


def build_multiindex_key(
    label: str,
    original_index: pd.MultiIndex,
    _fill: str,
    group_levels: tuple|None
) -> tuple:
    """Build the key tuple for MultiIndex aggregation row."""
    if group_levels is not None:
        # subagg case: preserve group levels + add subtotal
        padding = (_fill,) * (original_index.nlevels - len(group_levels) - 1)
        return group_levels + (label,) + padding
    else:
        # regular agg case: label + padding
        padding = (_fill,) * (original_index.nlevels - 1)
        return (label,) + padding if padding else label


def validate_index_key(
    original_index: pd.Index|pd.MultiIndex,
    key: str|tuple,
) -> None:
    """Validate that the key doesn't already exist."""
    if key in original_index:
        raise ValueError(f"Aggregation row with key {key} already exists")


def create_multiindex_row(
    agged_data: pd.Series,
    key: tuple,
    original_index: pd.MultiIndex
) -> pd.DataFrame:
    """Create aggregation row for MultiIndex."""
    idx = pd.MultiIndex.from_tuples([key], names=original_index.names)
    return pd.DataFrame([agged_data], index=idx)


def create_single_index_row(
    agged_data: pd.Series,
    label: str,
    original_index: pd.Index
) -> pd.DataFrame:
    """Create aggregation row for single Index."""
    idx = pd.Index([label], name=original_index.name)
    return pd.DataFrame([agged_data], index=idx)


# region aggregation
@tooling.handle_series_as_dataframe
@tooling.handle_axis_rotation
def add_agg(
    df: pd.DataFrame,
    aggfunc: str|Callable,
    *args,
    label: str|None = None,
    ignore_keys: str|list[str]|None = None,
    _fill: str|None = '',
    **kwargs,
) -> pd.DataFrame:
    data = df.copy()
    label = get_label(label, aggfunc)
    rows = chaining.get_data_mask(data.index, ignore_keys)

    agged = data.loc[rows].agg(aggfunc, *args, **kwargs)
    new_row = create_agg_row(
        agged,
        label = label,
        original_index = data.index,
        _fill = _fill,
    )
    return pd.concat([data, new_row], names=data.index.names)


# region subagg
@tooling.handle_series_as_dataframe
@tooling.handle_axis_rotation
def add_subagg(
    df: pd.DataFrame,
    aggfunc: str|Callable,
    *args,
    level: Level = 0,
    label: str|None = None,
    include_level_name: bool = False,
    ignore_keys: str|list[str]|None = None,
    skip_single_rows: bool = True,
    _fill = '',
    **kwargs,
):
    return _subagg_implementation(
        df.copy(),
        aggfunc,
        *args,
        level=level,
        label=label,
        include_level_name=include_level_name,
        ignore_keys=ignore_keys,
        skip_single_rows=skip_single_rows,
        _fill=_fill,
        **kwargs,
    )


def _subagg_implementation(
    data: pd.DataFrame,
    aggfunc: str|Callable,
    *args,
    level: Level = 0,
    label: str|None = None,
    include_level_name: bool = False,
    ignore_keys: str|list[str]|None = None,
    skip_single_rows: bool = True,
    _fill = '',
    **kwargs,
):
    names = data.index.names
    label = get_label(label, aggfunc)
    levels = get_levels(level, names)

    # checks
    msg = 'Flatbread cannot perform subaggregation if axis is not MultiIndex'
    assert isinstance(data.index, pd.MultiIndex), msg
    nlevels = data.index.nlevels
    for level in levels:
        assert level < nlevels - 1, f'Level must be smaller than {nlevels - 1}'

    def process_groups(groups):
        processed = []
        for levels, group in groups:
            levels = (levels,) if pd.api.types.is_scalar(levels) else levels
            level_value = levels[-1] if isinstance(levels, tuple) else levels

            subtotal_label = label
            if include_level_name:
                subtotal_label = f"{label} {level_value}"

            rows = chaining.get_data_mask(group.index, ignore_keys)
            if sum(rows) > (1 if skip_single_rows else 0):
                subagged = group.loc[rows].agg(aggfunc, *args, **kwargs)
                new_row = create_agg_row(
                    subagged,
                    subtotal_label,
                    original_index = data.index,
                    _fill = _fill,
                    group_levels = levels,
                )
                group = pd.concat([group, new_row])

            processed.append(group)
        return pd.concat(processed)

    output = data
    for level in sorted(levels, reverse=True):
        grouper = 0 if level == 0 else list(range(level + 1))
        output = output.groupby(level=grouper, sort=False).pipe(process_groups)
    return output
