from typing import Literal

import pandas as pd

from flatbread import DEFAULTS
from flatbread.types import Axis, Level
import flatbread.transforms.aggregation as agg
import flatbread.tooling as tooling
import flatbread.axes as axes
import flatbread.chaining as chaining


# region totals
@tooling.inject_defaults(DEFAULTS['totals'])
@chaining.persist_ignored('totals', 'label')
def add_totals(
    data: pd.DataFrame|pd.Series,
    axis: Axis|Literal[2, 'both'] = 2,
    label: str|None = 'Totals',
    ignore_keys: str|list[str]|None = 'Subtotals',
    _fill: str|None = '',
) -> pd.DataFrame|pd.Series:
    axis = axes.resolve_axis(axis)
    if axis < 2:
        output = agg.add_agg(
            data,
            'sum',
            axis = axis,
            label = label,
            ignore_keys = ignore_keys,
            _fill = _fill
        )
    else:
        output = (
            data
            .pipe(
                add_totals,
                axis = 0,
                label = label,
                ignore_keys = ignore_keys,
                _fill = _fill,
            )
            .pipe(
                add_totals,
                axis = 1,
                label = label,
                ignore_keys = ignore_keys,
                _fill = _fill,
            )
        )
    return output


# region subtotals
@tooling.inject_defaults(DEFAULTS['subtotals'])
@chaining.persist_ignored('totals', 'label')
def add_subtotals(
    data: pd.DataFrame|pd.Series,
    axis: Axis = 0,
    level: Level = 0,
    label: str|None = 'Subtotals',
    include_level_name: bool = False,
    ignore_keys: str|list[str]|None = 'Totals',
    skip_single_rows: bool = True,
    _fill: str = '',
) -> pd.DataFrame|pd.Series:
    axis = axes.resolve_axis(axis)
    if axis < 2:
        output = agg.add_subagg(
            data,
            'sum',
            axis = axis,
            level = level,
            label = label,
            include_level_name = include_level_name,
            ignore_keys = ignore_keys,
            skip_single_rows = skip_single_rows,
            _fill = _fill,
        )
    else:
        output = (
            data
            .pipe(
                add_subtotals,
                axis = 0,
                level = level,
                label = label,
                include_level_name = include_level_name,
                ignore_keys = ignore_keys,
                skip_single_rows = skip_single_rows,
                _fill = _fill,
            )
            .pipe(
                add_subtotals,
                axis = 1,
                level = level,
                label = label,
                include_level_name = include_level_name,
                ignore_keys = ignore_keys,
                skip_single_rows = skip_single_rows,
                _fill = _fill,
            )
        )
    return output


# region drop
def drop_totals(
    data: pd.DataFrame|pd.Series,
    ignore_keys: str|list[str]|None = None,
) -> pd.DataFrame|pd.Series:
    if ignore_keys is None:
        ignore_keys = data.attrs['flatbread']['totals']['ignore_keys']
    mask = chaining.get_data_mask(data.index, ignore_keys)
    return data.loc[mask].copy()
