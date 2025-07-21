from typing import Literal

import pandas as pd

from flatbread import DEFAULTS
from flatbread.types import Axis, Level
import flatbread.transforms.aggregation as agg
import flatbread.tooling as tooling
import flatbread.axes as axes
import flatbread.chaining as chaining


# region chaining
def _resolve_ignored_keys(
    data: pd.DataFrame|pd.Series,
    axis: int,
    ignore_keys: str|list[str]|None,
):
    keys_to_ignore = []

    if isinstance(ignore_keys, str):
        keys_to_ignore.append(ignore_keys)
    elif isinstance(ignore_keys, list):
        keys_to_ignore.extend(ignore_keys)

    tracked = data.attrs.get('flatbread', {}).get('labels', {})
    keys_to_ignore.extend(tracked.get('totals', []))
    if axis == 1:
        keys_to_ignore.extend(tracked.get('percentages', []))
        keys_to_ignore.extend(tracked.get('differences', []))

    return keys_to_ignore


# region totals
@tooling.inject_defaults(DEFAULTS['transforms']['totals'])
@chaining.tag_labels('totals')
def add_totals(
    data: pd.DataFrame|pd.Series,
    axis: Axis|Literal[2, 'both'] = 2,
    label: str|None = 'Totals',
    ignore_keys: str|list[str]|None = None,
    _fill: str|None = '',
    **kwargs,
) -> pd.DataFrame|pd.Series:
    axis = axes.resolve_axis(axis)
    keys_to_ignore = _resolve_ignored_keys(data, axis, ignore_keys)

    if axis < 2:
        output = agg.add_agg(
            data,
            'sum',
            axis = axis,
            label = label,
            ignore_keys = keys_to_ignore,
            _fill = _fill
        )
    else:
        output = (
            data
            .pipe(
                add_totals,
                axis = 0,
                label = label,
                ignore_keys = keys_to_ignore,
                _fill = _fill,
            )
            .pipe(
                add_totals,
                axis = 1,
                label = label,
                ignore_keys = keys_to_ignore,
                _fill = _fill,
            )
        )
    return output


# region subtotals
@tooling.inject_defaults(DEFAULTS['transforms']['subtotals'])
@chaining.tag_labels('totals')
def _add_subtotals(
    data: pd.DataFrame|pd.Series,
    axis: Axis = 0,
    level: Level = 0,
    label: str|None = 'Subtotals',
    include_level_name: bool = False,
    ignore_keys: str|list[str]|None = None,
    skip_single_rows: bool = True,
    _fill: str = '',
    **kwargs,
) -> pd.DataFrame|pd.Series:
    """Single-level subtotals implementation with tagging."""
    axis = axes.resolve_axis(axis)
    keys_to_ignore = _resolve_ignored_keys(data, axis, ignore_keys)

    if axis < 2:
        return agg.add_subagg(
            data,
            'sum',
            axis=axis,
            level=level,  # Single level only
            label=label,
            include_level_name=include_level_name,
            ignore_keys=keys_to_ignore,
            skip_single_rows=skip_single_rows,
            _fill=_fill,
        )
    else:
        output = (
            data
            .pipe(
                _add_subtotals,
                axis=0,
                level=level,
                label=label,
                include_level_name=include_level_name,
                ignore_keys=keys_to_ignore,
                skip_single_rows=skip_single_rows,
                _fill=_fill,
            )
            .pipe(
                _add_subtotals,
                axis=1,
                level=level,
                label=label,
                include_level_name=include_level_name,
                ignore_keys=keys_to_ignore,
                skip_single_rows=skip_single_rows,
                _fill=_fill,
            )
        )
        return output


def add_subtotals(
    data: pd.DataFrame|pd.Series,
    axis: Axis = 0,
    level: Level|list[Level] = 0,
    label: str|None = None,
    include_level_name: bool = False,
    ignore_keys: str|list[str]|None = None,
    skip_single_rows: bool = True,
    _fill: str = '',
    **kwargs,
) -> pd.DataFrame|pd.Series:
    """
    Add subtotal rows/columns at specified index levels using sum aggregation.

    This function creates subtotal rows or columns by grouping data at the specified
    index level(s) and applying sum aggregation. For MultiIndex data, subtotals are
    inserted at appropriate positions within the hierarchical structure. The function
    automatically excludes previously added flatbread labels (totals, percentages, etc.)
    from calculations to prevent double-counting.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        Input data with MultiIndex on the specified axis. Single-level indexes
        will raise an error.
    axis : Axis, default 0
        Axis along which to add subtotals:
        - 0 or 'index': subtotal rows (aggregate across columns)
        - 1 or 'columns': subtotal columns (aggregate across rows)
        - 2 or 'both': add subtotals to both axes
    level : Level | list[Level], default 0
        Index level(s) at which to create subtotals. Can be integer positions,
        level names, or a list of either. When multiple levels are specified,
        subtotals are added for each level sequentially.
    label : str | None, default None
        Base label for subtotal rows/columns. If None, uses configured default
        from flatbread settings. When include_level_name=True, the level value
        is appended to this label.
    include_level_name : bool, default False
        Whether to append the level name/value to the subtotal label. Results
        in labels like "Subtotals Region_A" instead of just "Subtotals".
    ignore_keys : str | list[str] | None, default None
        Additional labels to exclude from subtotal calculations. These are
        combined with automatically detected flatbread labels (totals, percentages, etc.).
    skip_single_rows : bool, default True
        Whether to skip creating subtotals for groups that contain only one
        data row. When True, single-row groups do not get subtotal rows added.
    _fill : str, default ''
        Fill value for empty positions in MultiIndex subtotal keys. Used
        internally for proper index alignment.

    Returns
    -------
    pd.DataFrame | pd.Series
        Data with subtotal rows/columns inserted at the specified levels.
        The output maintains the same type as the input.

    Raises
    ------
    AssertionError
        If the specified axis does not have a MultiIndex.
    AssertionError
        If any specified level is greater than or equal to the number of
        index levels minus one.
    ValueError
        If a subtotal label would conflict with existing index values.

    Notes
    -----
    - Subtotals are inserted in hierarchical order within the MultiIndex structure
    - The function automatically detects and ignores labels from previous flatbread
        operations (totals, percentages, differences) to prevent calculation errors
    - When axis=2, subtotals are added to both row and column axes sequentially
    - Level specification follows pandas conventions: 0 is outermost, -1 is innermost

    Examples
    --------
    Add subtotals at level 0 of a MultiIndex DataFrame:

    >>> df.pita.add_subtotals(level=0)

    Add subtotals at multiple levels with custom label:

    >>> df.pita.add_subtotals(level=[0, 1], label="Sub", include_level_name=True)

    Add subtotals to both axes:

    >>> df.pita.add_subtotals(axis=2, level=0)
    """
    # Handle single level case
    if isinstance(level, (int, str)):
        return _add_subtotals(
            data,
            axis=axis,
            level=level,
            label=label,
            include_level_name=include_level_name,
            ignore_keys=ignore_keys,
            skip_single_rows=skip_single_rows,
            _fill=_fill,
        )

    # Handle multiple levels case
    result = data
    for single_level in level:
        result = _add_subtotals(
            result,
            axis=axis,
            level=single_level,
            label=label,
            include_level_name=include_level_name,
            ignore_keys=ignore_keys,
            skip_single_rows=skip_single_rows,
            _fill=_fill,
        )
    return result


# region drop
def drop_totals(
    data: pd.DataFrame|pd.Series,
    ignore_keys: str|list[str]|None = None,
) -> pd.DataFrame|pd.Series:
    if ignore_keys is None:
        ignore_keys = data.attrs['flatbread']['totals']['ignore_keys']
    mask = chaining.get_data_mask(data.index, ignore_keys)
    return data.loc[mask].copy()
