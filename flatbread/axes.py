from functools import singledispatch
from typing import Any

import pandas as pd

from flatbread import DEFAULTS
from flatbread.types import Axis, Level


# region resolvers
def resolve_level(
    index: pd.Index,
    level: Level,
) -> int:
    """Resolve level specification to list of integer levels."""
    if isinstance(level, str):
        try:
            resolved = index.names.index(level)
        except ValueError:
            raise ValueError(f"Level name '{level}' not found in index names")
    else:
        if level >= index.nlevels or level < -index.nlevels:
            raise IndexError(f"Level {level} out of range for index with {index.nlevels} levels")
        resolved = level if level >= 0 else index.nlevels + level

    return resolved


def resolve_axis(axis: Axis) -> int:
    """
    Resolve axis parameter to integer.

    Parameters
    ----------
    axis : int | str | None
        Axis specification:
        - 0 or 'index': index axis
        - 1 or 'columns': columns axis
        - 2 or 'both': both axes

    Returns
    -------
    int
        Resolved axis as integer
    """
    match axis:
        case 0 | 'index' | 'rows' | None:
            return 0
        case 1 | 'columns':
            return 1
        case 2 | 'both':
            return 2
        case _:
            raise ValueError(f"Invalid axis: {axis}")


# region sort aggs
def sort_aggregates(
    data: pd.DataFrame|pd.Series,
    axis: Axis = 0,
    level: Level|list[Level]|None = None,
    labels: list[str]|None = None,
    aggregates_last: bool = True,
    sort_remaining: bool = True,
) -> pd.DataFrame|pd.Series:
    """
    Sort index/columns to position aggregate labels at start or end within groups.

    Reorders index or column labels so that specified aggregate labels (like 'Totals',
    'Subtotals') appear either first or last within their respective groups, while
    preserving the existing order of non-aggregate items.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        Data to sort.
    axis : Axis, default 0
        Axis to sort along:
        - 0 or 'index': sort the index (rows)
        - 1 or 'columns': sort the columns
    level : Level | list[Level] | None, default None
        Index level(s) to sort. Can be level number(s), level name(s), or None for all levels.
        When sorting a specific level in a MultiIndex, only reorders within each group
        at that level.
    labels : list[str] | None, default None
        List of labels to treat as aggregates. If None, no special positioning is applied.
        Labels are matched as substrings for MultiIndex levels.
    aggregates_last : bool, default True
        Whether to place aggregate labels at the end (True) or beginning (False) of each group.
    sort_remaining : bool, default True
        Whether to sort non-target levels alphabetically. When False, preserves existing
        order of other levels.

    Returns
    -------
    pd.DataFrame | pd.Series
        Data with aggregate labels repositioned according to the specified parameters.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3]}, index=['Item1', 'Totals', 'Item2'])
    >>> sort_aggregates(df, labels=['Totals'])
                A
    Item1       1
    Item2       3
    Totals      2

    >>> # MultiIndex example - sort level 1 within each level 0 group
    >>> sort_aggregates(df, level=1, labels=['Subtotals'], aggregates_last=False)
    """
    if level is None:
        resolved_level = None
    elif isinstance(level, list):
        resolved_level = [resolve_level(data.index, lv) for lv in level]
    else:
        resolved_level = resolve_level(data.index, level)

    def create_sort_index(idx: pd.Index) -> pd.Index:
        label_score = len(idx) if aggregates_last else -1
        mapping = {
            key:label_score if key in labels else i
            for i,key
            in enumerate(idx.unique())
        }
        return idx.map(mapping)

    return data.sort_index(
        axis = axis, # type: ignore
        level = resolved_level,
        sort_remaining = sort_remaining,
        key = create_sort_index,
    )


def sort_totals(
    data: pd.DataFrame|pd.Series,
    axis: Axis = 0,
    level: Level|list[Level]|None = None,
    labels: list|None = None,
    totals_last: bool = True,
    sort_remaining: bool = True,
):
    """
    Sort index/columns to position totals and subtotals at start or end within groups.

    Convenience function that sorts common aggregate labels (totals, subtotals) to
    their appropriate positions, while leaving other items in their existing order.
    Uses default labels from flatbread configuration unless custom labels are provided.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        Data to sort.
    axis : Axis, default 0
        Axis to sort along:
        - 0 or 'index': sort the index (rows)
        - 1 or 'columns': sort the columns
    level : Level | list[Level] | None, default None
        Index level(s) to sort. Can be level number(s), level name(s), or None for all levels.
    labels : list[str] | None, default None
        Custom labels to treat as totals/subtotals. If None, uses default labels from
        flatbread configuration ('Totals', 'Subtotals').
    totals_last : bool, default True
        Whether to place totals/subtotals at the end (True) or beginning (False) of each group.
    sort_remaining : bool, default True
        Whether to sort non-target levels alphabetically.

    Returns
    -------
    pd.DataFrame | pd.Series
        Data with totals/subtotals repositioned according to the specified parameters.

    Notes
    -----
    This function uses the default aggregate labels from `DEFAULTS['totals']['label']`
    and `DEFAULTS['subtotals']['label']` unless custom labels are specified.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 3]}, index=['Item1', 'Totals', 'Item2'])
    >>> sort_totals(df)
                A
    Item1       1
    Item2       3
    Totals      2

    >>> # Place totals first instead of last
    >>> sort_totals(df, totals_last=False)
                A
    Totals      2
    Item1       1
    Item2       3
    """
    labels = [
        DEFAULTS['transforms']['subtotals']['label'],
        DEFAULTS['transforms']['totals']['label'],
    ] if labels is None else labels

    return sort_aggregates(
        data = data,
        axis = axis,
        level = level,
        labels = labels,
        aggregates_last = totals_last,
        sort_remaining = sort_remaining,
    )


# region add level
@singledispatch
def add_level(
    data,
    values: Any|list[Any],
    level: int = 0,
    level_name: Any = None,
    axis: Axis = 0,
) -> Any:
    raise NotImplementedError('No implementation for this type')


@add_level.register
def _(
    data: pd.DataFrame,
    value: Any|list[Any],
    level: int = 0,
    level_name: Any = None,
    axis: Axis = 0,
) -> pd.DataFrame:
    """
    Add a level containing the specified value to a DataFrame axis.

    Parameters
    ----------
    data (pd.DataFrame):
        Input DataFrame.
    value Any|list[Any]
        Either a single value to fill the entire level with, or a list of values with length matching the axis size. Values will be mapped in order.
    level (int, optional):
        Position to insert the new level. Defaults to 0 (start).
    level_name (Any, optional):
        Name for the new level. Defaults to None.
    axis (Axis):
        Axis to modify (0 for index, 1 for columns). Defaults to 0.

    Returns
    -------
    pd.DataFrame:
        DataFrame with the new level added to the specified axis.
    """
    data = data.copy()
    target = data.index if axis in [0, 'index'] else data.columns

    if isinstance(value, list):
        if len(value) != len(target):
            raise ValueError(
                f"Length of values list ({len(value)}) must match "
                f"length of {'index' if axis in [0, 'index'] else 'columns'} ({len(target)})"
            )

    if not isinstance(target, pd.MultiIndex):
        original_name = target.name
        target = pd.MultiIndex.from_arrays([target], names=[original_name])

    new_keys = [
        add_value_to_key(
            key,
            value[i] if isinstance(value, list) else value,
            level
        )
        for i, key in enumerate(target)
    ]
    new_names = add_value_to_key(target.names, level_name, level)
    new_index = pd.MultiIndex.from_tuples(new_keys, names=new_names)

    if axis in [0, 'index']:
        data.index = new_index
    else:
        data.columns = new_index
    return data


@add_level.register
def _(
    data: pd.Series,
    value: Any,
    level: int = 0,
    level_name: Any = None,
    axis: Axis = 0,
) -> pd.DataFrame:
    """
    Add a level containing the specified value to a Series index.

    Parameters
    ----------
    data (pd.Series):
        Input Series.
    value Any|list[Any]
        Either a single value to fill the entire level with, or a list of values with length matching the axis size. Values will be mapped in order.
    level (int, optional):
        Position to insert the new level. Defaults to 0 (start).
    level_name (Any, optional):
        Name for the new level. Defaults to None.
    axis (Axis):
        Added for symmetry with DataFrame method.

    Returns
    -------
    pd.Series:
        Series with the new level added to the specified axis.
    """
    data = data.copy()
    target = data.index

    if isinstance(value, list):
        if len(value) != len(target):
            raise ValueError(
                f"Length of values list ({len(value)}) must match "
                f"length of index ({len(target)})"
            )

    if not isinstance(target, pd.MultiIndex):
        original_name = target.name
        target = pd.MultiIndex.from_arrays([target], names=[original_name])

    new_keys = [
        add_value_to_key(
            key,
            value[i] if isinstance(value, list) else value,
            level
        )
        for i, key in enumerate(target)
    ]
    new_names = add_value_to_key(target.names, level_name, level)
    new_index = pd.MultiIndex.from_tuples(new_keys, names=new_names)
    data.index = new_index
    return data


def add_value_to_key(
    key: Any|tuple[Any, ...],
    value: Any,
    level: int = 0,
) -> tuple[Any, ...]:
    """
    Insert an value into a key (tuple or single value refering to a column position) at a specified level.

    Parameters
    ----------
    key (Any|tuple[Any, ...]):
        Original key.
    value (Any):
        Item to insert into the key.
    level (int, optional):
        Position to insert the value. Defaults to 0.

    Returns
    -------
    Tuple[Any, ...]:
        New key with the value inserted.
    """
    key = list(key) if isinstance(key, (tuple, list)) else [key]
    if level >= 0:
        key.insert(level, value)
    elif level == -1:
        key.append(value)
    else:
        key.insert(level + 1, value)
    return tuple(key)
