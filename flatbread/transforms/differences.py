from collections import defaultdict
from functools import singledispatch
from itertools import pairwise
from typing import Any, Literal

import pandas as pd

from flatbread import DEFAULTS
from flatbread.types import Axis, Level
import flatbread.chaining as chaining
import flatbread.tooling as tooling
import flatbread.axes as axes


type DiffMethods = Literal['diff', 'pct_change']


# region as diff
@tooling.inject_defaults(DEFAULTS['transforms']['differences'])
@chaining.tag_labels('differences')
@singledispatch
def as_differences(
    data,
    *args,
    periods: int = 1,
    method: DiffMethods = 'diff',
    label_n: str = 'n',
    label_diff: str = 'diff',
    **kwargs,
) -> Any:
    """
    Compute differences along an axis.

    Transforms data by applying the specified differencing method. For
    DataFrames with a MultiIndex, diffs are computed within each group
    defined by the parent levels.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Input data.
    periods : int
        Number of periods to shift for computing the difference.
    method : {'diff', 'pct_change'}
        Differencing method to apply.
    label_n : str
        Label for the original data. Used when combining with diffs.
    label_diff : str
        Label for the difference data.

    Returns
    -------
    pd.DataFrame or pd.Series
        Differenced data with consumed periods removed.

    See Also
    --------
    add_differences : Add differences alongside original data.
    """
    raise NotImplementedError('No implementation for this type')


as_diffs = as_differences


@as_differences.register
def _(
    data: pd.Series,
    *,
    periods: int = 1,
    method: DiffMethods = 'diff',
    label_diff: str = 'diff',
    **kwargs,
) -> pd.Series:
    """
    Compute differences for a Series.

    For a Series with a MultiIndex, diffs are computed within groups
    defined by the second-to-last level.

    Parameters
    ----------
    data : pd.Series
        Input series.
    periods : int
        Number of periods to shift.
    method : {'diff', 'pct_change'}
        Differencing method to apply.
    label_diff : str
        Label appended to the series name.

    Returns
    -------
    pd.Series
        Differenced series with label appended to name.
    """
    data = data.groupby(level=-2) if isinstance(data.index, pd.MultiIndex) else data
    results = data.apply(method, periods=periods)
    return results.pipe(relabel, label_diff)


@as_differences.register
@tooling.handle_axis_rotation
def _(
    data: pd.DataFrame,
    *,
    axis: Axis = 0,
    periods: int = 1,
    method: DiffMethods = 'diff',
    **kwargs,
) -> pd.DataFrame:
    """
    Compute differences for a DataFrame.

    For a DataFrame with a MultiIndex on the operated axis, diffs are
    computed within groups defined by the parent levels. When operating
    on columns (axis=1), the resulting columns are relabeled with
    pairwise labels indicating which columns were compared.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    periods : int
        Number of periods to shift.
    method : {'diff', 'pct_change'}
        Differencing method to apply.
    axis : {0, 1, 'index', 'columns'}
        Axis along which to compute diffs. 0 for rows, 1 for columns.
    label_diff : str
        Label for the difference data.

    Returns
    -------
    pd.DataFrame
        Differenced data with consumed periods removed. Fewer rows
        (axis=0) or columns (axis=1) than the input.
    """
    to_diff = data.groupby(level=-2) if isinstance(data.index, pd.MultiIndex) else data
    results = to_diff.apply(method, periods=periods).dropna(how='all')
    if axis == 1:
        new_labels = pairwise_labels(data.index, periods=periods)
        results.index = new_labels
    return results


# region add diff
@tooling.inject_defaults(DEFAULTS['transforms']['differences'])
@chaining.tag_labels('differences')
@singledispatch
def add_differences(
    data,
    *args,
    periods: int = 1,
    method: DiffMethods = 'diff',
    label_n: str = 'n',
    label_diff: str = 'diff',
    **kwargs,
) -> Any:
    """
    Add differences alongside original data.

    Computes differences and combines them with the original data,
    adding a level to distinguish between original values and diffs.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Input data.
    periods : int
        Number of periods to shift for computing the difference.
    method : {'diff', 'pct_change'}
        Differencing method to apply.
    label_n : str
        Label for the original data panel.
    label_diff : str
        Label for the difference panel.

    Returns
    -------
    pd.DataFrame
        Combined data with original values under ``label_n`` and
        differences under ``label_diff``.

    See Also
    --------
    as_differences : Compute differences without preserving originals.
    """
    raise NotImplementedError('No implementation for this type')


add_diffs = add_differences


@add_differences.register
def _(
    data: pd.Series,
    *,
    periods: int = 1,
    method: DiffMethods = 'diff',
    label_n: str = 'n',
    label_diff: str = 'diff',
    **kwargs,
) -> pd.DataFrame:
    """
    Add differences alongside original Series data.

    Parameters
    ----------
    data : pd.Series
        Input series.
    periods : int
        Number of periods to shift.
    method : {'diff', 'pct_change'}
        Differencing method to apply.
    label_n : str
        Label for the original data column.
    label_diff : str
        Label for the difference column.

    Returns
    -------
    pd.DataFrame
        Two-column DataFrame with original values and differences.
    """
    diffs = as_differences(data, periods=periods, method=method, label_diff=label_diff)
    return pd.concat([data.pipe(relabel, label_n), diffs], axis=1)


@add_differences.register
def _(
    data: pd.DataFrame,
    *,
    axis: Axis = 0,
    periods: int = 1,
    method: DiffMethods = 'diff',
    label_n: str = 'n',
    label_diff: str = 'diff',
    ignore_keys: str | list[str] | None = None,
    interleaf: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Add differences alongside original DataFrame data.

    Computes differences on the data portion (excluding columns or rows
    from prior flatbread operations) and combines with the original.
    Supports chaining: when prior differences exist, new diffs are
    appended without re-wrapping the existing structure.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    axis : {0, 1, 'index', 'columns'}
        Axis along which to compute diffs.
    periods : int
        Number of periods to shift.
    method : {'diff', 'pct_change'}
        Differencing method to apply.
    label_n : str
        Label for the original data panel.
    label_diff : str
        Label for the difference panel.
    interleaf : bool
        If True, interleave diff columns with their corresponding data
        columns instead of appending as a separate panel.
    ignore_keys : str or list[str] or None
        Additional keys to exclude from diff computation. Keys from
        prior flatbread operations are excluded automatically.

    Returns
    -------
    pd.DataFrame
        Combined data with differences added. Adds one level to the
        column index to distinguish data from diffs.
    """
    keys_to_ignore = chaining.resolve_ignored_keys(data, 'differences', ignore_keys)
    axis_resolved = axes.resolve_axis(axis)

    if axis_resolved == 1:
        mask = chaining.get_data_mask(data.columns, keys_to_ignore)
        source = data.loc[:, mask]
    else:
        mask = chaining.get_data_mask(data.index, keys_to_ignore)
        source = data.loc[mask]

    diffs = as_differences(
        data,
        axis=axis,
        periods=periods,
        method=method,
        label_diff=label_diff,
    )
    if mask.all():
        result = pd.concat({label_n: data, label_diff: diffs}, axis=1)
    else:
        result = pd.concat([data, pd.concat({label_diff: diffs}, axis=1)], axis=1)

    if interleaf:
        new_order = list(range(1, result.columns.nlevels))
        new_order.insert(-1, 0)
        result = (
            result
            .reorder_levels(new_order, axis=1)
            .pipe(
                tooling.reindex_by_levels,
                data,
                axis = 1,
                nlevels = result.columns.nlevels - 2,
            )
        )

    return result


# region labels
def relabel(s: pd.Series, name: str) -> pd.Series:
    """
    Append a label to a Series name.

    For a Series with a tuple name (from a MultiIndex column), the label
    is appended to the tuple. Otherwise, the name is wrapped in a tuple
    with the label.

    Parameters
    ----------
    s : pd.Series
        Input series.
    name : str
        Label to append.

    Returns
    -------
    pd.Series
        Series with modified name.
    """
    match s.name:
        case tuple() as t:
            new_name = (*t, name)
        case _:
            new_name = (s.name, name)
    return s.rename(new_name)


@singledispatch
def pairwise_labels(index, periods: int = 1) -> Any:
    """
    Create labels indicating which elements were compared in a diff.

    For each pair of elements separated by ``periods``, produces a label
    in the format "earlier-later". For a MultiIndex, pairing is applied
    at the leaf level within each group.

    Parameters
    ----------
    index : pd.Index or pd.MultiIndex
        The index to create pairwise labels from.
    periods : int
        Number of positions between compared elements. Positive values
        compare each element with the one ``periods`` steps earlier.
        Negative values compare with later elements.

    Returns
    -------
    pd.Index or pd.MultiIndex
        New index with pairwise comparison labels at the leaf level.
    """
    raise NotImplementedError('No implementation for this type')


@pairwise_labels.register
def _(index: pd.Index, periods: int = 1) -> pd.Index:
    return pd.Index([f"{a}-{b}" for a, b in _pair(index, periods)], name=index.name)


@pairwise_labels.register
def _(index: pd.MultiIndex, periods: int = 1) -> pd.MultiIndex:
    groups: dict[tuple, list] = defaultdict(list)
    for col in index:
        groups[col[:-1]].append(col[-1])

    tuples = []
    for key, leaves in groups.items():
        tuples.extend((*key, f"{a}-{b}") for a, b in _pair(leaves, periods))
    return pd.MultiIndex.from_tuples(tuples, names=index.names)


def _pair(index, periods: int):
    n = abs(periods)
    if periods > 0:
        return zip(index, index[n:])
    return zip(index[n:], index)