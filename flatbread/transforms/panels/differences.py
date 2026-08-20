from collections import defaultdict
from functools import singledispatch
from itertools import pairwise
from typing import Any, Literal

import pandas as pd

from flatbread import DEFAULTS
from flatbread.types import Axis, Level
from flatbread.transforms.panels.interleave import interleave
import flatbread.transforms.chaining as chaining
import flatbread.transforms.panels.state as state
import flatbread.tooling as tooling
import flatbread.axes as axes


type DiffMethods = Literal['diff', 'pct_change']


# region as diff
@tooling.inject_defaults(DEFAULTS['transforms']['differences'])
@chaining.track_margin_labels('differences')
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


# region panel
def _build_diff_panel(
    data: pd.DataFrame,
    axis: Axis,
    *,
    label_n: str,
    label: str,
    config_default: str,
    panel_type: str,
    method: DiffMethods,
    ignore_keys: str | list[str] | None,
    periods: int,
    interleaf: bool,
) -> pd.DataFrame:
    axis_resolved = axes.resolve_axis(axis)

    # resolve panel label
    if label == config_default:
        label = resolve_panel_label(label, axis_resolved)

    # check state and resolve data
    source = state.check_panel_state(data, label)

    # compute differences
    diffs = as_differences(
        source,
        axis = axis,
        periods = periods,
        method = method,
        label_diff = label,
        ignore_keys = ignore_keys,
    )

    # build paneled output
    saved_attrs = data.attrs
    panels = chaining.get_nested_key(data.attrs, ['flatbread', 'panels'])
    if panels is None:
        result = pd.concat({label_n: data, label: diffs}, axis=1)
    else:
        result = pd.concat([data, pd.concat({label: diffs}, axis=1)], axis=1)
    result.attrs = saved_attrs

    # register panel
    state.register_panel(result, label, panel_type, axis_resolved)

    # interleave
    if interleaf:
        result = interleave(result)

    return result


# region add diff
@tooling.inject_defaults(DEFAULTS['transforms']['differences'])
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
    result = pd.concat([data.pipe(relabel, label_n), diffs], axis=1)
    state.register_panel(result, label_diff, 'differences', axis=0)
    return result


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
    Add differences alongside original DataFrame data in a paneled layout.

    Computes differences from the original data (ignoring any prior
    flatbread panels) and appends them as a new panel. On the first
    panel transform the data is also wrapped under a ``label_n`` panel;
    subsequent transforms append without re-wrapping.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    axis : Axis
        Axis along which to compute differences:
        - 0 or ``'index'``: row-wise differences
        - 1 or ``'columns'``: column-wise differences
    periods : int
        Number of periods to shift for computing the difference.
    method : {'diff', 'pct_change'}
        Differencing method to apply.
    label_n : str
        Label for the original data panel.
    label_diff : str
        Label for the difference panel. When equal to the configured
        default, an axis suffix is appended automatically (e.g.
        ``'diff_row'``). A custom value is used as-is.
    ignore_keys : str or list[str] or None
        Additional keys to exclude from diff computation. Keys from
        prior flatbread operations are excluded automatically.
    interleaf : bool
        If True, interleave diff columns with data columns and mark
        the DataFrame as interleaved.

    Returns
    -------
    pd.DataFrame
        DataFrame with difference panel appended.

    Raises
    ------
    ValueError
        If a panel with the resolved label already exists or if the
        DataFrame has already been interleaved.
    """
    return _build_diff_panel(
        data, axis,
        label_n = label_n,
        label = label_diff,
        config_default = DEFAULTS['transforms']['differences']['label_diff'],
        panel_type = 'differences',
        method = method,
        ignore_keys = ignore_keys,
        periods = periods,
        interleaf = interleaf,
    )


# region pct_change
@tooling.inject_defaults(DEFAULTS['transforms']['pct_change'])
@singledispatch
def add_pct_change(data, *args, **kwargs) -> Any:
    raise NotImplementedError('No implementation for this type')


@add_differences.register
def _(
    data: pd.Series,
    *,
    periods: int = 1,
    method: DiffMethods = 'pct_change',
    label_n: str = 'n',
    label_diff: str = 'pct_change',
    **kwargs,
) -> pd.DataFrame:
    """
    Add percentage change alongside original Series data.

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
    result = pd.concat([data.pipe(relabel, label_n), diffs], axis=1)
    state.register_panel(result, label_diff, 'pct_change', axis=0)
    return result


@add_pct_change.register
def _(
    data: pd.DataFrame,
    axis: Axis = 0,
    *,
    label_n: str = 'n',
    label_pct_change: str = 'pct_change',
    ignore_keys: str | list[str] | None = None,
    periods: int = 1,
    interleaf: bool = False,
    **kwargs,
) -> pd.DataFrame:
    return _build_diff_panel(
        data, axis,
        label_n = label_n,
        label = label_pct_change,
        config_default = DEFAULTS['transforms']['pct_change']['label_pct_change'],
        panel_type = 'pct_change',
        method = 'pct_change',
        ignore_keys = ignore_keys,
        periods = periods,
        interleaf = interleaf,
    )


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


def resolve_panel_label(label: str, axis: Axis) -> str:
    suffix = DEFAULTS['panels']['axis_suffixes'][str(axis)]
    return f"{label}_{suffix}"
