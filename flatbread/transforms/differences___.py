from functools import singledispatch
from typing import Any, Literal

import pandas as pd

from flatbread import DEFAULTS
from flatbread.types import Axis, Level
import flatbread.chaining as chaining
import flatbread.tooling as tooling
import flatbread.axes as axes


# region chaining
def _resolve_ignored_keys(
    data: pd.DataFrame | pd.Series,
    axis: int,
    ignore_keys: str | list[str] | None,
):
    keys_to_ignore = []

    if isinstance(ignore_keys, str):
        keys_to_ignore.append(ignore_keys)
    elif isinstance(ignore_keys, list):
        keys_to_ignore.extend(ignore_keys)

    tracked = data.attrs.get('flatbread', {}).get('labels', {})
    keys_to_ignore.extend(tracked.get('differences', []))
    return keys_to_ignore


# region add diff
@singledispatch
def add_differences(
    data,
    periods: int = 1,
    axis: Axis = 0,
) -> Any:
    raise NotImplementedError('No implementation for this type')


add_diffs = add_differences


@add_differences.register
@tooling.inject_defaults(DEFAULTS['transforms']['differences'])
@chaining.tag_labels('differences')
def _(
    data: pd.DataFrame,
    method: str = 'diff',
    periods: int = 1,
    axis: Axis = 0,
    level: Level | None = None,
    label_n: str = 'n',
    label_diff: str = 'diff',
    ignore_keys: str | list[str] | None = None,
    **kwargs,
) -> pd.DataFrame:
    return _diff_implementation(
        data        = data,
        method      = method,
        periods     = periods,
        axis        = axis,
        level       = level,
        label_n     = label_n,
        label_diff  = label_diff,
        ignore_keys = ignore_keys,
    )


# region implement
def _diff_implementation(
    data: pd.DataFrame,
    method: str = 'diff',
    periods: int = 1,
    axis: Axis = 0,
    level: Level | None = None,
    label_n: str = 'n',
    label_diff: str = 'diff',
    ignore_keys: str | list[str] | None = None,
) -> pd.DataFrame:

    axis_resolved = axes.resolve_axis(axis)
    if axis_resolved == 0:
        rows = chaining.get_data_mask(data.index, ignore_keys)
        data = data.loc[rows]
    else:
        cols = chaining.get_data_mask(data.columns, ignore_keys)
        data = data.loc[:, cols]

    if level is None:
        if method != 'both':
            return _apply_diff_and_combine(
                data       = data,
                method     = method,
                periods    = periods,
                axis       = axis_resolved,
                label_n    = label_n,
                label_diff = label_diff,
            )
        else:
            return _diff_both_axis1(
                data,
                periods = periods,
                label_n    = label_n,
                label_diff = label_diff,
            )
    else:
        level_resolved = axes.resolve_level(data.index, level)
        if axis == 0:
            return (
                data.groupby(
                    level = level_resolved,
                    group_keys = False,
                )
                .apply(
                    _apply_diff_and_combine,
                    method    = method,
                    periods    = periods,
                    label_n    = label_n,
                    label_diff = label_diff,
                )
            )
        else:
            if method != 'both':
                return _diff_axis1_with_level(
                    data       = data,
                    method     = method,
                    periods    = periods,
                    level      = level,
                    label_n    = label_n,
                    label_diff = label_diff,
                )
            else:
                return _diff_both_axis1_with_level(
                    data       = data,
                    periods    = periods,
                    level      = level,
                    label_n    = label_n,
                    label_diff = label_diff,
                )


def _diff_axis1_with_level(
    data: pd.DataFrame,
    method: str,
    periods: int,
    level: Level,
    label_n: str,
    label_diff: str,
) -> pd.DataFrame:
    level_resolved = axes.resolve_level(data.index, level)
    diffs = (
        data.T.groupby(
            level = level_resolved,
            group_keys = False,
        )
        .apply(
            _apply_diff,
            method = method,
            periods = periods,
        ).T
    )
    combined = (
        data
        .pipe(
            _combine_parts,
            diffs      = diffs,
            label_n    = label_n,
            label_diff = label_diff,
        )
    )
    new_order = list(range(1, combined.columns.nlevels))
    new_order.insert(level_resolved + 1, 0)
    return (
        combined
        .reorder_levels(new_order, axis=1)
        .pipe(tooling.reindex_by_levels, data, nlevels=level_resolved+1)
    )

def _diff_both_axis1(
    data: pd.DataFrame,
    periods: int,
    label_n: str,
    label_diff: str,
) -> pd.DataFrame:

    diffs_abs = (
        data
        .apply('diff', axis=1, periods=periods)
        .iloc[:, -1]
        .rename('abs')
    )
    diffs_pct = (
        data
        .T.apply('pct_change', periods=periods).T
        .iloc[:, -1]
        .rename('pct')
    )

    diffs_both = pd.concat([diffs_abs, diffs_pct], axis=1)
    return pd.concat({label_n: data, label_diff: diffs_both}, axis=1)

def _diff_both_axis1_with_level(
    data: pd.DataFrame,
    periods: int,
    level: Level,
    label_n: str,
    label_diff: str,
) -> pd.DataFrame:
    level_resolved = axes.resolve_level(data.index, level)
    def make_diffs(data, method, periods, level, label):
        diffs = (
            data.T.groupby(
                level = level,
                group_keys = False,
            )
            .apply(
                _apply_diff,
                method = method,
                periods = periods,
            ).T
            .pita.add_level(label, axis=1, level=-1)
        )
        return diffs
    diffs_abs = make_diffs(data, 'diff', periods, level_resolved, 'abs')
    diffs_pct = make_diffs(data, 'pct_change', periods, level_resolved, 'pct')
    diffs = pd.concat([diffs_abs, diffs_pct], axis=1)
    combined = (
        data
        .pita.add_level('', axis=1, level=-1)
        .pipe(
            _combine_parts,
            diffs      = diffs,
            label_n    = label_n,
            label_diff = label_diff,
        )
    )
    new_order = list(range(1, combined.columns.nlevels))
    new_order.insert(level_resolved + 1, 0)
    return (
        combined
        .reorder_levels(new_order, axis=1)
        .pipe(tooling.reindex_by_levels, data, nlevels=level_resolved+1)
    )


def _apply_diff_and_combine(
    data: pd.DataFrame,
    method: str = 'diff',
    periods: int = 1,
    axis: Axis = 0,
    label_n: str = 'n',
    label_diff: str = 'diff',
) -> pd.DataFrame:
    diffs = _apply_diff(
        data,
        method  = method,
        periods = periods,
        axis    = axis,
    )
    return _combine_parts(data, diffs, label_n, label_diff)


def _apply_diff(
    data: pd.DataFrame,
    method: str = 'diff',
    periods: int = 1,
    axis: Axis = 0,
) -> pd.DataFrame:
    axis_resolved = axes.resolve_axis(axis)
    return (
        data
        .apply(
            method,                   # type: ignore
            axis    = axis_resolved,  # type: ignore
            periods = periods,
        )
        .dropna(how='all', axis=axis_resolved)
    )


def _combine_parts(
    data: pd.DataFrame,
    diffs: pd.DataFrame,
    label_n: str = 'n',
    label_diff: str = 'diff',
) -> pd.DataFrame:
    combined = pd.concat({label_n: data, label_diff: diffs}, axis=1)
    return combined
