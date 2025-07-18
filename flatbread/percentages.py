from dataclasses import dataclass
from functools import singledispatch
from typing import Any
import warnings

import pandas as pd

from flatbread import DEFAULTS
from flatbread.types import Axis, Level
import flatbread.chaining as chaining
import flatbread.tooling as tooling
import flatbread.axes as axes


# region vals and totes
@dataclass
class ValuesAndTotals:
    values: pd.DataFrame
    totals: pd.Series | int | float
    axis: int

    @classmethod
    def from_data(
        cls,
        data: pd.DataFrame,
        axis: Axis,
        label_totals: str|None = None,
    ) -> 'ValuesAndTotals':
        """
        Create ValuesAndTotals by splitting input data based on axis and totals location.

        Parameters
        ----------
        data : pd.DataFrame
            Input data containing both values and totals
        axis : Axis
            Axis along which to split data and totals
        label_totals : str | None
            Label of the totals row/column. If None, assumes totals are in the last position

        Returns
        -------
        ValuesAndTotals
            Instance with separated values and totals
        """
        axis_resolved = axes.resolve_axis(axis)

        if label_totals is None:
            if axis_resolved == 0:  # column totals in last row
                values = data.iloc[:-1, :]
                totals = data.iloc[-1, :]
            elif axis_resolved == 1:  # row totals in last column
                values = data.iloc[:, :-1]
                totals = data.iloc[:, -1]
            else:  # grand total in bottom-right corner
                values = data.iloc[:-1, :-1]
                totals = data.iloc[-1, -1]
        else:
            # if label_totals is given:
            if axis_resolved == 0:  # column totals in specified row
                values = data.drop(label_totals, axis=0)
                totals = data.loc[label_totals, :]
            elif axis_resolved == 1:  # row totals in specified column
                values = data.drop(label_totals, axis=1)
                totals = data.loc[:, label_totals]
            else:  # grand total at specified row/column intersection
                values = data.drop(label_totals, axis=0).drop(label_totals, axis=1)
                totals = data.loc[label_totals, label_totals]

        return cls(
            values = values,
            totals = totals, # type: ignore
            axis = axis_resolved,
        )

    @property
    def should_use_apportioned_rounding(self) -> bool:
        """Check if values represent complete proportions of totals."""
        tolerance = 1e-10

        if self.axis == 0:  # column percentages
            column_sums = self.values.sum(axis=0)
            return (abs(column_sums - self.totals) < tolerance).all()
        elif self.axis == 1:  # row percentages
            row_sums = self.values.sum(axis=1)
            return (abs(row_sums - self.totals) < tolerance).all()
        else:  # axis == 2, grand total
            return abs(self.values.sum().sum() - self.totals) < tolerance


# region as pct
@singledispatch
def as_percentages(
    data,
    *args,
    label_pct: str = 'pct',
    ndigits: int = -1,
    base: int = 1,
    apportioned_rounding: bool = True,
    **kwargs,
) -> Any:
    """
    Transform data to percentages based on specified axis.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        The input data containing values and totals.
    axis : int | Literal["index", "columns", "both"], optional
        The axis along which percentages are calculated (DataFrame only):
        - 0 or "index": percentages based on column totals
        - 1 or "columns": percentages based on row totals
        - 2 or "both": percentages based on grand total
        Default is 2.
    label_totals : str | None, optional
        Label of the totals row/column. If None, assumes totals are in the last
        position. Default is None.
    ignore_keys : str | list[str] | None, optional
        Keys of rows/columns to ignore when calculating percentages (DataFrame only).
    ndigits : int, optional
        Number of decimal places to round percentages. Default is -1 (no rounding).
    base : int, optional
        The whole quantity against which to calculate the fraction. Default is 1.
    apportioned_rounding : bool | None, optional
        Whether to use apportioned rounding that ensures percentages sum to the base.
        If None, uses heuristic: apportioned rounding when data represents complete
        proportions of totals. Default is None.

    Returns
    -------
    pd.DataFrame | pd.Series
        Data transformed to percentages.

    Notes
    -----
    When `apportioned_rounding` is None, the function automatically determines whether
    to use apportioned rounding by checking if the data values sum to their corresponding totals within floating-point tolerance (1e-10).
    """
    raise NotImplementedError('No implementation for this type')


@as_percentages.register
@tooling.inject_defaults(DEFAULTS['percentages'])
def _(
    data: pd.Series,
    *,
    label_pct: str = 'pct',
    label_totals: str|None = None,
    ndigits: int = -1,
    base: int = 1,
    apportioned_rounding: bool|None = None,
    **kwargs,
) -> pd.Series:
    """Series implementation of as_percentages."""
    total = data.iloc[-1] if label_totals is None else data.loc[label_totals]

    pcts = (
        data
        .div(total)
        .mul(base)
    )

    if ndigits == -1:
        return pcts.rename(label_pct)

    if apportioned_rounding is None:
        # For Series: check if values sum to total (complete proportions)
        if label_totals is None:
            values = data.iloc[:-1]
        else:
            values = data.drop(label_totals)
        apportioned_rounding = abs(values.sum() - total) < 1e-10

    rounding = round_apportioned if apportioned_rounding else round
    return pcts.pipe(rounding, ndigits=ndigits).rename(label_pct)


@as_percentages.register
@tooling.inject_defaults(DEFAULTS['percentages'])
@chaining.persist_ignored('percentages', 'label_pct')
def _(
    df: pd.DataFrame,
    axis: Axis = 2,
    *,
    label_totals: str|None = None,
    ignore_keys: str|list[str]|None = 'pct',
    ndigits: int = -1,
    base: int = 1,
    apportioned_rounding: bool|None = None,
    **kwargs,
) -> pd.DataFrame:
    """DataFrame implementation of as_percentages."""
    cols = chaining.get_data_mask(df.columns, ignore_keys)
    data = df.loc[:, cols]
    vt = ValuesAndTotals.from_data(data, axis, label_totals)

    # reverse axis for consistency
    axis = 0 if axis == 1 else 1 if axis == 0 else None
    pcts = (
        data # type: ignore
        .div(vt.totals, axis=axis)
        .mul(base)
    )

    if ndigits < 0:
        return pcts

    if apportioned_rounding is None:
        apportioned_rounding = vt.should_use_apportioned_rounding

    rounding = round_apportioned if apportioned_rounding else round

    return pcts.pipe(rounding, ndigits=ndigits)


# region add pct
@singledispatch
def add_percentages(
    data,
    *args,
    label_pct: str = 'pct',
    ndigits: int = -1,
    base: int = 1,
    apportioned_rounding: bool = True,
    **kwargs,
) -> Any:
    raise NotImplementedError('No implementation for this type')


@add_percentages.register
@tooling.inject_defaults(DEFAULTS['percentages'])
def _(
    data: pd.Series,
    *,
    label_n: str = 'n',
    label_pct: str = 'pct',
    label_totals: str|None = None,
    ndigits: int = -1,
    base: int = 1,
    apportioned_rounding: bool = True,
    **kwargs,
) -> pd.DataFrame:
    pcts = data.pipe(
        as_percentages,
        label_pct = label_pct,
        label_totals = label_totals,
        ndigits = ndigits,
        base = base,
        apportioned_rounding = apportioned_rounding,
    )
    output = pd.concat([data, pcts], keys=[label_n, label_pct], axis=1)
    return output


@add_percentages.register
@tooling.inject_defaults(DEFAULTS['percentages'])
@chaining.persist_ignored('percentages', 'label_pct')
def _(
    df: pd.DataFrame,
    axis: int = 2,
    *,
    label_n: str = 'n',
    label_pct: str = 'pct',
    label_totals: str|None = None,
    ignore_keys: str|list[str]|None = 'pct',
    ndigits: int = -1,
    base: int = 1,
    apportioned_rounding: bool = True,
    interleaf: bool = False,
    **kwargs,
) -> pd.DataFrame:

    cols = chaining.get_data_mask(df.columns, ignore_keys)
    data = df.loc[:, cols]

    # totals = get_totals(data, axis, label_totals)
    # axis = axis if axis < 2 else None
    # pcts = (
    #     data
    #     .div(totals, axis=axis)
    #     .mul(100)
    #     .pipe(round_apportioned, ndigits=ndigits)
    # )
    pcts = data.pipe(
        as_percentages,
        axis = axis,
        label_totals = label_totals,
        ignore_keys = ignore_keys,
        ndigits = ndigits,
        base = base,
        apportioned_rounding = apportioned_rounding,
    )

    # check if there are already percentages in the table
    if cols.all():
        # if not then add them, original table gets `label_n`
        # percentages get `label_pct` as key
        keys = [label_n, label_pct]
        output = pd.concat([df, pcts], keys=keys, axis=1)
    else:
        # if percentages are present then transform them first
        # keys are already present in the original df
        # so we do not add new keys
        pcts = pcts.rename(columns={label_n: label_pct})
        output = pd.concat([df, pcts], axis=1)
    if interleaf:
        # return output.stack(0).unstack(-1)
        keys = [label_n, label_pct]
        return output.swaplevel(axis=1).sort_index(axis=1, level=0)
    return output


# region rounding
def round_apportioned(
    s: pd.Series,
    *,
    ndigits: int = -1
) -> pd.Series:
    """
    Round percentages in a way that they always add up to total.
    Taken from this SO answer:

    <https://stackoverflow.com/a/13483486/10403856>

    Parameters
    ----------
    s (pd.Series):
        A series of unrounded percentages adding up to total.
    ndigits (int):
        Number of digits to round percentages to. Default is -1 (no rounding).

    Returns
    -------
    pd.Series:
        Rounded percentages.
    """
    if ndigits < 0:
        return s
    cumsum = s.fillna(0).cumsum().round(ndigits)
    prev_baseline = cumsum.shift(1).fillna(0)
    rounded = cumsum - prev_baseline
    keep_na = rounded.mask(s.isna())
    return keep_na
