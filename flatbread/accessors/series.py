from typing import Any, Callable, Hashable, Literal, TypeAlias
from pathlib import Path

import pandas as pd

import flatbread.percentages as pct
import flatbread.agg.aggregation as agg
import flatbread.agg.totals as totals
import flatbread.axes as axes
from flatbread.types import Axis, Level
from flatbread.render.display import PitaDisplayMixin


@pd.api.extensions.register_series_accessor("pita")
class PitaSeries(PitaDisplayMixin):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    #region aggregation
    def add_agg(
        self,
        aggfunc: str|Callable,
        *args,
        label: str|None = None,
        ignore_keys: str|list[str]|None = None,
        _fill: str = '',
        **kwargs,
    ) -> pd.Series:
        """
        Add aggregate to a Series.

        Parameters
        ----------
        aggfunc (str|Callable):
            Function to use for aggregating the data.
        label (str|None):
            Label for the aggregated row. Default None.
        ignore_keys (str|list[str]|None):
            Keys of rows to ignore when aggregating.
        *args:
            Positional arguments to pass to func.
        **kwargs:
            Keyword arguments to pass to func.

        Returns
        -------
        pd.Series:
            Series with aggregated row added.
        """
        return agg.add_agg(
            self._obj,
            aggfunc,
            *args,
            label = label,
            ignore_keys = ignore_keys,
            _fill = _fill,
            **kwargs,
        )

    def add_subagg(
        self,
        aggfunc: str|Callable,
        level: Level|list[Level] = 0,
        label: str|None = None,
        include_level_name: bool = False,
        ignore_keys: str|list[str]|None = None,
        skip_single_rows: bool = True,
        _fill: str = '',
    ) -> pd.Series:
        """
        Add aggregates of specified levels to a Series.

        Parameters
        ----------
        aggfunc (str|Callable):
            Function to use for aggregating the data.
        level (int|str|list[int|str]):
            Level(s) to aggregate with func. Default 0.
        label (str|None):
            Label for the aggregated rows. Default None.
        include_level_name (bool):
            Whether to add level name to subtotal label.
        ignore_keys (str|list[str]|None):
            Keys of rows to ignore when aggregating. Default 'Totals'
        skip_single_rows (bool):
            Whether to skip single rows when aggregating. Default True.
        *args:
            Positional arguments to pass to func.
        **kwargs:
            Keyword arguments to pass to func.

        Returns
        -------
        pd.Series:
            Table with aggregated rows added.
        """
        return agg.add_agg(
            self._obj,
            aggfunc,
            level = level,
            label = label,
            include_level_name = include_level_name,
            ignore_keys = ignore_keys,
            skip_single_rows = skip_single_rows,
            _fill = _fill,
        )

    #region value counts
    def value_counts(
        self,
        fillna: str = '<NA>',
        label_n: str = 'count',
        add_pct: bool = False,
        label_pct: str = 'pct',
        ndigits: int = -1,
        base: int = 1,
    )-> pd.Series|pd.DataFrame:
        """
        Similar to pandas `value_counts` except *null* values are by default also counted and a total is added. Optionally, percentages may also be added to the output.

        Parameters
        ----------
        fillna (str):
            What value to give *null* values. Set to None to not count null values. Default is '<NA>'.
        label_n (str):
            Name for the count column. Default is 'count'.
        add_pct (bool):
            Whether to add a percentage column. Default is False.
        label_pct (str):
            Name for the percentage column. Default is 'pct'.
        ndigits (int):
            Number of decimal places to round the percentages. Default is -1 (no rounding).
        base (int):
            The whole quantity against which to calculate the fraction.

        Returns
        -------
        pd.Series:
            Series reporting the count of each value in the original series.
        """
        s = self._obj if fillna is None else self._obj.fillna(fillna)
        result = s.value_counts().rename(label_n).pipe(totals.add_totals)
        if add_pct:
            return result.pipe(
                pct.add_percentages,
                label_n = label_n,
                label_pct = label_pct,
                ndigits = ndigits,
                base = base,
            )
        return result

    #region percentages
    def as_percentages(
        self,
        label_pct: str|None = None,
        label_totals: str|None = None,
        ndigits: int|None = None,
        base: int = 1,
        apportioned_rounding: bool|None = None,
    ) -> pd.Series:
        """
        Transform data into percentages.

        Parameters
        ----------
        data (pd.Series):
            The input Series.
        label_pct (str):
            Label for the percentage column. Default is 'pct'.
        label_totals (str|None):
            Label of the totals row. If no label is supplied then totals will be assumed to be the last row. Default is None.
        ndigits (int):
            Number of decimal places to round the percentages. Default is -1 (no rounding).
        base (int):
            The whole quantity against which to calculate the fraction.

        Returns
        -------
        pd.Series:
            Series transformed into percentages.
        """
        return pct.as_percentages(
            self._obj,
            label_pct = label_pct,
            label_totals = label_totals,
            ndigits = ndigits,
            base = base,
            apportioned_rounding = apportioned_rounding,
        )

    def as_pct(self, *args, **kwargs):
        return self.as_percentages(*args, **kwargs)

    def add_percentages(
        self,
        label_n: str|None = None,
        label_pct: str|None = None,
        label_totals: str|None = None,
        ndigits: int|None = None,
        base: int = 1,
        apportioned_rounding: bool|None = None,
    ) -> pd.DataFrame:
        """
        Add percentage column to a Series.

        Parameters
        ----------
        data (pd.Series):
            The input Series.
        label_n (str):
            Label for the original count column. Default is 'n'.
        label_pct (str):
            Label for the percentage column. Default is 'pct'.
        label_totals (str|None):
            Label of the totals row. If no label is supplied then totals will be assumed to be the last row. Default is None.
        ndigits (int):
            Number of decimal places to round the percentages. Default is -1 (no rounding).
        base (int):
            The whole quantity against which to calculate the fraction.

        Returns
        -------
        pd.DataFrame:
            DataFrame with the original Series and an additional column for the percentages.
        """
        return pct.add_percentages(
            self._obj,
            label_n = label_n,
            label_pct = label_pct,
            label_totals = label_totals,
            ndigits = ndigits,
            base = base,
            apportioned_rounding = apportioned_rounding,
        )

    def add_pct(self, *args, **kwargs):
        return self.add_percentages(*args, **kwargs)

    #region totals
    def add_totals(
        self,
        label: str|None = None,
        ignore_keys: str|list[str]|None = None,
        _fill: str = '',
    ) -> pd.Series:
        """
        Add totals to a Series.

        Parameters
        ----------
        label (str|None):
            Label for the totals row. Default 'Totals'.
        ignore_keys (str|list[str]|None):
            Keys of rows to ignore when aggregating. Default 'Subtotals'

        Returns
        -------
        pd.Series:
            Series with totals row added.
        """
        return totals.add_totals( # type: ignore
            self._obj,
            label = label,
            ignore_keys = ignore_keys,
            _fill = _fill,
        )

    def add_subtotals(
        self,
        level: Level|list[Level] = 0,
        label: str|None = None,
        include_level_name: bool = False,
        ignore_keys: str|list[str]|None = None,
        skip_single_rows: bool = True,
        _fill: str = '',
    ) -> pd.Series:
        """
        Add subtotals to a Series.

        Parameters
        ----------
        level (int|str|list[int|str]):
            Level(s) to add subtotals to. Default 0.
        label (str|None):
            Label for the subtotals rows. Default 'Subtotals'.
        include_level_name (bool):
            Whether to add level name to subtotal label.
        ignore_keys (str|list[str]|None):
            Keys of rows to ignore when aggregating. Default 'Totals'
        skip_single_rows (bool):
            Whether to skip single rows when aggregating. Default True.

        Returns
        -------
        pd.Series:
            Series with subtotal rows added.
        """
        return totals.add_subtotals( # type: ignore
            self._obj,
            level = level,
            label = label,
            include_level_name = include_level_name,
            ignore_keys = ignore_keys,
            skip_single_rows = skip_single_rows,
            _fill = _fill,
        )

    def sort_totals(
        self,
        axis: Axis = 0,
        level: Level|list[Level]|None = None,
        labels: list[str]|None = None,
        totals_last: bool = True,
        sort_remaining: bool = True,
    ) -> pd.Series:
        """
        Sort index/columns to position totals and subtotals at start or end within groups.

        Convenience function that sorts common aggregate labels (totals, subtotals) to
        their appropriate positions, while leaving other items in their existing order.
        Uses default labels from flatbread configuration unless custom labels are provided.

        Parameters
        ----------
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
        pd.Series
            Series with totals/subtotals repositioned according to the specified parameters.
        """
        return axes.sort_totals( # type: ignore
            self._obj,
            axis = axis,
            level = level,
            labels = labels,
            totals_last = totals_last,
            sort_remaining = sort_remaining,
        )

    # region io
    def export_excel(
        self,
        filepath: str | Path,
        title: str | None = None,
        number_formats: dict | None = None,
        border_specs: dict | None = None,
        **kwargs
    ) -> None:
        """
        Export Series to Excel with automatic formatting based on flatbread configuration.

        Parameters
        ----------
        filepath : str | Path
            Path to save the Excel file
        title : str, optional
            Title for the worksheet
        number_formats : dict, optional
            Custom number formats (overrides auto-detected ones)
        border_specs : dict, optional
            Custom border specifications (merged with margin borders)
        **kwargs
            Additional arguments passed to pandasxl WorksheetManager
        """
        import flatbread.io.excel as excel
        return excel.export_excel(
            self._obj,
            filepath,
            title=title,
            number_formats=number_formats,
            border_specs=border_specs,
            **kwargs
        )

    # region tooling
    def add_level(
        self,
        value: Any,
        level: int = 0,
        level_name: Any = None,
        axis: Axis = 0,
    ):
        """
        Add a level containing the specified value to a Series index.

        Parameters
        ----------
        data (pd.Series):
            Input Series.
        value (Any):
            Value to fill the new level with.
        level (int, optional):
            Position to insert the new level. Defaults to 0 (start).
        level_name (Any, optional):
            Name for the new level. Defaults to None.
        axis (int | Literal["index", "columns", "both"]):
            Added for symmetry with DataFrame method.

        Returns
        -------
        pd.Series:
            Series with the new level added to the specified axis.
        """
        return axes.add_level(
            self._obj,
            value = value,
            level = level,
            level_name = level_name,
            axis = axis,
        )
