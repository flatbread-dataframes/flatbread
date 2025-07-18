from typing import Any, Callable, Hashable, Literal, TypeAlias
from pathlib import Path

import pandas as pd

import flatbread.percentages as pct
import flatbread.agg.aggregation as agg
import flatbread.agg.totals as totals
import flatbread.axes as axes
from flatbread.types import Axis, Level
from flatbread.render.display import PitaDisplayMixin


@pd.api.extensions.register_dataframe_accessor("pita")
class PitaFrame(PitaDisplayMixin):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    #region aggregation
    def add_agg(
        self,
        aggfunc: str|Callable,
        *args,
        axis: Axis = 0,
        label: str|None = None,
        ignore_keys: str|list[str]|None = None,
        _fill: str = '',
        **kwargs,
    ) -> pd.DataFrame:
        """
        Add aggregation to df.

        Parameters
        ----------
        aggfunc (str|Callable):
            Function to use for aggregating the data.
        axis (int | Literal["index", "columns", "both"]):
            Axis to aggregate. Default 0.
        label (str|None):
            Label for the aggregation row/column. Default None.
        ignore_keys (str|list[str]|None):
            Keys of rows to ignore when aggregating.
        *args:
            Positional arguments to pass to func.
        **kwargs:
            Keyword arguments to pass to func.

        Returns
        -------
        pd.DataFrame:
            Table with aggregated rows/columns added.
        """
        return agg.add_agg(
            self._obj,
            aggfunc,
            *args,
            axis = axis,
            label = label,
            ignore_keys = ignore_keys,
            _fill = _fill,
            **kwargs,
        )

    def add_subagg(
        self,
        aggfunc: str|Callable,
        axis: Axis = 0,
        level: int|str|list[int|str] = 0,
        label: str|None = None,
        include_level_name: bool = False,
        ignore_keys: str|list[str]|None = None,
        skip_single_rows: bool = True,
        _fill: str = '',
    ) -> pd.DataFrame:
        """
        Add aggregation to specified levels of the df.

        Parameters
        ----------
        aggfunc (str|Callable):
            Function to use for aggregating the data.
        axis (int | Literal["index", "columns", "both"]):
            Axis to aggregate. Default 0.
        levels (int|str|list[int|str]):
            Levels to aggregate. Default 0.
        label (str|None):
            Label for the aggregation row/column. Default None.
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
        pd.DataFrame:
            Table with aggregated rows/columns added.
        """
        return agg.add_agg(
            self._obj,
            aggfunc,
            axis = axis,
            level = level,
            label = label,
            include_level_name = include_level_name,
            ignore_keys = ignore_keys,
            skip_single_rows = skip_single_rows,
            _fill = _fill,
        )

    #region percentages
    def as_percentages(
        self,
        axis: Axis = 2,
        label_totals: str|None = None,
        ignore_keys: str|list[str]|None = None,
        ndigits: int|None = None,
        base: int = 1,
        apportioned_rounding: bool|None = None,
    ) -> pd.DataFrame:
        """
        Transform data to percentages based on specified axis.

        Parameters
        ----------
        data (pd.DataFrame):
            The input DataFrame.
        axis (int | Literal["index", "columns", "both"]):
            The axis along which percentages are calculated. Percentages are based on:
            - when axis is 2 then grand total
            - when axis is 1 then column totals
            - when axis is 0 then row totals
            Default is 2.
        label_totals (str|None):
            Label of the totals column/row. If no label is supplied then totals will be assumed to be either the last row, last column or last row/column field. Default is None.
        ignore_keys (str|list[str]|None):
            Keys of rows/columns to ignore when calculating percentages.
        ndigits (int):
            Number of decimal places to round the percentages. Default is -1 (no rounding).
        base (int):
            The whole quantity against which to calculate the fraction.

        Returns
        -------
        pd.DataFrame:
            DataFrame with data transformed to percentages.
        """
        return pct.as_percentages(
            self._obj,
            axis = axis,
            label_totals = label_totals,
            ignore_keys = ignore_keys,
            ndigits = ndigits,
            base = base,
            apportioned_rounding = apportioned_rounding,
        )

    def as_pct(self, *args, **kwargs):
        return self.as_percentages(*args, **kwargs)

    def add_percentages(
        self,
        axis: Axis = 2,
        label_n: str|None = None,
        label_pct: str|None = None,
        label_totals: str|None = None,
        ignore_keys: str|list[str]|None = None,
        ndigits: int|None = None,
        base: int = 1,
        apportioned_rounding: bool|None = None,
        interleaf: bool = False,
    ) -> pd.DataFrame:
        """
        Add percentage columns to a DataFrame based on specified axis.

        Parameters
        ----------
        data (pd.DataFrame):
            The input DataFrame.
        axis (int | Literal["index", "columns", "both"]):
            The axis along which percentages are calculated. Percentages are based on:
            - when axis is 2 then grand total
            - when axis is 1 then row totals
            - when axis is 0 then column totals
            Default is 2.
        label_n (str):
            Label for the original count columns. Default is 'n'.
        label_pct (str):
            Label for the percentage columns. Default is 'pct'.
        label_totals (str|None):
            Label of the totals column/row. If no label is supplied then totals will be assumed to be either the last row, last column or last row/column field. Default is None.
        ignore_keys (str|list[str]|None):
            Keys of rows/columns to ignore when calculating percentages.
        ndigits (int):
            Number of decimal places to round the percentages. Default is -1 (no rounding).
        base (int):
            The whole quantity against which to calculate the fraction.
        interleaf (bool):
            If `interleaf` is True then percentages columns will be placed next to count columns. If set to False the percentages columns will have their own separate block in the table. Default is False.

        Returns
        -------
        pd.DataFrame:
            DataFrame with additional columns for percentages.
        """
        return pct.add_percentages(
            self._obj,
            axis = axis,
            label_n = label_n,
            label_pct = label_pct,
            label_totals = label_totals,
            ignore_keys = ignore_keys,
            ndigits = ndigits,
            base = base,
            apportioned_rounding = apportioned_rounding,
            interleaf = interleaf,
        )

    def add_pct(self, *args, **kwargs):
        return self.add_percentages(*args, **kwargs)

    #region totals
    def add_totals(
        self,
        axis: Axis = 2,
        label: str|None = None,
        ignore_keys: str|list[str]|None = None,
        _fill: str = '',
    ) -> pd.DataFrame:
        """
        Add totals to df.

        Parameters
        ----------
        axis (int | Literal["index", "columns", "both"]):
            Axis to sum. If axis == 2 then add totals to both rows and columns. Default 2.
        label (str|None):
            Label for the totals row/column. Default 'Totals'.
        ignore_keys (str|list[str]|None):
            Keys of rows to ignore when aggregating. Default 'Subtotals'

        Returns
        -------
        pd.DataFrame:
            Table with total rows/columns added.
        """
        return totals.add_totals( # type: ignore
            self._obj,
            axis = axis,
            label = label,
            ignore_keys = ignore_keys,
            _fill = _fill,
        )

    def add_subtotals(
        self,
        axis: Axis = 2,
        level: int|str|list[int|str] = 0,
        label: str|None = None,
        include_level_name: bool = False,
        ignore_keys: str|list[str]|None = None,
        skip_single_rows: bool = True,
        _fill: str = '',
    ) -> pd.DataFrame:
        """
        Add subtotals to df.

        Parameters
        ----------
        axis (int | Literal["index", "columns", "both"]):
            Axis to sum. If axis == 2 then add totals to both rows and columns. Default 2.
        levels (int|str|list[int|str]):
            Levels to sum with func. Default 0.
        label (str|None):
            Label for the subtotals row/column. Default 'Subtotals'.
        include_level_name (bool):
            Whether to add level name to subtotal label.
        ignore_keys (str|list[str]|None):
            Keys of rows to ignore when aggregating. Default 'Totals'
        skip_single_rows (bool):
            Whether to skip single rows when aggregating. Default True.

        Returns
        -------
        pd.DataFrame:
            Table with total rows/columns added.
        """
        return totals.add_subtotals( # type: ignore
            self._obj,
            axis = axis,
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
    ) -> pd.DataFrame:
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
        pd.DataFrame
            DataFrame with totals/subtotals repositioned according to the specified parameters.
        """
        return axes.sort_totals( # type: ignore
            self._obj,
            axis = axis,
            level = level,
            labels = labels,
            totals_last = totals_last,
            sort_remaining = sort_remaining,
        )

    def drop_totals(
        self
    ):
        return totals.drop_totals(self._obj)

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
        Export DataFrame to Excel with automatic formatting based on flatbread configuration.

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
        axis: int = 0,
    ):
        """
        Add a level containing the specified value to a DataFrame axis.

        Parameters
        ----------
        data (pd.DataFrame):
            Input DataFrame.
        value (Any):
            Value to fill the new level with.
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
        return axes.add_level(
            self._obj,
            value = value,
            level = level,
            level_name = level_name,
            axis = axis,
        )
