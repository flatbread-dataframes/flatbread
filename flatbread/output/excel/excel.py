from functools import singledispatch
from pathlib import Path
from typing import Any

import pandas as pd

from flatbread import DEFAULTS
from flatbread.output.formats import FormatResolver


def _get_auto_number_formats(df: pd.DataFrame) -> dict[str, str]:
    """Extract number formats from flatbread configuration.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to extract formats for.

    Returns
    -------
    dict[str, str]
        Mapping of column names to Excel number format strings.
    """
    resolver = FormatResolver(df)
    return {
        col: fmt
        for col in df.columns
        if (fmt := resolver.get_excel_format(col))
    }


def _get_auto_border_specs(df: pd.DataFrame) -> dict[str, list[str]]:
    """Extract border specifications from flatbread margin labels."""
    border_specs = {'rows': [], 'columns': []}

    # Get margin labels from attrs or defaults
    margin_labels = set()

    # Check DataFrame attrs for stored margin labels
    if hasattr(df, 'attrs') and df.attrs.get('flatbread'):
        fb_attrs = df.attrs['flatbread']
        if 'totals' in fb_attrs and 'ignore_keys' in fb_attrs['totals']:
            margin_labels.update(fb_attrs['totals']['ignore_keys'])
        if 'percentages' in fb_attrs and 'ignore_keys' in fb_attrs['percentages']:
            margin_labels.update(fb_attrs['percentages']['ignore_keys'])

    # Add default margin labels
    margin_labels.update([
        DEFAULTS['transforms']['totals']['label'],
        DEFAULTS['transforms']['subtotals']['label'],
        DEFAULTS['transforms']['percentages']['label_pct'],
    ])

    # Remove duplicates
    margin_labels = list(set(margin_labels))

    # Find matching rows and columns
    for label in margin_labels:
        # Check index for row borders
        for idx in df.index:
            if _matches_label(idx, label):
                border_specs['rows'].append(label)
                break

        # Check columns for column borders
        for col in df.columns:
            if _matches_label(col, label):
                border_specs['columns'].append(label)
                break

    return border_specs


def _matches_label(target: Any, label: str) -> bool:
    """Check if a target (index/column) matches a label pattern."""
    if isinstance(target, tuple):
        # For MultiIndex, check if label appears in any level
        return any(str(level) == label for level in target)
    else:
        # Direct string comparison
        return str(target) == label


@singledispatch
def export_excel(
    data,
    filepath: str | Path,
    title: str | None = None,
    number_formats: dict | None = None,
    border_specs: dict | None = None,
    **kwargs
) -> None:
    raise NotImplementedError('No implementation for this type')


@export_excel.register
def _(
    data: pd.DataFrame,
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
    df : pd.DataFrame
        DataFrame to export
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
    try:
        from flatbreadxl.worksheet import WorksheetManager
    except ImportError:
        raise ImportError(
            "flatbreadxl is required for Excel export. "
            "Install it with: pip install flatbreadxl"
        )

    # Strip timezone info (Excel does not support timezones)
    tz_cols = data.select_dtypes(include=["datetimetz"]).columns
    if len(tz_cols) > 0:
        data = data.copy()
        for col in tz_cols:
            data[col] = data[col].dt.tz_localize(None)

    # Extract flatbread settings and translate to pandasxl format
    auto_number_formats = _get_auto_number_formats(data)
    auto_border_specs = _get_auto_border_specs(data)

    # Merge user overrides
    final_number_formats = {**auto_number_formats, **(number_formats or {})}
    final_border_specs = {**auto_border_specs, **(border_specs or {})}

    # Set NA representation from flatbread defaults
    na_rep = DEFAULTS.get('na_rep', '-')

    # Create worksheet and export
    manager = WorksheetManager.from_filepath(filepath)
    manager.add_table(
        data,
        title=title,
        number_formats=final_number_formats,
        border_specs=final_border_specs,
        **kwargs
    )

    # Set NA representation on the worksheet if supported
    if hasattr(manager.worksheet, 'NA_REPRESENTATION'):
        manager.worksheet.NA_REPRESENTATION = na_rep

    manager.save()


@export_excel.register
def _(
    data: pd.Series,
    filepath: str | Path,
    title: str | None = None,
    number_formats: dict | None = None,
    border_specs: dict | None = None,
    **kwargs
) -> None:
    """
    Export Series to Excel with flatbread formatting.

    Parameters
    ----------
    s : pd.Series
        Series to export
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
    return export_excel(
        data.to_frame(),
        filepath,
        title = title,
        number_formats = number_formats,
        border_specs = border_specs,
        **kwargs
    )
