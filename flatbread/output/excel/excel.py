from functools import singledispatch
from pathlib import Path
from typing import Any

import pandas as pd

from flatbread import DEFAULTS
from flatbread.render.constants import SMART_FORMATS, USER_PRESETS


def _get_auto_number_formats(df: pd.DataFrame) -> dict[str, str]:
    """Extract number formats from flatbread configuration."""
    formats = {}

    # Check smart formats (percentages, differences, etc.)
    for format_name, format_config in SMART_FORMATS.items():
        excel_format = format_config.get('excel_format')
        if not excel_format:
            continue

        labels = format_config.get('labels', [])
        for label in labels:
            # Check both column names and index values
            for col in df.columns:
                if _matches_label(col, label):
                    formats[col] = excel_format
            for idx in df.index:
                if _matches_label(idx, label):
                    formats[idx] = excel_format

    # Check user presets
    for preset_name, preset_config in USER_PRESETS.items():
        excel_format = preset_config.get('excel_format')
        if not excel_format:
            continue

        # Apply to columns based on dtype matching
        dtypes = preset_config.get('dtypes', [])
        for col in df.columns:
            col_dtype = str(df[col].dtype)
            # Simple dtype matching - could be more sophisticated
            if any(dtype_name in col_dtype for dtype_name in dtypes):
                formats[col] = excel_format

    return formats


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
        DEFAULTS['totals']['label'],
        DEFAULTS['subtotals']['label'],
        DEFAULTS['percentages']['label_pct']
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
