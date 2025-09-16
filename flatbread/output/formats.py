from typing import Any

import pandas as pd
from flatbread import DEFAULTS


class FormatResolver:
    """Central format resolution for all output types"""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.output_formats: dict = DEFAULTS.get('output_formats', {}) # type: ignore
        self.format_presets: dict = DEFAULTS.get('format_presets', {}) # type: ignore
        self.dtype_mappings: dict = DEFAULTS.get('dtype_mappings', {}) # type: ignore

    def resolve_formats(self) -> dict[Any, str]:
        """Resolve format types for all columns"""
        formats = {}

        for col in self.data.columns:
            format_type = self._resolve_format_type(col)
            if format_type:
                formats[col] = format_type

        return formats

    def get_html_format(self, column) -> dict[str, Any] | None:
        """Get HTML-specific format options for a column"""
        format_type = self._resolve_format_type(column)
        if not format_type:
            return None

        # Check output_formats first
        if format_type in self.output_formats:
            return self.output_formats[format_type].get('html_options')

        # Check format_presets
        if format_type in self.format_presets:
            return self.format_presets[format_type].get('html_options')

        return None

    def get_excel_format(self, column) -> str | None:
        """Get Excel-specific format string for a column"""
        format_type = self._resolve_format_type(column)
        if not format_type:
            return None

        # Check output_formats first
        if format_type in self.output_formats:
            return self.output_formats[format_type].get('excel_format')

        # Check format_presets
        if format_type in self.format_presets:
            return self.format_presets[format_type].get('excel_format')

        return None

    def _resolve_format_type(self, column) -> str | None:
        """Determine the format type for a column"""

        # 1. Check explicit format metadata (highest priority)
        explicit_format = (
            self.data.attrs
            .get('flatbread', {})
            .get('formats', {})
            .get(column)
        )
        if explicit_format:
            return explicit_format

        # 2. Check smart format detection (fallback)
        return self._detect_smart_format_type(column)

    def _detect_smart_format_type(self, column) -> str | None:
        """Detect format type based on column name patterns"""
        column_text = self._get_column_text(column)

        for format_type, format_config in self.output_formats.items():
            smart_labels = format_config.get('smart_labels', [])
            for label in smart_labels:
                if label in column_text:
                    return format_type
        return None

    def _get_column_text(self, column) -> str:
        """Extract searchable text from column (handle tuples)"""
        if isinstance(column, tuple):
            return ' '.join(str(part).lower() for part in column)
        else:
            return str(column).lower()

    def set_output_format(self, column, format_type: str) -> None:
        """Set explicit output format metadata for a column (utility method)"""
        if not hasattr(self.data, 'attrs'):
            self.data.attrs = {}

        (
            self.data.attrs
            .setdefault('flatbread', {})
            .setdefault('formats', {})[column]
         ) = format_type
