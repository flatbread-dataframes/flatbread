import json
import decimal
from typing import Any, Callable

import pandas as pd

from flatbread.output.formats import FormatResolver


ColumnFormat = str | dict[str, Any]
ColumnFormats = dict[str, ColumnFormat] | list[ColumnFormat]
FormatSpec = ColumnFormats | Callable[[pd.DataFrame], ColumnFormats]


class TableSpecBuilder:
    """Converts pandas objects to data-viewer specifications"""
    def __init__(self, data: pd.DataFrame | pd.Series):
        self._data = data.to_frame() if isinstance(data, pd.Series) else data
        self._format_options: dict[str, str | dict[str, Any]] = {}
        self._format_resolver = FormatResolver(self._data)

    def build_spec(self) -> dict:
        return {
            "values": self._prepare_values(),
            "columns": self._prepare_columns(),
            "index": self._prepare_index(),
            "columnNames": self._data.columns.names,
            "indexNames": self._data.index.names,
            "dtypes": self._prepare_dtypes(),
            "formatOptions": self._prepare_format_options()
        }

    def get_spec_as_json(self) -> str:
        spec = self.build_spec()
        as_json = self._serialize_to_json(spec)
        return as_json

    def _prepare_values(self) -> list[list]:
        """Convert DataFrame values to nested list format"""
        values = [
            [None if pd.isna(i) else i for i in row]
            for row in self._data.values.tolist()
        ]
        return values

    def _prepare_columns(self) -> list:
        """Prepare column labels"""
        return list(self._data.columns)

    def _prepare_index(self) -> list:
        """Prepare index labels"""
        return list(self._data.index)

    def _prepare_dtypes(self) -> list[str]:
        return [
            self._format_resolver.dtype_mappings.get(str(dtype), 'str')
            for dtype in self._data.dtypes
        ]

    def _prepare_format_options(self) -> list[str | dict[str, Any] | None]:
        """Get format options for each column"""
        return [
            self._get_format_for_column(col)
            for col in self._data.columns
        ]

    def _get_format_for_column(self, column: Any) -> ColumnFormat | None:
        # First check manual overrides (set via set_format method)
        if format_spec := self._format_options.get(column):
            return format_spec
        # Then use format resolver for smart + explicit detection
        return self._format_resolver.get_html_format(column)

    def set_format(self, column: str, format_spec: str | dict[str, Any]) -> None:
        """Set format options for a column

        Parameters
        ----------
        column : str
            Column to format
        format_spec : str | dict
            Either a preset name (e.g. 'currency_eur') or format options dict
        """
        if isinstance(format_spec, str):
            # Check if it's a user-defined preset from format_presets
            if format_spec in self._format_resolver.format_presets:
                pandas_dtype = str(self._data[column].dtype)
                simple_dtype = self._format_resolver.dtype_mappings.get(pandas_dtype, 'str')

                # Get allowed dtypes for this preset
                preset_config = self._format_resolver.format_presets[format_spec]
                allowed_dtypes = preset_config.get("dtypes", ["float", "int"])

                if simple_dtype in allowed_dtypes:
                    # Store the HTML options directly
                    self._format_options[column] = preset_config.get("html_options", {})
                    return
                else:
                    raise ValueError(
                        f"Preset '{format_spec}' is not compatible with column '{column}' "
                        f"of dtype {pandas_dtype} (mapped to {simple_dtype}). "
                        f"This preset supports: {', '.join(allowed_dtypes)}"
                    )

            # Check if it's an output format type
            elif format_spec in self._format_resolver.output_formats:
                format_config = self._format_resolver.output_formats[format_spec]
                self._format_options[column] = format_config.get("html_options", {})
                return

            else:
                # Not a valid preset or format type
                available_presets = list(self._format_resolver.format_presets.keys())
                available_formats = list(self._format_resolver.output_formats.keys())
                all_available = available_presets + available_formats

                raise ValueError(
                    f"Unknown format '{format_spec}'. Available options: {', '.join(all_available)}"
                )

        # If we reached here, format_spec is a dict of HTML options
        self._format_options[column] = format_spec

    def set_formats(self, formats: FormatSpec) -> None:
        """Set multiple column formats at once.

        Parameters
        ----------
        formats : str, dict, list or callable
            - If string: apply the same format preset to all columns
            - If dict: mapping column names to format specs
            - If list: format specs in same order as columns
            - If callable: function that takes DataFrame and returns a dict
        """
        if isinstance(formats, str):
            formats = {column: formats for column in self._data.columns}

        if callable(formats):
            formats = formats(self._data)

        if isinstance(formats, list):
            if len(formats) != len(self._data.columns):
                raise ValueError(f"Expected {len(self._data.columns)} formats, got {len(formats)}")
            formats = dict(zip(self._data.columns, formats))

        # Handle pattern matching for dictionary keys
        if isinstance(formats, dict):
            # Use an ordered dictionary to maintain the order of pattern matches
            pattern_matches = {}

            # Apply patterns in the order they were provided
            for pattern, format_spec in formats.items():
                for column in self._data.columns:
                    if self._is_pattern_match(column, pattern):
                        # This will overwrite any previous match for this column
                        pattern_matches[column] = format_spec

            # Apply all the formats
            for column, format_spec in pattern_matches.items():
                self.set_format(column, format_spec)
        else:
            for column, format_spec in formats.items():
                self.set_format(column, format_spec)

    def _is_pattern_match(self, column: Any, pattern: Any) -> bool:
        """
        Check if a column matches a pattern.

        Parameters
        ----------
        column : Any
            The column name to check (could be tuple for MultiIndex)
        pattern : Any
            The pattern to match against

        Returns
        -------
        bool
            True if the column matches the pattern
        """
        # Direct equality check
        if column == pattern:
            return True

        # For MultiIndex columns (tuples)
        if isinstance(column, tuple):
            # Case 1: If pattern is also a tuple, check if it's a prefix
            if isinstance(pattern, tuple) and len(pattern) <= len(column):
                return column[:len(pattern)] == pattern

            # Case 2: If pattern is a scalar, check if it's in any level
            else:
                return any(part == pattern for part in column)

        # For string columns, check if pattern is substring
        elif isinstance(column, str) and isinstance(pattern, str):
            return pattern in column

        return False

    def _serialize_to_json(self, data: dict) -> str:
        """Safely serialize data to JSON for JS consumption"""
        return json.dumps(data, separators=(',', ':'), default=self._json_serialize)

    @staticmethod
    def _json_serialize(obj):
        """Handle special types for JSON serialization"""
        if isinstance(obj, pd.Timestamp):
            timestamp = obj.isoformat()
            if timestamp.endswith('T00:00:00'):
                return timestamp[:-9]
            return timestamp
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        if isinstance(obj, (pd.Index, pd.arrays.IntervalArray)):
            return list(obj)
        if pd.isna(obj):
            return None
        if isinstance(obj, pd._libs.interval.Interval):
            return str(obj)
        if hasattr(obj, 'dtype'):
            return obj.item()
        return str(obj)
