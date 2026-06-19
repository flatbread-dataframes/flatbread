import decimal
import json
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
            "columns": {
                "values": self._prepare_columns(),
                "names": list(self._data.columns.names),
                "dtypes": self._prepare_column_dtypes(),
                "formatOptions": self._prepare_column_format_options(),
            },
            "index": {
                "values": self._prepare_index(),
                "names": list(self._data.index.names),
                "dtypes": self._prepare_index_dtypes(),
                "formatOptions": self._prepare_index_format_options(),
            },
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

    def _prepare_column_dtypes(self) -> list[str]:
        return [
            self._format_resolver.dtype_mappings.get(str(dtype), "str")
            for dtype in self._data.dtypes
        ]

    def _prepare_column_format_options(self) -> list[str | dict[str, Any] | None]:
        """Get format options for each column"""
        return [self._get_format(col) for col in self._data.columns]

    def _prepare_index_dtypes(self) -> list[str | None]:
        """Convert index level dtypes to simplified type names."""
        index = self._data.index
        return [
            self._format_resolver.dtype_mappings.get(
                str(index.get_level_values(level).dtype), None
            )
            for level in range(index.nlevels)
        ]

    def _prepare_index_format_options(self) -> list[dict | str | None]:
        """Get format options for each index level."""
        return [self._get_format(name) for name in self._data.index.names]

    def _get_format(self, key: str | None) -> ColumnFormat | None:
        if not key:
            return None
        if format_spec := self._format_options.get(key):
            return format_spec
        return self._format_resolver.get_html_format(key)

    def _resolve_dtype(self, key) -> str | None:
        """Resolve simplified dtype for a column or index level name.

        Parameters
        ----------
        key : str
            Name to look up in columns and index.

        Returns
        -------
        str | None
            Simplified dtype string, or None if key not found.

        Raises
        ------
        KeyError
            If key matches neither a column nor an index level name.
        """
        if key in self._data.columns:
            return self._format_resolver.dtype_mappings.get(
                str(self._data[key].dtype), "str"
            )
        if key in (self._data.index.names or []):
            level = self._data.index.names.index(key)
            dtype = self._data.index.get_level_values(level).dtype
            return self._format_resolver.dtype_mappings.get(str(dtype), "str")
        raise KeyError(f"'{key}' not found in columns or index level names.")

    def set_format(self, key: str, format_spec: str | dict[str, Any]) -> None:
        """Set format options for a column or index level.

        Parameters
        ----------
        key : str
            Column name or index level name to format.
        format_spec : str | dict
            Either a preset name (e.g. 'currency_eur') or format options dict.

        Raises
        ------
        KeyError
            If key matches neither a column nor an index level name.
        ValueError
            If a preset is not compatible with the key's dtype.
        """
        if isinstance(format_spec, str):
            simple_dtype = self._resolve_dtype(key)

            # Check user-defined presets
            if format_spec in self._format_resolver.format_presets:
                preset_config = self._format_resolver.format_presets[format_spec]
                allowed_dtypes = preset_config.get("dtypes", ["float", "int"])

                if simple_dtype in allowed_dtypes:
                    self._format_options[key] = preset_config.get("html_options", {})
                    return
                raise ValueError(
                    f"Preset '{format_spec}' is not compatible with '{key}' "
                    f"of dtype '{simple_dtype}'. "
                    f"This preset supports: {', '.join(allowed_dtypes)}"
                )

            # Check output format types
            if format_spec in self._format_resolver.output_formats:
                format_config = self._format_resolver.output_formats[format_spec]
                self._format_options[key] = format_config.get("html_options", {})
                return

            # Unknown preset
            available_presets = list(self._format_resolver.format_presets.keys())
            available_formats = list(self._format_resolver.output_formats.keys())
            all_available = available_presets + available_formats
            raise ValueError(
                f"Unknown format '{format_spec}'. "
                f"Available options: {', '.join(all_available)}"
            )

        self._format_options[key] = format_spec

    def set_formats(self, formats: FormatSpec) -> None:
        """Set formats for columns and/or index levels.

        Parameters
        ----------
        formats : str, dict, list or callable
            - If string: apply the same format preset to all columns
            - If dict: mapping of names/patterns to format specs,
            matched against both column names and index level names
            - If list: format specs in same order as columns
            - If callable: function that takes DataFrame and returns a dict
        """
        if isinstance(formats, str):
            formats = {column: formats for column in self._data.columns}

        if callable(formats):
            formats = formats(self._data)

        if isinstance(formats, list):
            if len(formats) != len(self._data.columns):
                raise ValueError(
                    f"Expected {len(self._data.columns)} formats, got {len(formats)}"
                )
            formats = dict(zip(self._data.columns, formats))

        pattern_matches = {}
        for pattern, format_spec in formats.items():
            for column in self._data.columns:
                if self._is_pattern_match(column, pattern):
                    pattern_matches[column] = format_spec
            for name in self._data.index.names:
                if name and self._is_pattern_match(name, pattern):
                    pattern_matches[name] = format_spec

        for key, format_spec in pattern_matches.items():
            self.set_format(key, format_spec)

    def _is_pattern_match(self, key: Any, pattern: Any) -> bool:
        """
        Check if a column matches a pattern.

        Parameters
        ----------
        key : Any
            The key name to check (could be tuple for MultiIndex)
        pattern : Any
            The pattern to match against

        Returns
        -------
        bool
            True if the column matches the pattern
        """
        # Direct equality check
        if key == pattern:
            return True

        # For MultiIndex columns (tuples)
        if isinstance(key, tuple):
            # Case 1: If pattern is also a tuple, check if it's a prefix
            if isinstance(pattern, tuple) and len(pattern) <= len(key):
                return key[: len(pattern)] == pattern

            # Case 2: If pattern is a scalar, check if it's in any level
            else:
                return any(part == pattern for part in key)

        # For string columns, check if pattern is substring
        elif isinstance(key, str) and isinstance(pattern, str):
            return pattern in key

        return False

    def _serialize_to_json(self, data: dict) -> str:
        """Safely serialize data to JSON for JS consumption"""
        return json.dumps(data, separators=(",", ":"), default=self._json_serialize)

    @staticmethod
    def _json_serialize(obj):
        """Handle special types for JSON serialization"""
        if isinstance(obj, pd.Timestamp):
            timestamp = obj.isoformat()
            if timestamp.endswith("T00:00:00"):
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
        if hasattr(obj, "dtype"):
            return obj.item()
        return str(obj)
