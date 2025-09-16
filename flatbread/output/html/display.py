import uuid
from dataclasses import dataclass, field, fields, MISSING
from typing import Any

from jinja2 import Environment, PackageLoader

from flatbread import DEFAULTS
from flatbread.output.html.tablespec import TableSpecBuilder, FormatSpec


# region config
@dataclass
class DisplayConfig:
    # Data handling
    locale: str | None = None
    na_rep: str = "-"
    margin_labels: set[str] = field(default_factory=set)

    # Layout control
    collapse_columns: bool | None = None
    max_rows: int = 30
    max_columns: int = 30
    trim_size: int = 5
    separator: str = "..."

    # Border controls
    hide_column_borders: bool = False
    hide_row_borders: bool = False
    hide_thead_border: bool = False
    hide_index_border: bool = False

    # Visual effects
    show_hover: bool = False

    @classmethod
    def from_defaults(
        cls,
        defaults: dict[str, Any],
        data_attrs: dict|None = None,
    ) -> "DisplayConfig":
        """Create config instance from defaults dict"""
        if not defaults:
            return cls()

        # Extract standard config fields from defaults
        standard_fields = {
            field.name: defaults.get(field.name, field.default)
            for field in fields(cls)
            if field.name != 'margin_labels'
        }

        # Handle computed fields with custom logic
        computed_fields = {
            'margin_labels': cls._extract_margin_labels(defaults, data_attrs)
        }

        return cls(**(standard_fields | computed_fields))

    @classmethod
    def _extract_margin_labels(
        cls,
        defaults: dict[str, Any],
        data_attrs: dict|None
    ) -> set[str]:
        """Extract margin labels from defaults and data_attrs"""
        margin_labels = set()
        transforms = defaults.get('transforms', {})
        data_attrs = {} if data_attrs is None else data_attrs
        attr_labels = data_attrs.get('flatbread', {}).get('labels')

        for transform_config in transforms.values():
            config_labels = transform_config.get('margin_labels', [])
            for margin_label in config_labels:
                if margin_label in transform_config:
                    label_value = transform_config[margin_label]
                    if label_value is not None:
                        margin_labels.add(label_value)
                if attr_labels and margin_label in attr_labels:
                    attr_label = attr_labels[margin_label]
                    if attr_label is not None:
                        margin_labels.add(attr_label)

        return margin_labels


# region manager
class TemplateManager:
    """Manages rendering templates"""
    def __init__(self):
        self._env = Environment(loader=PackageLoader("flatbread", "output/html"))

    def render(self, spec: str, config: DisplayConfig) -> str:
        template = self._env.get_template("templates/template.jinja.html")
        html = template.render(
            data   = spec,
            config = config,
            id     = f"id-{uuid.uuid4()}"
        )
        return html


# region display
class PitaDisplayMixin:
    """Mixin for displaying pandas objects using data-viewer"""
    @property
    def _config(self) -> DisplayConfig:
        if not hasattr(self, '_display_config'):
            self._display_config = DisplayConfig.from_defaults(
                DEFAULTS,
                self._obj.attrs if hasattr(self._obj, 'attrs') else None
            )
        return self._display_config

    @property
    def _table_spec_builder(self) -> TableSpecBuilder:
        """Lazy initialization of spec builder"""
        if not hasattr(self, '_spec_builder'):
            self._spec_builder = TableSpecBuilder(self._obj)
        return self._spec_builder

    @property
    def _template_manager(self) -> TemplateManager:
        """Lazy initialization of template manager"""
        if not hasattr(self, '_template_mgr'):
            self._template_mgr = TemplateManager()
        return self._template_mgr

    def configure_display(self, **kwargs) -> "PitaDisplayMixin":
        """Configure display options"""
        self._config.update(**kwargs)
        return self

    def set_locale(self, locale: str) -> "PitaDisplayMixin":
        """Set the locale for number/date formatting"""
        self._config.locale = locale
        return self

    def set_na_rep(self, na_rep: str) -> "PitaDisplayMixin":
        """Set null value representation"""
        self._config.na_rep = na_rep
        return self

    def set_max_rows(self, max_rows: int) -> "PitaDisplayMixin":
        """Set maximum rows before truncating"""
        self._config.max_rows = max_rows
        return self

    def set_max_columns(self, max_columns: int) -> "PitaDisplayMixin":
        """Set maximum columns before truncating"""
        self._config.max_columns = max_columns
        return self

    def set_trim_size(self, n: int) -> "PitaDisplayMixin":
        """Set number of items to show when truncated"""
        self._config.trim_size = n
        return self

    def set_separator(self, sep: str) -> "PitaDisplayMixin":
        """Set truncation indicator"""
        self._config.separator = sep
        return self

    def hide_borders(self, hide: bool = True) -> "PitaDisplayMixin":
        """Hide all borders"""
        self._config.hide_column_borders = hide
        self._config.hide_row_borders = hide
        self._config.hide_thead_border = hide
        self._config.hide_index_border = hide
        return self

    def show_column_borders(self, show: bool = True) -> "PitaDisplayMixin":
        """Show/hide vertical column borders"""
        self._config.hide_column_borders = not show
        return self

    def show_row_borders(self, show: bool = True) -> "PitaDisplayMixin":
        """Show/hide horizontal row borders"""
        self._config.hide_row_borders = not show
        return self

    def show_header_border(self, show: bool = True) -> "PitaDisplayMixin":
        """Show/hide header bottom border"""
        self._config.hide_thead_border = not show
        return self

    def show_index_border(self, show: bool = True) -> "PitaDisplayMixin":
        """Show/hide index right border"""
        self._config.hide_index_border = not show
        return self

    def show_hover(self, show: bool = True) -> "PitaDisplayMixin":
        """Enable row hover effect"""
        self._config.show_hover = show
        return self

    def collapse_columns(self, collapse: bool = True) -> "PitaDisplayMixin":
        """Collapse column headers"""
        self._config.collapse_columns = collapse
        return self

    def set_section_levels(self, levels: int) -> "PitaDisplayMixin":
        """Set index levels to show as sections"""
        self._config.section_levels = levels
        return self

    def set_margin_labels(self, *labels: str) -> "PitaDisplayMixin":
        """Set labels to be treated as margins"""
        self._config.margin_labels = set(labels)
        return self

    def format(
        self,
        column: str,
        format_spec: str | dict[str, Any],
    ) -> "PitaDisplayMixin":
        """Set format options for a column

        Parameters
        ----------
        column : str
            Column to format
        format_spec : str | dict
            Either a preset name (e.g. 'currency') or format options dict

        Returns
        -------
        PitaDisplayMixin
            Self for method chaining
        """
        self._table_spec_builder.set_format(column, format_spec)
        return self

    def format_columns(self, formats: FormatSpec) -> "PitaDisplayMixin":
        """Set multiple column formats at once"""
        self._table_spec_builder.set_formats(formats)
        return self

    def get_format_presets(self, dtype: str | None = None) -> dict[str, dict]:
        """Get available format presets, optionally filtered by dtype"""
        # Use format resolver instead of importing constants
        resolver = self._table_spec_builder._format_resolver

        # Combine output formats and format presets
        all_presets = {}

        # Add output formats (these are always available)
        for name, config in resolver.output_formats.items():
            all_presets[name] = config.get('html_options', {})

        # Add user-defined format presets
        for name, config in resolver.format_presets.items():
            all_presets[name] = config.get('html_options', {})

        # Filter by dtype if specified (simplified logic)
        if dtype:
            # Could add dtype filtering logic here if needed
            pass

        return all_presets

    def _repr_html_(self) -> str:
        """Generate HTML representation for Jupyter display"""
        spec = self._table_spec_builder.get_spec_as_json()
        return self._template_manager.render(spec, self._config)

    def get_table_spec(self) -> dict:
        """
        Get the raw table specification as a dictionary.

        Returns
        -------
        dict
            Dictionary containing the complete table specification including:
            - values: 2D array of cell values
            - columns: Column labels
            - index: Row labels
            - columnNames: Names for column levels
            - indexNames: Names for index levels
            - dtypes: Data types per column
            - formatOptions: Format configuration per column
        """
        return self._table_spec_builder.build_spec()

    def get_table_spec_json(self) -> str:
        """
        Get the table specification as a JSON string.

        Returns
        -------
        str
            JSON string containing the complete table specification.
            The JSON is serialized using the same custom serialization logic
            used for display, handling pandas-specific types like Timestamps
            and Intervals.
        """
        return self._table_spec_builder.get_spec_as_json()
