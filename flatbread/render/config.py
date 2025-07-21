from dataclasses import dataclass, field
from typing import Any

@dataclass
class DisplayConfig:
    # Data handling
    locale: str | None = None
    na_rep: str = "-"
    margin_labels: list[str] = field(default_factory=list)

    # Layout control
    collapse_columns: bool = None
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

        margin_labels = set()
        transforms = defaults.get('transforms', {})
        data_attrs = {} if data_attrs is None else data_attrs
        attr_labels = data_attrs.get('flatbread', {}).get('labels')

        for transform_config in transforms.values():
            config_labels = transform_config.get('margin_labels', [])

            for margin_label in config_labels:
                # Config-based label
                if margin_label in transform_config:
                    label_value = transform_config[margin_label]
                    if label_value is not None:
                        margin_labels.add(label_value)

                # Runtime attrs label
                if attr_labels and margin_label in attr_labels:
                    attr_label = attr_labels[margin_label]
                    if attr_label is not None:
                        margin_labels.add(attr_label)

        # Extract values from defaults, using dataclass defaults if not present
        return cls(
            locale = defaults.get("locale", cls.locale),
            na_rep = defaults.get("na_rep", cls.na_rep),
            margin_labels = list(set(margin_labels)),
            collapse_columns=defaults.get("collapse_columns", cls.collapse_columns),
            max_rows=defaults.get("max_rows", cls.max_rows),
            max_columns=defaults.get("max_columns", cls.max_columns),
            trim_size=defaults.get("trim_size", cls.trim_size),
            separator=defaults.get("separator", cls.separator),
            hide_column_borders=defaults.get("hide_column_borders", cls.hide_column_borders),
            hide_row_borders=defaults.get("hide_row_borders", cls.hide_row_borders),
            hide_thead_border=defaults.get("hide_thead_border", cls.hide_thead_border),
            hide_index_border=defaults.get("hide_index_border", cls.hide_index_border),
            show_hover=defaults.get("show_hover", cls.show_hover)
        )
