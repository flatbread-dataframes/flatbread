"""Constants for data-viewer formatting"""
from flatbread import DEFAULTS


USER_PRESETS = DEFAULTS.get("format_presets", {})
USER_PRESETS_BY_DTYPE = {
    dtype: set()
    for dtype in ["float", "int", "datetime", "str", "category"]
}

for preset_name, preset_config in USER_PRESETS.items():
    preset_dtypes = preset_config.get("dtypes", ["float", "int"])  # Default to numeric
    for dtype in preset_dtypes:
        if dtype in USER_PRESETS_BY_DTYPE:
            USER_PRESETS_BY_DTYPE[dtype].add(preset_name)


SMART_FORMATS = {
    'percentages': {
        'labels': [DEFAULTS['transforms']['percentages']['label_pct']],
        'options': {
            'style': 'percent',
            'minimumFractionDigits': 0,
            'maximumFractionDigits':
                DEFAULTS['transforms']['percentages']['ndigits'] if DEFAULTS['transforms']['percentages']['ndigits'] >= 0 else 21,
        },
        'excel_format': '0.0%'
    },
    'difference': {
        'labels': ['diff'],
        'options': {
            'signDisplay': 'always',
        },
        'excel_format': '+#,##0;-#,##0',
    }
}

DEFAULT_DTYPES: dict[str, str] = {
    'object':         'str',
    'string':         'str',
    'bool':           'bool',
    'boolean':        'bool',
    'category':       'category',
    'datetime64[ns]': 'datetime',
    'float64':        'float',
    'Float64':        'float',
    'int8':           'int',
    'int16':          'int',
    'int32':          'int',
    'int64':          'int',
    'Int8':           'int',
    'Int16':          'int',
    'Int32':          'int',
    'Int64':          'int',
}

NUMBER_PRESETS = {"default", "currency", "percentage", "compact", "diffs"}
DATE_PRESETS = {"default", "date", "datetime"}

DTYPE_TO_PRESETS = {
    "float": NUMBER_PRESETS.union(USER_PRESETS_BY_DTYPE["float"]),
    "int": NUMBER_PRESETS.union(USER_PRESETS_BY_DTYPE["int"]),
    "datetime": DATE_PRESETS.union(USER_PRESETS_BY_DTYPE["datetime"]),
}
