# Configuration

Flatbread uses a layered configuration system. Settings are resolved by
merging up to three sources in order, where later sources override
earlier ones:

1. **Package defaults** — built-in `config.defaults.json`
2. **User config** — `~/.flatbread.json`
3. **Project config** — `.flatbread.json` in or above the working directory

The merged result is available as `flatbread.DEFAULTS` and can be
further adjusted at runtime.

## Config files

Both user and project config files are plain JSON. They don't need to
contain the full config — only the keys you want to override. Flatbread
deep-merges them into the defaults, so nested keys you don't mention
are preserved.

For example, to change the totals label and set a default locale,
create a `.flatbread.json` in your project root:

```json
{
    "transforms": {
        "totals": {
            "label": "Total"
        }
    },
    "locale": "nl-NL"
}
```

The user config at `~/.flatbread.json` works the same way but applies
across all projects. Project config takes precedence over user config.

### Project config lookup

Flatbread searches for `.flatbread.json` starting from the current
working directory and walking up to 5 parent directories. It stops at
the home directory or filesystem root, whichever comes first.

## Runtime overrides

For in-session changes, use `update_runtime`:

```python
from flatbread import DEFAULTS

DEFAULTS.update_runtime({
    "transforms": {
        "subtotals": {
            "label": "Subtotal",
            "include_level_name": True,
        }
    }
})
```

This deep-merges into the already-resolved config. The change persists
for the rest of the session but is not written to disk.

## Inspecting the config

`print(DEFAULTS)` shows which files were loaded and the final merged
result:

```python
from flatbread import DEFAULTS

print(DEFAULTS)
# ConfigService loaded from 2 sources:
#   1. config.defaults.json
#   2. .flatbread.json
#
# Final config: { ... }
```

`DEFAULTS.sources` returns the list of file paths that were loaded.
`DEFAULTS.reload()` clears the cached config and reloads from disk on
next access.

## Config reference

The full default configuration:

```json
{
    "transforms": {
        "totals": {
            "label": "Totals",
            "ignore_transforms": ["totals", "percentages", "differences"]
        },
        "subtotals": {
            "label": "Subtotals",
            "include_level_name": false,
            "ignore_transforms": ["totals", "percentages", "differences"]
        },
        "percentages": {
            "label_n": "n",
            "label_pct": "pct",
            "ndigits": -1,
            "base": 1,
            "ignore_transforms": []
        },
        "differences": {
            "label_n": "n",
            "label_diff": "diff",
            "ignore_transforms": ["totals", "percentages", "differences", "pct_change"]
        },
        "pct_change": {
            "label_n": "n",
            "label_pct_change": "pct_change",
            "ignore_transforms": ["totals", "percentages", "differences", "pct_change"]
        }
    },
    "panels": {
        "axis_suffixes": {
            "0": "col",
            "1": "row",
            "2": "total"
        }
    },
    "locale": null,
    "output_formats": { ... },
    "format_presets": { ... },
    "dtype_mappings": { ... }
}
```

The `transforms` section controls labels and chaining behavior for each
transform type. The `label` fields set default row/column labels.
The `label_n`, `label_pct`, `label_diff`, and `label_pct_change` fields
control the panel column labels added by `add_percentages`,
`add_differences`, and `add_pct_change`. The `ignore_transforms` lists
determine which existing transforms are excluded from calculations when
chaining — for example, totals rows are excluded when computing
differences.

The `panels.axis_suffixes` section controls the suffixes appended to
panel labels based on the axis parameter. When you call
`add_percentages(axis=0)`, the percentage column label becomes `pct_col`
(the configured `label_pct` plus the axis 0 suffix).

The `output_formats`, `format_presets`, and `dtype_mappings` sections
are covered in the [Formatting](formatting.md) guide.