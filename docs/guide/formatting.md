# Formatting

Flatbread automatically formats values for display and export based on
the type of data in each column. Percentage columns get percentage
formatting, difference columns show a sign — this works out of the box
when you use flatbread's transforms. You can also define your own
format presets and apply formats manually.

## How format resolution works

When rendering a table (HTML or Excel), flatbread resolves the format
for each column through three steps, stopping at the first match:

1. **Explicit format** — set manually via `.format()` or
   `.format_columns()` (highest priority)
2. **Panel metadata** — if the column belongs to a panel created by
   `add_percentages`, `add_differences`, or `add_pct_change`,
   flatbread maps the panel type to a format automatically
3. **Smart label detection** — column names are checked against
   `smart_labels` patterns defined in `output_formats`

In practice, step 2 handles most cases. If you use `add_percentages`,
the resulting percentage columns are formatted as percentages without
any manual intervention. Step 3 catches columns that happen to contain
keywords like `pct` or `diff` in their name even without panel metadata.

## Output formats

The `output_formats` section in the [configuration](configuration.md)
defines format types that flatbread uses internally. Each entry has an
`html_options` dict (passed to `Intl.NumberFormat` in the browser), an
`excel_format` string, and `smart_labels` for automatic detection:

```json
{
    "output_formats": {
        "percentage": {
            "smart_labels": ["pct", "%"],
            "html_options": {
                "style": "percent",
                "minimumFractionDigits": 0,
                "maximumFractionDigits": 1
            },
            "excel_format": "0.0%"
        },
        "signed_integer": {
            "smart_labels": ["diff", "Δ"],
            "html_options": {
                "signDisplay": "always"
            },
            "excel_format": "+#,##0;-#,##0"
        },
        "signed_percentage": {
            "smart_labels": ["pct_change", "Δ%"],
            "html_options": {
                "style": "percent",
                "signDisplay": "always"
            },
            "excel_format": "+0.0%;-0.0%"
        }
    }
}
```

These are designed to match flatbread's own panel labels. You can
override them in your `.flatbread.json` if you need different
formatting for these types.

## Format presets

Format presets are reusable named formats you can reference by name
instead of spelling out the full options dict. They're defined in the
`format_presets` section of the config:

```json
{
    "format_presets": {
        "currency_eur": {
            "dtypes": ["float", "int"],
            "html_options": {
                "style": "currency",
                "currency": "EUR"
            },
            "excel_format": "#,##0.00 €"
        }
    }
}
```

The `dtypes` field restricts which column types the preset can be
applied to. If you try to apply `currency_eur` to a string column,
flatbread raises a `ValueError`.

You can add your own presets in `~/.flatbread.json` or a project-level
`.flatbread.json`:

```json
{
    "format_presets": {
        "currency_usd": {
            "dtypes": ["float", "int"],
            "html_options": {
                "style": "currency",
                "currency": "USD"
            },
            "excel_format": "$#,##0.00"
        },
        "compact": {
            "dtypes": ["float", "int"],
            "html_options": {
                "notation": "compact",
                "maximumFractionDigits": 1
            },
            "excel_format": "0.0"
        }
    }
}
```

## Applying formats

### Single column

Use `.format()` with a preset name to format a single column. Here
we format a conservation funding table:

```python
--8<-- "examples/formatting/01_format_single.py"
```

<flatbread-table src="/assets/examples/formatting/01_format_single.json" hide-settings-menu></flatbread-table>

### Multiple columns

`.format_columns()` applies formats to several columns at once. Pass
a dict mapping column names (or patterns) to format specs — these can
be preset names or raw `Intl.NumberFormat` options dicts:

```python
--8<-- "examples/formatting/02_format_columns.py"
```

<flatbread-table src="/assets/examples/formatting/02_format_columns.json" margin-labels="Totals" hide-settings-menu></flatbread-table>

The other input shapes for `format_columns`:

```python
# A single preset applied to all columns
df.pita.format_columns("currency_eur")

# A list in column order
df.pita.format_columns(["currency_eur", "currency_eur", None])

# A callable that receives the DataFrame and returns a dict
df.pita.format_columns(lambda df: {
    col: "currency_eur" for col in df.columns if "budget" in str(col)
})
```

### Pattern matching

When passing a dict to `.format_columns()`, keys are matched against
column names using pattern matching. For string columns, a pattern
matches on equality or substring containment. For MultiIndex columns
(tuples), a pattern matches if it equals any level of the tuple or is
a tuple prefix:

```python
# Matches all columns with "budget" in any level
df.pita.format_columns({"budget": "currency_eur"})

# Matches a specific MultiIndex column
df.pita.format_columns({("Q1", "budget"): "currency_eur"})
```

Patterns also match against index level names, so you can format
index levels the same way.

## Discovering available presets

Call `get_format_presets()` to see all available presets — both the
built-in output formats and any user-defined format presets:

```python
df.pita.get_format_presets()
# {
#     'percentage': {'style': 'percent', ...},
#     'signed_integer': {'signDisplay': 'always'},
#     'signed_percentage': {'style': 'percent', 'signDisplay': 'always'},
#     'currency_eur': {'style': 'currency', 'currency': 'EUR'},
# }
```

## HTML and Excel

The same format definitions drive both outputs. The `html_options` dict
is passed to `Intl.NumberFormat` in the `flatbread-table` viewer for
browser rendering. The `excel_format` string is applied as the cell
number format when exporting via `export_excel`. This means a preset
defined once in your config works consistently across both display
contexts.