# Display & Export

Flatbread renders tables using the `flatbread-table` web component. In a
notebook, this happens automatically via `_repr_html_`. In static pages
like these docs, you use the `<flatbread-table>` HTML tag directly.

Display options can be set from Python using `configure_display` (for
notebooks) or as HTML attributes on the `<flatbread-table>` tag. Both
control the same settings.

## Section headers

For tables with a MultiIndex, `section-levels` converts one or more
index levels into section headers instead of repeated values in rows:

In a notebook:

```python
df.pita.configure_display(section_levels=1)
```

As HTML:

```html
<flatbread-table
    src="data.json"
    margin-labels="Subtotals;Totals"
    section-levels="1"
    hide-settings-menu
></flatbread-table>
```

<flatbread-table src="/assets/examples/display/02_hierarchical.json" margin-labels="Subtotals;Totals" section-levels="1" hide-settings-menu></flatbread-table>

Without section headers, the same table looks like this:

<flatbread-table src="/assets/examples/display/02_hierarchical.json" margin-labels="Subtotals;Totals" hide-settings-menu></flatbread-table>

## Collapsing column headers

By default `flatbread-table` will merge the header row into the index header, to prevent this behavior
set `collapse-columns` to false. 

In a notebook:

```python
df.pita.configure_display(collapse_columns=False)
```

As HTML:

```html
<flatbread-table
    src="data.json"
    margin-labels="Totals"
    collapse-columns="false"
    hide-settings-menu
></flatbread-table>
```

<flatbread-table src="/assets/examples/display/01_flat.json" margin-labels="Totals" collapse-columns="false" hide-settings-menu></flatbread-table>

## Borders and hover

Several attributes control borders and interactivity:

In a notebook:

```python
df.pita.configure_display(
    show_hover = True,
    hide_row_borders = True,
    hide_index_border = True,
)
```

As HTML:

```html
<flatbread-table
    src="data.json"
    margin-labels="Subtotals;Totals"
    section-levels="1"
    show-hover
    hide-row-borders
    hide-index-border
    hide-settings-menu
></flatbread-table>
```

<flatbread-table src="/assets/examples/display/02_hierarchical.json" margin-labels="Subtotals;Totals" section-levels="1" show-hover hide-row-borders hide-index-border hide-settings-menu></flatbread-table>

## Margin labels

`margin-labels` tells the viewer which rows and columns represent
margins (totals, subtotals). These get visual styling — typically a
border separator and different background. Separate multiple labels with
semicolons:

```html
<flatbread-table margin-labels="Subtotals;Totals"></flatbread-table>
```

In a notebook, flatbread detects margin labels automatically from
`df.attrs` when you use `add_totals` or `add_subtotals`. To set them
manually:

```python
df.pita.configure_display(margin_labels={"Totals", "Subtotals"})
```

## Truncation

Large tables are truncated by default. The relevant settings:

In a notebook:

```python
df.pita.configure_display(
    max_rows = 30,        # default
    max_columns = 30,     # default
    trim_size = 5,        # rows shown at head and tail
    separator = "...",    # shown in the separator row
)
```

These map directly to the HTML attributes `max-rows`, `max-columns`,
`trim-size`, and `separator`.

## Locale and null values

```python
df.pita.configure_display(locale="nl-NL", na_rep="-")
```

`locale` controls number formatting (decimal separators, grouping).
`na_rep` controls what's shown for null values.

## All attributes

| Attribute | Default | Description |
|---|---|---|
| `src` | — | URL to load JSON data from |
| `locale` | `"default"` | Locale for number formatting |
| `na-rep` | — | String for null values |
| `margin-labels` | — | Semicolon-separated margin labels |
| `section-levels` | `0` | Index levels to show as sections |
| `collapse-columns` | `false` | Merge column header into index row |
| `column-border-levels` | `1` | Column border levels (-1: none, 0: all) |
| `show-hover` | `false` | Row hover highlighting |
| `hide-column-borders` | `false` | Hide vertical column borders |
| `hide-row-borders` | `false` | Hide horizontal row borders |
| `hide-thead-border` | `false` | Hide header bottom border |
| `hide-index-border` | `false` | Hide index right border |
| `no-wrap` | `false` | Prevent text wrapping |
| `max-rows` | `30` | Max rows before truncating |
| `max-columns` | `30` | Max columns before truncating |
| `trim-size` | `5` | Head/tail rows when truncated |
| `separator` | `"..."` | Separator row text |
| `hide-settings-menu` | `false` | Hide the settings menu |

## Excel export

Flatbread can export to Excel with automatic formatting. This requires
the `flatbreadxl` package:

```bash
pip install flatbreadxl
```

The basic call:

```python
df.pita.add_totals().pita.add_percentages().pita.export_excel("output.xlsx")
```

Flatbread auto-detects formatting from its own operations — percentage
columns get percentage number formats, margin rows and columns get
border separators. You can override both:

```python
df.pita.export_excel(
    "output.xlsx",
    title = "Wildlife Sightings",
    number_formats = {"Coast": "0.0"},
    border_specs = {"rows": ["Totals"], "columns": ["Totals"]},
)
```

The `number_formats` dict maps row/column labels to Excel format strings.
The `border_specs` dict specifies which labels should get border
separators, keyed by `"rows"` and `"columns"`.