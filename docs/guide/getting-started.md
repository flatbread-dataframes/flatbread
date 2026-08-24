# Getting Started

Flatbread is a pandas extension for adding totals, percentages, differences,
and other tabulations to DataFrames and Series. It registers a `pita` accessor
on both types.

## Installation

```bash
pip install flatbread
```

## A first example

We'll work with a small dataset of wildlife sightings across three regions.
The raw data records individual counts by region, species class, species,
and season. We pivot it into a summary table:

```python
--8<-- "examples/getting-started/01_input.py"
```

<flatbread-table src="/assets/examples/getting-started/01_input.json" hide-settings-menu></flatbread-table>

## Adding totals

The `pita.add_totals()` method appends totals to both rows and columns.
Use the `axis` parameter to control which: `0` for rows only, `1` for
columns only, or `2` (the default) for both.

```python
--8<-- "examples/getting-started/02_add_totals.py"
```

<flatbread-table src="/assets/examples/getting-started/02_add_totals.json" margin-labels="Totals" hide-settings-menu></flatbread-table>

## Adding percentages

Chain `pita.add_percentages()` after totals to add a percentage panel
alongside the counts. Flatbread calculates percentages from the totals
already present in the table.

```python
--8<-- "examples/getting-started/03_add_percentages.py"
```

<flatbread-table src="/assets/examples/getting-started/03_add_percentages.json" margin-labels="Totals" hide-settings-menu></flatbread-table>

The `axis` parameter controls what the percentages are relative to:
`0` for row totals, `1` for column totals, `2` (the default) for the
grand total.

## `as_*` vs `add_*`

Most transforms come in two forms:

- `as_percentages()` — replaces the data with percentages
- `add_percentages()` — keeps the original data and adds percentages as a separate panel

The same pattern applies to differences and percentage change. The getting
started examples use the `add_*` form. Both are covered in detail in the
[Panels](panels.md) guide.

## Next steps

- [Aggregation](guide/aggregation.md) — totals, subtotals, custom aggregations, sorting
- [Panels](guide/panels.md) — percentages, differences, percentage change, interleaving
- [Configuration](guide/configuration.md) — config files, layering, runtime overrides
- [Formatting](guide/formatting.md) — format presets, manual formatting, pattern matching
- [Display & Export](guide/display.md) — table styling, Excel export