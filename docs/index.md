# Flatbread

Flatbread is a pandas extension for tabulation — totals, subtotals,
percentages, differences, and more. It registers a `pita` accessor on
DataFrames and Series.

```python
--8<-- "examples/index/01_showcase.py"
```

<flatbread-table src="/assets/examples/index/01_showcase.json" margin-labels="Subtotals;Totals" section-levels="1" hide-row-borders hide-settings-menu></flatbread-table>

## Installation

```bash
pip install flatbread
```

## Guide

- [Getting Started](guide/getting-started.md) — install, import, basic pipeline
- [Aggregation](guide/aggregation.md) — totals, subtotals, custom aggregations, sorting
- [Panels](guide/panels.md) — percentages, differences, percentage change, interleaving
- [Display & Export](guide/display-export.md) — table styling, Excel export

## API Reference

- [PitaFrame](api/dataframe.md) — DataFrame accessor
- [PitaSeries](api/series.md) — Series accessor
- [Display](api/display.md) — display configuration