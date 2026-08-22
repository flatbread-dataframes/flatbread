# Flatbread

Flatbread extends pandas with tabulation features. Access it through the `pita` accessor on DataFrames and Series.

## Features

- Totals and subtotals (rows and columns)
- Percentages, differences, percentage change
- Custom aggregations
- Interactive table display for Jupyter notebooks via [wc-simple-table](https://github.com/lcvriend/wc-simple-table) ([examples](https://lcvriend.github.io/wc-simple-table/))

## Quick Example

```python
import pandas as pd
import flatbread

df = pd.DataFrame(...)

result = (
    df
    .pita.add_totals()
    .pita.add_subtotals(level=0)
    .pita.add_percentages()
)

result.pita.configure_display(
    locale="en-US",
    show_hover=True,
    section_levels=1,
)
```

## Installation

```bash
pip install flatbread
```