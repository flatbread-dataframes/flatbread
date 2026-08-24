# Flatbread

Pandas extension for aggregation and tabular display. Access it through
the `pita` accessor on DataFrames and Series.

**[Documentation](https://flatbread-dataframes.github.io/flatbread/)**

## Installation

```bash
pip install flatbread
```

## Quick example

```python
import pandas as pd
import flatbread

result = (
    df
    .pita.add_totals()
    .pita.add_subtotals(level=0)
    .pita.add_percentages(interleaf=True)
)
```

## Features

- Totals and subtotals (rows and columns)
- Percentages, differences, percentage change
- Custom aggregations
- Layered configuration (defaults, user, project)
- Interactive table display for Jupyter notebooks