import pandas as pd
import flatbread

result = (
    pd.read_json("docs/examples/sightings.json")
    .pivot_table(
        index = "region",
        columns = "season",
        values = "count",
        aggfunc = "sum",
    )
    .pita.add_totals()
    .pita.add_pct_change(axis=1)
)