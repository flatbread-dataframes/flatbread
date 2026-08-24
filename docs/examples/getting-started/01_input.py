import pandas as pd

result = (
    pd.read_json("docs/examples/sightings.json")
    .pivot_table(
        index = "species",
        columns = "region",
        values = "count",
        aggfunc = "sum",
    )
)