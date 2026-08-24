import pandas as pd
import flatbread

result = (
    pd.read_json("docs/examples/sightings.json")
    .pivot_table(
        index = ["class", "species"],
        columns = "region",
        values = "count",
        aggfunc = "sum",
    )
    .pita.add_agg("mean", label="Mean")
)