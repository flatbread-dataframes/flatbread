import pandas as pd
import flatbread

df = pd.DataFrame(
    {"A": [1, 2, 3], "B": [4, 5, 6]},
    index=pd.Index(["x", "y", "z"], name="row"),
)

examples = {
    "totals_basic": df.pita.add_totals(),
}
