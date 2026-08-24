import pandas as pd
import flatbread

df = pd.DataFrame({
    "budget":  [12500, 8300, 15200],
    "spent":   [11800, 9100, 14600],
    "area_ha": [1200, 3400, 850],
}, index=pd.Index(["Coast", "Forest", "Wetland"], name="region"))

result = (
    df.pita.add_totals(axis=0)
    .pita.format_columns({
        "budget": "currency_eur",
        "spent":  "currency_eur",
        "area_ha": {
            "style": "unit",
            "unit": "hectare",
            "unitDisplay": "short",
        },
    })
)