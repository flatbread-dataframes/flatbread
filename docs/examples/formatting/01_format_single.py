import pandas as pd
import flatbread

df = pd.DataFrame({
    "budget":    [12500, 8300, 15200],
    "spent":     [11800, 9100, 14600],
    "sightings": [90, 130, 80],
}, index=pd.Index(["Coast", "Forest", "Wetland"], name="region"))

result = (
    df.pita
    .format("budget", "currency_eur")
    .format("spent", "currency_eur")
)