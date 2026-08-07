from typing import Literal, TypeAlias, TypeVar
import pandas as pd


PandasObj = TypeVar('PandasObj', pd.DataFrame, pd.Series)
Axis: TypeAlias = Literal[0, 1, 2, 'index', 'columns', 'rows', 'both'] | None
Level: TypeAlias = int | str
