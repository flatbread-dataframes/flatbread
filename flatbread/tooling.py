from functools import wraps
from typing import Any, Callable, TypeVar

import pandas as pd

from flatbread.types import Axis, Level


T = TypeVar('T', pd.Series, pd.DataFrame)


def handle_series_as_dataframe(func: Callable[..., pd.DataFrame]) -> Callable[..., T]:
    """
    Decorator that converts Series to DataFrame, runs the function, then converts back.
    """
    @wraps(func)
    def wrapper(data: pd.DataFrame|pd.Series, *args: Any, **kwargs: Any) -> T:
        is_series = isinstance(data, pd.Series)
        if is_series:
            data = data.to_frame()

        result = func(data, *args, **kwargs)

        if is_series:
            result = result.iloc[:, 0]

        return result # type: ignore
    return wrapper


def handle_axis_rotation(func) -> Callable:
    """
    Decorator that handles axis=1 by transposing before and after the operation.
    """
    @wraps(func)
    def wrapper(df, *args, **kwargs):
        axis = kwargs.pop('axis', 0)
        if axis in [1, 'columns']:
            df = df.T
            result = func(df, *args, **kwargs)
            return result.T
        return func(df, *args, **kwargs)
    return wrapper


def inject_defaults(defaults: dict) -> Callable:
    """
    Load defaults if keywords are None or undefined when calling a function.

    Arguments
    ---------
    defaults (dict):
        Dictionary of keywords and default values.

    Return
    ------
    func:
        Function that will load defaults.

    Notes
    -----
    This decorator will override any default values set in the function definition.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for key, val in defaults.items():
                if kwargs.get(key) is None:
                    kwargs[key] = val
            return func(*args, **kwargs)
        return wrapper
    return decorator


# region offset date
def offset_date_field(
    df: pd.DataFrame,
    date_field: str,
    year_field: str,
) -> pd.DataFrame:
    offset_year = df[year_field].max()

    def shift_dates(group):
        offset = pd.DateOffset(years = offset_year - group.name)
        return group.shift(freq = offset)

    return (
        df
        .set_index(date_field, drop=False)
        .groupby(year_field, group_keys=False)
        .apply(shift_dates)
        .rename_axis(date_field + '_offs')
        .reset_index()
    )


# region sort index
def _sort_index_from_list(
    df: pd.DataFrame,
    order: list|pd.CategoricalDtype,
    axis: Axis = 0,
    level: Level|None = None,
) -> pd.DataFrame:
    index = df.index if axis in [0, 'index'] else df.columns
    if isinstance(index, pd.MultiIndex):
        index = index.levels[level]
    order = [i for i in order if i in index]
    return df.reindex(order, axis=axis, level=level)


def sort_index_from_list(
    data: pd.DataFrame|pd.Series,
    order: list,
    axis: Axis = 0,
    level: int|str|None = None,
) -> pd.DataFrame|pd.Series:
    sorter = lambda idx: idx.map({n:m for m,n in enumerate(order)})
    return data.sort_index(axis=axis, level=level, key=sorter)


def reindex_by_levels(
    df_target: pd.DataFrame,
    df_reference: pd.DataFrame,
    nlevels: int | None = None,
) -> pd.DataFrame:
   """
   Reindex df_target columns according to the level ordering in df_reference.

   Parameters
   ----------
   df_target : pd.DataFrame
       DataFrame to reindex, with n+k levels in columns
   df_reference : pd.DataFrame
       Reference DataFrame with n levels in columns that define the ordering
   nlevels : int or None, optional
       Number of levels to reindex. If None, reindex all levels in df_reference.
       If int, reindex only the first `levels` levels from df_reference.

   Returns
   -------
   pd.DataFrame
       df_target reindexed with specified column levels ordered as in df_reference

   Notes
   -----
   Any additional levels in df_target beyond those reindexed are left unchanged.
   """
   df_reindexed = df_target.copy()
   max_levels = df_reference.columns.nlevels if nlevels is None else nlevels

   for level in range(max_levels):
       df_reindexed = df_reindexed.reindex(
           columns=df_reference.columns.get_level_values(level).unique(),
           level=level
       )
   return df_reindexed
