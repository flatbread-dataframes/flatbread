from functools import wraps
from typing import Any, Callable, TypeVar

import pandas as pd

from flatbread import axes
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


def handle_axis_rotation(
    func: Callable[..., pd.DataFrame]
) -> Callable[..., pd.DataFrame]:
    """Decorator that handles axis=1 by transposing before and after the operation."""

    @wraps(func)
    def wrapper(df: pd.DataFrame, *args, axis: Axis = 0, **kwargs) -> pd.DataFrame:
        resolved = axes.resolve_axis(axis)
        if resolved == 2:
            raise ValueError("axis='both' not supported for this operation")
        transpose = resolved == 1
        if transpose:
            df = df.T
        result = func(df, *args, axis=axis, **kwargs)
        return result.T if transpose else result

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
def align_dates_by_year(
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
