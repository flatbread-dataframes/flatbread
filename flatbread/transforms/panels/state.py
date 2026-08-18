import pandas as pd

from flatbread.types import Axis
from flatbread.config import DEFAULTS
from flatbread.transforms import chaining


def register_panel(
    data: pd.DataFrame,
    label: str,
    panel_type: str,
    axis: Axis,
) -> None:
    """
    Register a panel in the DataFrame's attrs metadata.

    On first call, initializes the panels dict with an ``n`` entry for
    the data panel. Raises if a panel with the given label already exists.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame whose attrs will be mutated.
    label : str
        Label for the new panel (e.g. 'pct_col', 'diff_row').
    panel_type : str
        Type of transform that produced the panel (e.g. 'percentages', 'differences').
    axis : Axis
        Axis along which the transform was applied.
    """
    if chaining.get_nested_key(data.attrs, ['flatbread', 'panels']) is None:
        chaining.set_nested_key(
            data.attrs,
            keys = ['flatbread', 'panels', 'n'],
            value = {'type': 'data'},
        )
    chaining.set_nested_key(
        data.attrs,
        keys = ['flatbread', 'panels', label],
        value = {'type': panel_type, 'axis': axis},
    )


def check_panel_state(
    data: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """
    Validate panel state and return the source data to operate on.

    If the DataFrame is already paneled, returns the data panel.
    Otherwise returns the full DataFrame. Raises if the label
    already exists or the DataFrame is interleaved.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame, possibly already paneled.
    label : str
        Label for the panel to be added.

    Returns
    -------
    pd.DataFrame
        The source data to transform.

    Raises
    ------
    ValueError
        If ``label`` already exists or DataFrame is interleaved.
    """
    if chaining.get_nested_key(data.attrs, ['flatbread', 'interleaved']):
        raise ValueError(
            "Cannot add panel to an interleaved DataFrame. "
            "Call interleave at the end of the chain."
        )

    panels = chaining.get_nested_key(data.attrs, ['flatbread', 'panels'])
    if panels is None:
        return data

    if label in panels:
        raise ValueError(f"Panel '{label}' already exists.")

    data_label = next(k for k, v in panels.items() if v['type'] == 'data')
    return data[data_label]


def resolve_panel_label(label: str, axis: Axis) -> str:
    suffix = DEFAULTS['panels']['axis_suffixes'][str(axis)]
    return f"{label}_{suffix}"