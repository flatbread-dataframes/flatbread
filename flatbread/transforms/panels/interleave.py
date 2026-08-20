import pandas as pd

from flatbread.transforms import chaining
from flatbread.transforms.panels import state
import flatbread.tooling as tooling


def _is_asymmetric(panel_meta: dict) -> bool:
    asymmetric = ('differences', 'pct_change')
    return panel_meta['type'] in asymmetric and panel_meta['axis'] == 1


def validate(data: pd.DataFrame) -> str:
    """
    Validate that a DataFrame can be interleaved and classify the layout.

    Parameters
    ----------
    data : pd.DataFrame
        Paneled DataFrame to validate.

    Returns
    -------
    str
        ``'symmetric'`` if all panels share the data's column structure,
        ``'asymmetric'`` if any panel has reduced columns (diff axis=1).

    Raises
    ------
    ValueError
        If already interleaved, no panels exist, or symmetric and
        asymmetric panels are mixed.
    """
    if chaining.get_nested_key(data.attrs, ['flatbread', 'interleaved']):
        raise ValueError("DataFrame is already interleaved.")

    panels = chaining.get_nested_key(data.attrs, ['flatbread', 'panels'])
    if panels is None:
        raise ValueError("No panels to interleave.")

    non_data = {k: v for k, v in panels.items() if v['type'] != 'data'}
    if not non_data:
        raise ValueError("No panels to interleave.")

    asymmetric = [k for k, v in non_data.items() if _is_asymmetric(v)]
    symmetric = [k for k, v in non_data.items() if not _is_asymmetric(v)]

    if asymmetric and symmetric:
        raise ValueError(
            "Cannot interleave symmetric panels "
            f"({', '.join(symmetric)}) with asymmetric panels "
            f"({', '.join(asymmetric)})."
        )

    return 'asymmetric' if asymmetric else 'symmetric'


def interleave(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interleave panel columns with data columns.

    Validates panel state, classifies the layout as symmetric or
    asymmetric, and reorders columns so each data column is grouped
    with its panel counterparts.

    Parameters
    ----------
    df : pd.DataFrame
        Paneled DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with interleaved columns, marked as interleaved
        in attrs.

    Raises
    ------
    ValueError
        If the DataFrame cannot be interleaved.
    """
    layout = validate(df)

    panels = chaining.get_nested_key(df.attrs, ['flatbread', 'panels']) or {}
    data_label = next(k for k, v in panels.items() if v['type'] == 'data')
    reference = df[data_label]

    if layout == 'symmetric':
        new_order = list(range(1, df.columns.nlevels)) + [0]
        nlevels = None
    else:
        new_order = list(range(1, df.columns.nlevels))
        new_order.insert(-1, 0)
        nlevels = reference.columns.nlevels - 1

    result = (
        df
        .reorder_levels(new_order, axis=1)
        .pipe(tooling.reindex_by_levels, reference, axis=1, nlevels=nlevels)
    )
    chaining.set_nested_key(result.attrs, ['flatbread', 'interleaved'], True)
    return result
