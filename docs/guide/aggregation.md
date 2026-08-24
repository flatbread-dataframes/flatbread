# Aggregation

Flatbread provides several methods for adding aggregate rows and columns
to a table. All methods leave the original data unmodified and return
a new DataFrame.

## Subtotals

When your table has a MultiIndex, `add_subtotals` inserts aggregate rows
at a specified level. Here we pivot sightings by class and species, then
add subtotals per class and an overall total:

```python
--8<-- "examples/aggregation/01_subtotals.py"
```

<flatbread-table src="/assets/examples/aggregation/01_subtotals.json" margin-labels="Subtotals;Totals" hide-settings-menu></flatbread-table>

The `level` parameter controls which index level to subtotal by. Level 0
is the outermost level. You can pass multiple levels as a list:

```python
df.pita.add_subtotals(level=[0, 1])
```

Set `include_level_name=True` to append the group value to the subtotal
label (e.g. "Subtotals Bird" instead of "Subtotals").

By default, groups with only one row are skipped. Set
`skip_single_rows=False` to add subtotals for those too.

## Custom aggregations

`add_agg` works like `add_totals` but with any aggregation function — a
string like `'mean'`, `'median'`, `'max'`, or a callable:

```python
--8<-- "examples/aggregation/02_add_agg.py"
```

<flatbread-table src="/assets/examples/aggregation/02_add_agg.json" margin-labels="Mean" hide-settings-menu></flatbread-table>

Its counterpart `add_subagg` is to `add_agg` what `add_subtotals` is to
`add_totals` — it applies the aggregation at a specified MultiIndex level.

## The `axis` parameter

The `axis` parameter appears on most aggregation methods:

- `0` — aggregate along rows (adds a row)
- `1` — aggregate along columns (adds a column)
- `2` — both (the default for `add_totals`)

`add_subtotals` defaults to `0`. `add_totals` defaults to `2`.

## Sorting

Totals and subtotals are placed at the end of their group by default.
`sort_totals` lets you reposition them — for example, placing them first:

```python
--8<-- "examples/aggregation/03_sort_totals.py"
```

<flatbread-table src="/assets/examples/aggregation/03_sort_totals.json" margin-labels="Subtotals;Totals" hide-settings-menu></flatbread-table>

`sort_totals` uses the labels tracked in `df.attrs` by flatbread's
operations. If those attrs were lost (some pandas operations strip them),
pass the labels explicitly:

```python
df.pita.sort_totals(labels=["Totals", "Subtotals"])
```

## Dropping totals

`drop_totals` removes all aggregate rows that flatbread has tracked:

```python
df.pita.add_totals().pita.drop_totals()
```

This uses the labels stored in `df.attrs` by flatbread's operations, so
it only removes rows that flatbread added.

## Chaining

Operations can be chained freely. Flatbread tracks which labels it has
added (totals, subtotals) so that subsequent operations automatically
exclude them from calculations. This prevents double-counting:

```python
--8<-- "examples/aggregation/04_chaining.py"
```

<flatbread-table src="/assets/examples/aggregation/04_chaining.json" margin-labels="Subtotals;Totals" hide-settings-menu></flatbread-table>

Here `add_percentages` knows to exclude the Subtotals and Totals rows
from its calculation because `add_subtotals` and `add_totals` registered
those labels. This is also what `ignore_keys` controls — you can pass
additional labels to exclude manually when needed.