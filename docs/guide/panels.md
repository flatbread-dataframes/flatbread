# Panels

Panels add derived columns alongside the original data — percentages,
differences, or percentage change. Each panel adds a level to the column
index to distinguish it from the source data.

## `as_*` vs `add_*`

Flatbread offers two forms for each transform:

- `as_percentages()` — replaces the data with percentages
- `add_percentages()` — keeps the original data and adds percentages as a
  separate panel

The same pattern applies to differences (`as_differences` / `add_differences`).
Use the `as_*` form when you only need the derived values. Use the `add_*`
form when you want both side by side.

## Percentages

### Replacing data with percentages

`as_percentages` transforms the values in place. The original counts are gone:

```python
--8<-- "examples/panels/01_as_percentages.py"
```

<flatbread-table src="/assets/examples/panels/01_as_percentages.json" margin-labels="Totals" hide-settings-menu></flatbread-table>

### Adding a percentage panel

`add_percentages` keeps the counts and adds percentages alongside them:

```python
--8<-- "examples/panels/02_add_percentages.py"
```

<flatbread-table src="/assets/examples/panels/02_add_percentages.json" margin-labels="Totals" hide-settings-menu></flatbread-table>

### The `axis` parameter

The `axis` parameter controls what the percentages are relative to:

- `0` — column totals (each column sums to 100%)
- `1` — row totals (each row sums to 100%)
- `2` — grand total (the default; entire table sums to 100%)

### Rounding

By default, percentages are not rounded (`ndigits=-1`). Set `ndigits` to
control decimal places. When rounding, flatbread uses apportioned rounding
by default — this ensures the rounded percentages still sum to the base
value, avoiding the common issue where rounded percentages add up to 99%
or 101%.

The default `base=1` produces proportions (0–1). The flatbread-table
viewer automatically formats these as percentages when it detects the
panel label. Use `base=100` if you need literal percentage values
(e.g. for export or further calculation).

## Differences

`add_differences` computes the difference between consecutive values
along an axis. This works well with time-based data — here we pivot
sightings by region and season:

```python
--8<-- "examples/panels/04_differences.py"
```

<flatbread-table src="/assets/examples/panels/04_differences.json" margin-labels="Totals" hide-settings-menu></flatbread-table>

The `periods` parameter controls how many steps to look back (default 1).

## Percentage change

`add_pct_change` works like `add_differences` but computes the relative
change instead of the absolute difference:

```python
--8<-- "examples/panels/05_pct_change.py"
```

<flatbread-table src="/assets/examples/panels/05_pct_change.json" margin-labels="Totals" hide-settings-menu></flatbread-table>

Percentages and differences have aliases: `add_pct`, `as_pct`,
`add_diffs`, `as_diffs`.

## Combining panels

Multiple panels can be added to the same DataFrame. Flatbread tracks
which panels exist and excludes them from subsequent calculations:

```python
df.pita.add_totals().pita.add_percentages().pita.add_differences()
```

## Interleaving

By default, panels are grouped: all data columns first, then all
panel columns. Set `interleaf=True` to place panel columns next to
their corresponding data columns:

```python
--8<-- "examples/panels/03_interleaved.py"
```

<flatbread-table src="/assets/examples/panels/03_interleaved.json" margin-labels="Totals" hide-settings-menu></flatbread-table>

### Symmetric vs group interleaving

How interleaving works depends on the panel shape. Percentages and
differences along axis=0 produce a panel with the same columns as the
data — one panel column per data column. Interleaving pairs them
one-to-one:

```python
df.pita.add_totals().pita.add_percentages(interleaf=True)
# → Spring | Spring pct | Summer | Summer pct | ...
```

Differences and percentage change along axis=1 produce fewer columns
than the data, since each value compares two adjacent columns. These
panels can only be interleaved when the original data has more than
one column level (e.g. region and season). In that case, interleaving
places the data and diff columns together within each group rather
than pairing them one-to-one:

```python
# columns: (region, season)
df.pita.add_differences(axis=1, interleaf=True)
# Coast:  Spring | Summer | Autumn | Spring-Summer | Summer-Autumn
# Forest: Spring | Summer | Autumn | Spring-Summer | Summer-Autumn
```

Panels with the same column structure can be interleaved together,
but the two shapes cannot be mixed. Call `interleave()` at the end
of the chain, after all panels have been added.