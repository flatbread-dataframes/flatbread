import unittest
from random import randint

import pandas as pd

from flatbread import DEFAULTS
from flatbread.testing.dataframe import make_test_df
from flatbread.transforms import chaining
from flatbread.transforms.panels import state
import flatbread.transforms.aggregation.totals as totals
import flatbread.transforms.panels.percentages as pcts
import flatbread.transforms.panels.differences as diffs


# region get_data_mask
class TestGetDataMask_SimpleIndex(unittest.TestCase):
    def setUp(self):
        self.index = pd.Index(['a', 'b', 'Totals', 'c'])

    def test_masks_matching_key(self):
        mask = chaining.get_data_mask(self.index, ['Totals'])
        self.assertEqual(list(mask), [True, True, False, True])

    def test_none_keeps_all(self):
        mask = chaining.get_data_mask(self.index, None)
        self.assertTrue(mask.all())

    def test_prefix_matching(self):
        index = pd.Index(['a', 'Subtotals x', 'b'])
        mask = chaining.get_data_mask(index, ['Subtotals'])
        self.assertEqual(list(mask), [True, False, True])


class TestGetDataMask_MultiIndex(unittest.TestCase):
    def setUp(self):
        self.index = pd.MultiIndex.from_tuples([
            ('A', 'x'),
            ('A', 'Totals'),
            ('B', 'y'),
        ])

    def test_masks_key_in_any_level(self):
        mask = chaining.get_data_mask(self.index, ['Totals'])
        self.assertEqual(list(mask), [True, False, True])


# region ignored_keys
class TestResolveIgnoredKeys(unittest.TestCase):
    def setUp(self):
        self.df = make_test_df(
            nrows=5,
            ncols=4,
            data_gen_f=lambda r, c: randint(1, 100),
        )

    def test_no_attrs_no_ignore(self):
        result = chaining.resolve_ignored_keys(self.df, 'differences')
        self.assertEqual(result, [])

    def test_user_ignore_keys_passed_through(self):
        result = chaining.resolve_ignored_keys(
            self.df, 'differences', ignore_keys='custom',
        )
        self.assertIn('custom', result)

    def test_user_ignore_keys_list(self):
        result = chaining.resolve_ignored_keys(
            self.df, 'differences', ignore_keys=['a', 'b'],
        )
        self.assertIn('a', result)
        self.assertIn('b', result)

    def test_picks_up_tracked_margin_labels(self):
        # simulate add_totals having run
        df = totals.add_totals(self.df, axis=0)
        result = chaining.resolve_ignored_keys(df, 'differences')
        totals_label = DEFAULTS['transforms']['totals']['label']
        self.assertIn(totals_label, result)

    def test_picks_up_panel_labels(self):
        df = diffs.add_differences(self.df, axis=0)
        result = chaining.resolve_ignored_keys(df, 'differences')
        # the diff panel label should be in ignore keys
        panels = chaining.get_nested_key(df.attrs, ['flatbread', 'panels'])
        diff_label = next(
            k for k, v in panels.items()
            if v['type'] == 'differences'
        )
        self.assertIn(diff_label, result)


# region panel state
class TestRegisterPanel(unittest.TestCase):
    def setUp(self):
        self.df = make_test_df(
            nrows=3,
            ncols=2,
            data_gen_f=lambda r, c: randint(1, 100),
        )

    def test_first_registration_creates_data_entry(self):
        state.register_panel(self.df, 'pct_col', 'percentages', 0)
        panels = chaining.get_nested_key(self.df.attrs, ['flatbread', 'panels'])
        self.assertIn('n', panels)
        self.assertEqual(panels['n']['type'], 'data')

    def test_registers_panel_with_metadata(self):
        state.register_panel(self.df, 'pct_col', 'percentages', 0)
        panels = chaining.get_nested_key(self.df.attrs, ['flatbread', 'panels'])
        self.assertIn('pct_col', panels)
        self.assertEqual(panels['pct_col']['type'], 'percentages')
        self.assertEqual(panels['pct_col']['axis'], 0)

    def test_second_registration_adds_panel(self):
        state.register_panel(self.df, 'pct_col', 'percentages', 0)
        state.register_panel(self.df, 'diff_col', 'differences', 0)
        panels = chaining.get_nested_key(self.df.attrs, ['flatbread', 'panels'])
        self.assertEqual(len(panels), 3)  # n, pct_col, diff_col


class TestCheckPanelState(unittest.TestCase):
    def setUp(self):
        self.df = make_test_df(
            nrows=3,
            ncols=2,
            data_gen_f=lambda r, c: randint(1, 100),
        )

    def test_no_panels_returns_full_df(self):
        result = state.check_panel_state(self.df, 'pct_col')
        pd.testing.assert_frame_equal(result, self.df)

    def test_duplicate_label_raises(self):
        paneled = diffs.add_differences(self.df, axis=0)
        panels = chaining.get_nested_key(paneled.attrs, ['flatbread', 'panels'])
        existing_label = next(
            k for k, v in panels.items() if v['type'] == 'differences'
        )
        self.assertRaises(
            ValueError,
            state.check_panel_state, paneled, existing_label,
        )

    def test_interleaved_raises(self):
        chaining.set_nested_key(
            self.df.attrs, ['flatbread', 'interleaved'], True,
        )
        self.assertRaises(
            ValueError,
            state.check_panel_state, self.df, 'pct_col',
        )

    def test_paneled_returns_data_panel(self):
        paneled = diffs.add_differences(self.df, axis=0)
        result = state.check_panel_state(paneled, 'new_panel')
        pd.testing.assert_frame_equal(result, self.df)


# region totals → pct
class TestChain_TotalsPercentages(unittest.TestCase):
    def setUp(self):
        self.df = make_test_df(
            nrows=5,
            ncols=4,
            data_gen_f=lambda r, c: randint(1, 100),
        )

    def test_totals_excluded_from_pct(self):
        result = (
            self.df
            .pipe(totals.add_totals, axis=0)
            .pipe(pcts.add_percentages, axis=0, ndigits=-1, base=1)
        )
        panels = chaining.get_nested_key(result.attrs, ['flatbread', 'panels'])
        pct_label = next(
            k for k, v in panels.items() if v['type'] == 'percentages'
        )
        # data rows (excluding totals) in pct panel should sum to 1
        pct_data = result[pct_label]
        totals_label = DEFAULTS['transforms']['totals']['label']
        data_rows = pct_data.drop(totals_label)
        self.assertAlmostEqual(data_rows.iloc[:, 0].sum(), 1, places=7)

    def test_both_panels_registered(self):
        result = (
            self.df
            .pipe(totals.add_totals, axis=0)
            .pipe(pcts.add_percentages, axis=0)
        )
        panels = chaining.get_nested_key(result.attrs, ['flatbread', 'panels'])
        types = {v['type'] for v in panels.values()}
        self.assertIn('data', types)
        self.assertIn('percentages', types)

    def test_data_panel_unchanged(self):
        with_totals = self.df.pipe(totals.add_totals, axis=0)
        result = with_totals.pipe(pcts.add_percentages, axis=0)
        panels = chaining.get_nested_key(result.attrs, ['flatbread', 'panels'])
        data_label = next(k for k, v in panels.items() if v['type'] == 'data')
        pd.testing.assert_frame_equal(result[data_label], with_totals)


# region totals → diffs
class TestChain_TotalsDifferences(unittest.TestCase):
    def setUp(self):
        self.df = make_test_df(
            nrows=5,
            ncols=4,
            data_gen_f=lambda r, c: (r + 1) * (c + 1) * 10,
        )

    def test_totals_excluded_from_diff(self):
        with_totals = self.df.pipe(totals.add_totals, axis=0)
        result = with_totals.pipe(diffs.add_differences, axis=0)
        panels = chaining.get_nested_key(result.attrs, ['flatbread', 'panels'])
        diff_label = next(
            k for k, v in panels.items() if v['type'] == 'differences'
        )
        diff_data = result[diff_label]
        totals_label = DEFAULTS['transforms']['totals']['label']
        self.assertTrue(diff_data.loc[totals_label].isna().all())

    def test_diff_values_correct(self):
        with_totals = self.df.pipe(totals.add_totals, axis=0)
        result = with_totals.pipe(diffs.add_differences, axis=0)
        panels = chaining.get_nested_key(result.attrs, ['flatbread', 'panels'])
        diff_label = next(
            k for k, v in panels.items() if v['type'] == 'differences'
        )
        totals_label = DEFAULTS['transforms']['totals']['label']
        diff_col = result[diff_label].iloc[:, 0]

        # first data row has no predecessor, margin row was excluded
        self.assertTrue(pd.isna(diff_col.loc['r0']))
        self.assertTrue(pd.isna(diff_col.loc[totals_label]))

        # remaining data rows: values are 10, 20, 30, 40, 50 → constant diff of 10
        data_diffs = diff_col.loc[['r1', 'r2', 'r3', 'r4']]
        expected = pd.Series([10.0] * 4, index=['r1', 'r2', 'r3', 'r4'])
        pd.testing.assert_series_equal(data_diffs, expected, check_names=False)


# region totals → pct → diffs
class TestChain_TotalsPercentagesDifferences(unittest.TestCase):
    def setUp(self):
        self.df = make_test_df(
            nrows=5,
            ncols=4,
            data_gen_f=lambda r, c: (r + 1) * (c + 1) * 10,
        )

    def test_full_chain_all_panels_registered(self):
        result = (
            self.df
            .pipe(totals.add_totals, axis=0)
            .pipe(pcts.add_percentages, axis=0)
            .pipe(diffs.add_differences, axis=0)
        )
        panels = chaining.get_nested_key(result.attrs, ['flatbread', 'panels'])
        types = {v['type'] for v in panels.values()}
        self.assertEqual(types, {'data', 'percentages', 'differences'})

    def test_full_chain_data_panel_has_totals(self):
        result = (
            self.df
            .pipe(totals.add_totals, axis=0)
            .pipe(pcts.add_percentages, axis=0)
            .pipe(diffs.add_differences, axis=0)
        )
        panels = chaining.get_nested_key(result.attrs, ['flatbread', 'panels'])
        data_label = next(k for k, v in panels.items() if v['type'] == 'data')
        totals_label = DEFAULTS['transforms']['totals']['label']
        self.assertIn(totals_label, result[data_label].index)

    def test_full_chain_attrs_track_margin_labels(self):
        result = (
            self.df
            .pipe(totals.add_totals, axis=0)
            .pipe(pcts.add_percentages, axis=0)
            .pipe(diffs.add_differences, axis=0)
        )
        tracked = chaining.get_nested_key(result.attrs, ['flatbread', 'labels'])
        totals_label = DEFAULTS['transforms']['totals']['label']
        self.assertIn(totals_label, tracked.get('totals', set()))


if __name__ == '__main__':
    unittest.main()
