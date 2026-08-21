import unittest

import pandas as pd

import flatbread.transforms.panels.differences as diffs
from flatbread.transforms import chaining


# region relabel
class TestRelabel(unittest.TestCase):
    def test_simple_name(self):
        s = pd.Series([1, 2], name='val')
        result = diffs.relabel(s, 'diff')
        self.assertEqual(result.name, ('val', 'diff'))

    def test_tuple_name(self):
        s = pd.Series([1, 2], name=('a', 'b'))
        result = diffs.relabel(s, 'diff')
        self.assertEqual(result.name, ('a', 'b', 'diff'))

    def test_none_name(self):
        s = pd.Series([1, 2])
        result = diffs.relabel(s, 'diff')
        self.assertEqual(result.name, (None, 'diff'))


# region pairwise_labels
class TestPairwiseLabels_Index(unittest.TestCase):
    def setUp(self):
        self.index = pd.Index(['a', 'b', 'c', 'd'], name='col')

    def test_periods_1(self):
        result = diffs.pairwise_labels(self.index, periods=1)
        expected = pd.Index(['a-b', 'b-c', 'c-d'], name='col')
        pd.testing.assert_index_equal(result, expected)

    def test_periods_2(self):
        result = diffs.pairwise_labels(self.index, periods=2)
        expected = pd.Index(['a-c', 'b-d'], name='col')
        pd.testing.assert_index_equal(result, expected)

    def test_negative_periods(self):
        result = diffs.pairwise_labels(self.index, periods=-1)
        expected = pd.Index(['b-a', 'c-b', 'd-c'], name='col')
        pd.testing.assert_index_equal(result, expected)


class TestPairwiseLabels_MultiIndex(unittest.TestCase):
    def setUp(self):
        self.index = pd.MultiIndex.from_tuples(
            [('X', 'a'), ('X', 'b'), ('X', 'c'),
             ('Y', 'a'), ('Y', 'b'), ('Y', 'c')],
            names=['grp', 'item'],
        )

    def test_pairs_within_groups(self):
        result = diffs.pairwise_labels(self.index, periods=1)
        expected = pd.MultiIndex.from_tuples(
            [('X', 'a-b'), ('X', 'b-c'),
             ('Y', 'a-b'), ('Y', 'b-c')],
            names=['grp', 'item'],
        )
        pd.testing.assert_index_equal(result, expected)


# region as_differences Series
class TestAsDifferences_Series(unittest.TestCase):
    def setUp(self):
        self.s = pd.Series([10, 20, 50], index=['a', 'b', 'c'], name='val')

    def test_diff_values(self):
        result = diffs.as_differences(self.s, method='diff')
        expected_values = [float('nan'), 10.0, 30.0]
        pd.testing.assert_series_equal(
            result, pd.Series(
                expected_values,
                index = ['a', 'b', 'c'],
                name = ('val', 'diff'),
            ),
        )

    def test_pct_change_values(self):
        result = diffs.as_differences(self.s, method='pct_change')
        expected_values = [float('nan'), 1.0, 1.5]
        pd.testing.assert_series_equal(
            result, pd.Series(
                expected_values,
                index = ['a', 'b', 'c'],
                name = ('val', 'pct_change'),
            ),
            check_names=False,
        )

    def test_name_is_relabeled(self):
        result = diffs.as_differences(self.s, label_diff='delta')
        self.assertEqual(result.name, ('val', 'delta'))


# region as_differences DataFrame
class TestAsDifferences_DataFrame(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {'x': [10, 20, 50], 'y': [100, 200, 500]},
            index=['a', 'b', 'c'],
        )

    def test_axis0_values(self):
        result = diffs.as_differences(self.df, axis=0, method='diff')
        # first row dropped (all NaN)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.loc['b', 'x'], 10)
        self.assertEqual(result.loc['c', 'x'], 30)

    def test_axis1_pairwise_columns(self):
        result = diffs.as_differences(self.df, axis=1, method='diff')
        # columns become pairwise labels
        self.assertIn('x-y', result.columns)

    def test_axis1_values(self):
        result = diffs.as_differences(self.df, axis=1, method='diff')
        # y - x = 90, 180, 450
        expected = [90, 180, 450]
        self.assertEqual(list(result['x-y']), expected)


# region add_differences Series
class TestAddDifferences_Series(unittest.TestCase):
    def setUp(self):
        self.s = pd.Series([10, 20, 50], index=['a', 'b', 'c'], name='val')

    def test_output_is_dataframe(self):
        result = diffs.add_differences(self.s)
        self.assertIsInstance(result, pd.DataFrame)

    def test_has_two_columns(self):
        result = diffs.add_differences(self.s)
        self.assertEqual(len(result.columns), 2)

    def test_panel_registered_in_attrs(self):
        result = diffs.add_differences(self.s)
        panels = chaining.get_nested_key(result.attrs, ['flatbread', 'panels'])
        self.assertIsNotNone(panels)
        self.assertIn('n', panels)


# region add_differences DataFrame
class TestAddDifferences_DataFrame(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {'x': [10, 20, 50], 'y': [100, 200, 500]},
            index=['a', 'b', 'c'],
        )

    def test_adds_panel_level_to_columns(self):
        result = diffs.add_differences(self.df, axis=0)
        self.assertIsInstance(result.columns, pd.MultiIndex)

    def test_panel_registered(self):
        result = diffs.add_differences(self.df, axis=0)
        panels = chaining.get_nested_key(result.attrs, ['flatbread', 'panels'])
        self.assertIn('n', panels)
        # default label 'diff' gets axis suffix → 'diff_col' for axis=0
        diff_keys = [k for k in panels if k != 'n']
        self.assertTrue(len(diff_keys) == 1)

    def test_data_panel_matches_original(self):
        result = diffs.add_differences(self.df, axis=0)
        panels = chaining.get_nested_key(result.attrs, ['flatbread', 'panels'])
        data_label = next(k for k, v in panels.items() if v['type'] == 'data')
        pd.testing.assert_frame_equal(result[data_label], self.df)

    def test_custom_label_no_suffix(self):
        result = diffs.add_differences(self.df, axis=0, label_diff='my_diff')
        panels = chaining.get_nested_key(result.attrs, ['flatbread', 'panels'])
        self.assertIn('my_diff', panels)


# region add_pct_change DataFrame
class TestAddPctChange_DataFrame(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {'x': [10, 20, 50], 'y': [100, 200, 500]},
            index=['a', 'b', 'c'],
        )

    def test_panel_type_is_pct_change(self):
        result = diffs.add_pct_change(self.df, axis=0)
        panels = chaining.get_nested_key(result.attrs, ['flatbread', 'panels'])
        pct_panels = [k for k, v in panels.items() if v['type'] == 'pct_change']
        self.assertEqual(len(pct_panels), 1)

    def test_values_are_ratios(self):
        result = diffs.add_pct_change(self.df, axis=0)
        panels = chaining.get_nested_key(result.attrs, ['flatbread', 'panels'])
        pct_label = next(k for k, v in panels.items() if v['type'] == 'pct_change')
        pct_data = result[pct_label]
        # 20/10 - 1 = 1.0, 50/20 - 1 = 1.5
        self.assertAlmostEqual(pct_data.loc['b', 'x'], 1.0)
        self.assertAlmostEqual(pct_data.loc['c', 'x'], 1.5)


if __name__ == '__main__':
    unittest.main()