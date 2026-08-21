import unittest

import pandas as pd

import flatbread.axes as axes


# region resolve_axis
class TestResolveAxis(unittest.TestCase):
    def test_int_0(self):
        self.assertEqual(axes.resolve_axis(0), 0)

    def test_int_1(self):
        self.assertEqual(axes.resolve_axis(1), 1)

    def test_int_2(self):
        self.assertEqual(axes.resolve_axis(2), 2)

    def test_string_index(self):
        self.assertEqual(axes.resolve_axis('index'), 0)

    def test_string_rows(self):
        self.assertEqual(axes.resolve_axis('rows'), 0)

    def test_string_columns(self):
        self.assertEqual(axes.resolve_axis('columns'), 1)

    def test_string_both(self):
        self.assertEqual(axes.resolve_axis('both'), 2)

    def test_none(self):
        self.assertEqual(axes.resolve_axis(None), 0)

    def test_invalid_raises(self):
        self.assertRaises(ValueError, axes.resolve_axis, 'squid')


# region resolve_level
class TestResolveLevel(unittest.TestCase):
    def setUp(self):
        self.index = pd.MultiIndex.from_tuples(
            [('A', 'x'), ('A', 'y'), ('B', 'x')],
            names=['L0', 'L1'],
        )

    def test_positive_int(self):
        self.assertEqual(axes.resolve_level(self.index, 0), 0)

    def test_positive_int_last(self):
        self.assertEqual(axes.resolve_level(self.index, 1), 1)

    def test_negative_int(self):
        self.assertEqual(axes.resolve_level(self.index, -1), 1)

    def test_negative_int_first(self):
        self.assertEqual(axes.resolve_level(self.index, -2), 0)

    def test_by_name(self):
        self.assertEqual(axes.resolve_level(self.index, 'L0'), 0)

    def test_by_name_last(self):
        self.assertEqual(axes.resolve_level(self.index, 'L1'), 1)

    def test_out_of_range_raises(self):
        self.assertRaises(IndexError, axes.resolve_level, self.index, 3)

    def test_negative_out_of_range_raises(self):
        self.assertRaises(IndexError, axes.resolve_level, self.index, -3)

    def test_nonexistent_name_raises(self):
        self.assertRaises(ValueError, axes.resolve_level, self.index, 'squid')


# region add_value_to_key
class TestAddValueToKey(unittest.TestCase):
    def test_insert_at_start(self):
        result = axes.add_value_to_key(('a', 'b'), 'x', level=0)
        self.assertEqual(result, ('x', 'a', 'b'))

    def test_insert_at_middle(self):
        result = axes.add_value_to_key(('a', 'b'), 'x', level=1)
        self.assertEqual(result, ('a', 'x', 'b'))

    def test_insert_at_end_negative(self):
        result = axes.add_value_to_key(('a', 'b'), 'x', level=-1)
        self.assertEqual(result, ('a', 'b', 'x'))

    def test_single_value_becomes_tuple(self):
        result = axes.add_value_to_key('a', 'x', level=0)
        self.assertEqual(result, ('x', 'a'))

    def test_negative_level_minus_2(self):
        result = axes.add_value_to_key(('a', 'b', 'c'), 'x', level=-2)
        self.assertEqual(result, ('a', 'b', 'x', 'c'))


# region add_level
class TestAddLevel_DataFrame(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {'A': [1, 2], 'B': [3, 4]},
            index=pd.Index(['r0', 'r1'], name='row'),
        )

    def test_add_to_index_single_value(self):
        result = axes.add_level(self.df, 'group', level=0, level_name='grp')
        self.assertIsInstance(result.index, pd.MultiIndex)
        self.assertEqual(result.index.nlevels, 2)
        self.assertEqual(result.index.names, ['grp', 'row'])
        self.assertTrue(all(result.index.get_level_values(0) == 'group'))

    def test_add_to_index_list_of_values(self):
        result = axes.add_level(self.df, ['g1', 'g2'], level=0, level_name='grp')
        values = list(result.index.get_level_values(0))
        self.assertEqual(values, ['g1', 'g2'])

    def test_add_to_columns(self):
        result = axes.add_level(self.df, 'top', axis=1, level=0, level_name='panel')
        self.assertIsInstance(result.columns, pd.MultiIndex)
        self.assertEqual(result.columns.nlevels, 2)
        self.assertTrue(all(result.columns.get_level_values(0) == 'top'))

    def test_list_length_mismatch_raises(self):
        self.assertRaises(
            ValueError,
            axes.add_level, self.df, ['a', 'b', 'c'], level=0,
        )

    def test_does_not_mutate_original(self):
        original_index = self.df.index.copy()
        axes.add_level(self.df, 'group', level=0)
        pd.testing.assert_index_equal(self.df.index, original_index)


class TestAddLevel_Series(unittest.TestCase):
    def setUp(self):
        self.s = pd.Series([1, 2, 3], index=pd.Index(['a', 'b', 'c'], name='idx'))

    def test_add_single_value(self):
        result = axes.add_level(self.s, 'group', level=0, level_name='grp')
        self.assertIsInstance(result.index, pd.MultiIndex)
        self.assertEqual(result.index.nlevels, 2)
        self.assertEqual(result.index.names, ['grp', 'idx'])

    def test_does_not_mutate_original(self):
        original_index = self.s.index.copy()
        axes.add_level(self.s, 'group', level=0)
        pd.testing.assert_index_equal(self.s.index, original_index)


# region sort_aggregates
class TestSortAggregates(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {'val': [1, 2, 3, 4]},
            index=['Item1', 'Totals', 'Item2', 'Item3'],
        )

    def test_aggregates_last(self):
        result = axes.sort_aggregates(self.df, labels=['Totals'])
        expected_order = ['Item1', 'Item2', 'Item3', 'Totals']
        self.assertEqual(list(result.index), expected_order)

    def test_aggregates_first(self):
        result = axes.sort_aggregates(
            self.df, labels=['Totals'], aggregates_last=False,
        )
        self.assertEqual(result.index[0], 'Totals')

    def test_multiple_labels(self):
        df = pd.DataFrame(
            {'val': [1, 2, 3, 4, 5]},
            index=['Subtotals', 'Item1', 'Totals', 'Item2', 'Item3'],
        )
        result = axes.sort_aggregates(df, labels=['Totals', 'Subtotals'])
        non_agg = list(result.index[:-2])
        self.assertNotIn('Totals', non_agg)
        self.assertNotIn('Subtotals', non_agg)

    def test_no_labels_is_noop(self):
        result = axes.sort_aggregates(self.df, labels=None)
        self.assertEqual(list(result.index), list(self.df.index))


if __name__ == '__main__':
    unittest.main()