import unittest
from random import randint

import pandas as pd

from flatbread import DEFAULTS
import flatbread.transforms.totals as totals
from flatbread.testing.dataframe import make_test_df


# region simple
class TestTotalsAdd_DataFrameSimple(unittest.TestCase):
    def setUp(self):
        self.totals_label = DEFAULTS['totals']['label']
        self.df = make_test_df(
            nrows=5,
            ncols=4,
            data_gen_f=lambda r, c: randint(1, 100),
        )

    def test_add_column_total_to_rows(self):
        s = totals.add_totals(self.df, axis=0).loc[self.totals_label]
        self.assertTrue(s.equals(self.df.sum()))

    def test_add_row_total_to_cols(self):
        s = totals.add_totals(self.df, axis=1).loc[:, self.totals_label]
        self.assertTrue(s.equals(self.df.sum(axis=1)))

    def test_add_by_axis_name(self):
        s = totals.add_totals(self.df, axis='columns').loc[:, self.totals_label]
        self.assertTrue(s.equals(self.df.sum(axis=1)))

    def test_add_both(self):
        rows, cols = self.totals_label, self.totals_label
        val = totals.add_totals(self.df, axis=2).loc[rows, cols]
        self.assertTrue(val == self.df.sum().sum())

    def test_preserve_axis_names(self):
        r1 = totals.add_totals(self.df)
        r2 = totals.add_totals(self.df, axis=1)
        self.assertEqual(self.df.index.names, r1.index.names)
        self.assertEqual(self.df.index.names, r2.index.names)

    def test_custom_totals_label(self):
        custom_label = "Totes"
        result = totals.add_totals(self.df, label=custom_label)
        self.assertTrue(custom_label in result.index)

    def test_ignore_keys(self):
        label = "To be ignored"
        to_be_ignored = pd.DataFrame(
            [[999, 999, 999, 999]],
            index=[label],
            columns=self.df.columns
        )
        df_with_ignored_row = pd.concat([self.df, to_be_ignored])

        result = totals.add_totals(df_with_ignored_row, ignore_keys=[label])
        expected_total = self.df.sum().iloc[0]
        actual_total = result.loc[self.totals_label].iloc[0]
        self.assertEqual(actual_total, expected_total)


# region categorical
class TestTotalsAdd_DataFrameCategorical(unittest.TestCase):
    def setUp(self):
        self.totals_label = DEFAULTS['totals']['label']
        df = make_test_df(nrows=5, ncols=4)
        df.columns = pd.Categorical(df.columns)
        df.index = pd.Categorical(df.index)
        self.df = df

    def test_add_column_total_to_rows(self):
        s = totals.add_totals(self.df, axis=0).loc[self.totals_label]
        self.assertTrue(s.equals(self.df.sum()))

    def test_add_row_total_to_cols(self):
        s = totals.add_totals(self.df, axis=1).loc[:, self.totals_label]
        self.assertTrue(s.equals(self.df.sum(axis=1)))

    def test_add_both(self):
        v = totals.add_totals(self.df, axis=2).loc[
            self.totals_label, self.totals_label
        ]
        self.assertTrue(v == self.df.sum().sum())


# region multiindex
class TestTotalsAdd_DataFrameMultiIndex(unittest.TestCase):
    def setUp(self):
        self.subtotals_label = DEFAULTS['subtotals']['label']
        self.fill = ''
        self.df = make_test_df(
            nrows=7,
            ncols=4,
            data_gen_f=lambda r, c: randint(1, 100),
            col_levels=2,
            idx_levels=3,
            col_dupes=[2, 1],
            idx_dupes=[4, 2, 1],
        )

    def test_add_column_total_to_rows(self):
        s = totals.add_totals(self.df, axis=0).iloc[-1]
        self.assertTrue(s.equals(self.df.sum()))

    def test_add_row_total_to_cols(self):
        s = totals.add_totals(self.df, axis=1).iloc[:, -1]
        self.assertTrue(s.equals(self.df.sum(axis=1)))

    def test_add_both(self):
        value = totals.add_totals(self.df, axis=2).iloc[-1, -1]
        self.assertTrue(value == self.df.sum().sum())

    def test_add_rows_within(self):
        left = (
            totals
            .add_subtotals(self.df, level=1, skip_single_rows=False)
            .xs(self.subtotals_label, level=2)
        )
        right = self.df.groupby(level=1).sum()
        comparison = left.eq(right).all(axis=None)
        self.assertTrue(comparison)

    def test_add_cols_within(self):
        left = (
            totals
            .add_subtotals(self.df, axis=1)
            .xs(self.subtotals_label, axis=1, level=1)
        )
        right = self.df.T.groupby(level=0).sum().T
        comparison = left.eq(right).all(axis=None)
        self.assertTrue(comparison)

    def test_add_by_level_name(self):
        left = (
            totals
            .add_subtotals(self.df, axis=1, level='C0')
            .xs(self.subtotals_label, axis=1, level=1)
        )
        right = self.df.T.groupby(level=0).sum().T
        comparison = left.eq(right).all(axis=None)
        self.assertTrue(comparison)

    def test_add_multiple_levels(self):
        result = totals.add_subtotals(self.df, level=[0, 1], skip_single_rows=False)
        level0 = self.df.groupby(level=0).sum()
        level1 = self.df.groupby(level=1).sum()
        comparison1 = result.xs(self.subtotals_label, level=1).eq(level0).all(axis=None)
        comparison2 = result.xs(self.subtotals_label, level=2).eq(level1).all(axis=None)
        self.assertTrue(comparison1 & comparison2)

    def test_commutative_property(self):
        left = (
            self.df
            .pipe(totals.add_totals)
            .pipe(totals.add_subtotals, level=1)
        )
        right = (
            self.df
            .pipe(totals.add_subtotals, level=1)
            .pipe(totals.add_totals)
        )
        self.assertTrue(left.equals(right))

    def test_preserve_axis_names(self):
        r1 = totals.add_totals(self.df)
        r2 = totals.add_totals(self.df, axis=1)
        r3 = totals.add_subtotals(self.df, level=1)
        r4 = totals.add_subtotals(self.df, axis=1, level=0)
        self.assertEqual(self.df.index.names, r1.index.names)
        self.assertEqual(self.df.index.names, r2.index.names)
        self.assertEqual(self.df.index.names, r3.index.names)
        self.assertEqual(self.df.index.names, r4.index.names)

    def test_skip_single_rows_false(self):
        result = totals.add_subtotals(self.df, level=1, skip_single_rows=True)
        key = ('R_L0_G1', 'R_L1_G3', self.subtotals_label)
        self.assertTrue(key not in result.index)

    def test_custom_subtotals_label(self):
        custom_label = "Subtotes"
        result = totals.add_subtotals(
            self.df,
            axis=0,
            label=custom_label,
            _fill=self.fill,
        )
        key = ('R_L0_G0', custom_label, self.fill)
        self.assertTrue(key in result.index)

    def test_include_level_name(self):
        result = totals.add_subtotals(
            self.df,
            level=0,
            include_level_name=True,
            _fill=self.fill,
        )
        label_with_level = f"{self.subtotals_label} R_L0_G0"
        key = ('R_L0_G0', label_with_level, self.fill)
        self.assertTrue(key in result.index)

if __name__ == "__main__":
    unittest.main()
