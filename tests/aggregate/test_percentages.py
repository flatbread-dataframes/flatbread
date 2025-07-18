import unittest
from random import randint

import pandas as pd

import flatbread.percentages as pcts
from flatbread.testing.dataframe import make_test_df


# region vals & totes
class TestValuesAndTotals(unittest.TestCase):
    def setUp(self):
        self.label_totals = 'Totals'

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame({
            'A': [10, 20, 30],
            'B': [15, 25, 20],
        })

    def test_axis_0_complete_proportions_true(self):
        """Test axis=0 with complete proportions returns True"""
        data = self.df
        data.loc[self.label_totals] = data.sum()

        vt = pcts.ValuesAndTotals.from_data(
            data,
            axis = 0,
            label_totals = self.label_totals,
        )

        self.assertTrue(vt.should_use_apportioned_rounding)

    def test_axis_0_incomplete_proportions_false(self):
        """Test axis=0 with incomplete proportions returns False"""
        data = self.df
        data.loc[self.label_totals] = data.sum() * 2

        vt = pcts.ValuesAndTotals.from_data(
            data,
            axis = 0,
            label_totals = self.label_totals,
        )

        self.assertFalse(vt.should_use_apportioned_rounding)

    def test_axis_1_complete_proportions_true(self):
        """Test axis=1 with complete proportions returns True"""
        data = self.df
        data[self.label_totals] = data.sum(axis=1)

        vt = pcts.ValuesAndTotals.from_data(
            data,
            axis = 1,
            label_totals = self.label_totals,
        )

        self.assertTrue(vt.should_use_apportioned_rounding)

    def test_axis_1_incomplete_proportions_false(self):
        """Test axis=1 with incomplete proportions returns False"""
        data = self.df
        data[self.label_totals] = data.sum(axis=1) * 2

        vt = pcts.ValuesAndTotals.from_data(
            data,
            axis = 1,
            label_totals = self.label_totals,
        )

        self.assertFalse(vt.should_use_apportioned_rounding)

    def test_axis_2_complete_proportions_true(self):
        """Test axis=2 with complete proportions returns True"""
        data = self.df
        data[self.label_totals] = data.sum(axis=1)
        data.loc[self.label_totals] = data.sum()

        vt = pcts.ValuesAndTotals.from_data(
            data,
            axis = 2,
            label_totals = self.label_totals,
        )

        self.assertTrue(vt.should_use_apportioned_rounding)

    def test_axis_2_incomplete_proportions_false(self):
        """Test axis=2 with incomplete proportions returns False"""
        data = self.df
        data[self.label_totals] = data.sum(axis=1) * 2
        data.loc[self.label_totals] = data.sum()

        vt = pcts.ValuesAndTotals.from_data(
            data,
            axis = 2,
            label_totals = self.label_totals,
        )

        self.assertFalse(vt.should_use_apportioned_rounding)


# region transform
class TestPercsTransform_DataFrameSimple(unittest.TestCase):
    def setUp(self):
        self.df = make_test_df(
            nrows = 5,
            ncols = 4,
            data_gen_f = lambda r,c:randint(1,100),
        ).pita.add_totals()

    def test_first_value_of_row_pct(self):
        """Test first cell in row percentages"""
        result = self.df.pipe(
            pcts.as_percentages,
            axis = 0,
            ndigits = -1,
        )
        test_value = result.iloc[0, 0]
        cell = self.df.iloc[0, 0]
        total = self.df.iloc[:-1, 0].sum()
        percentage = cell / total

        self.assertEqual(test_value, percentage)

    def test_first_value_of_col_pct(self):
        """Test first cell in column percentages"""
        result = self.df.pipe(
            pcts.as_percentages,
            axis = 1,
            ndigits = -1,
        )
        test_value = result.iloc[0, 0]
        cell = self.df.iloc[0, 0]
        total = self.df.iloc[0, :-1].sum()
        percentage = cell / total

        self.assertEqual(test_value, percentage)

    def test_first_value_of_grand_pct(self):
        """Test first cell in grand total percentage"""
        result = self.df.pipe(
            pcts.as_percentages,
            axis = 2,
            ndigits = -1,
        )
        test_value = result.iloc[0, 0]
        cell = self.df.iloc[0, 0]
        total = self.df.iloc[-1, -1]
        percentage = cell / total

        self.assertEqual(test_value, percentage)

    def test_row_pct_adds_up_to_base(self):
        """Test first cell in column totals"""
        result = self.df.pipe(
            pcts.as_percentages,
            axis = 0,
            ndigits = -1,
            base = 1,
        )
        test_target = result.iloc[:-1, 0].sum()

        self.assertAlmostEqual(test_target, 1, places=7)

    def test_col_pct_adds_up_to_base(self):
        """Test first cell in column totals"""
        result = self.df.pipe(
            pcts.as_percentages,
            axis = 1,
            ndigits = -1,
            base = 1,
        )
        test_target = result.iloc[0, :-1].sum()

        self.assertAlmostEqual(test_target, 1, places=7)

    def test_grand_pct_adds_up_to_base(self):
        """Test first cell in column totals"""
        result = self.df.pipe(
            pcts.as_percentages,
            axis = 2,
            ndigits = -1,
            base = 1,
        )
        test_target = result.iloc[:-1, :-1].sum().sum()

        self.assertAlmostEqual(test_target, 1, places=7)


# region rounding
class TestApportionedRoundingBehavior(unittest.TestCase):
    def setUp(self):
        self.df = make_test_df(
            nrows=3,
            ncols=1,
            data_gen_f=lambda r,c:100/3,
        ).pita.add_totals()

    def test_dataframe_apportioned_true_sums_to_base(self):
        """Test that apportioned=True makes percentages sum to base"""
        result = pcts.as_percentages(
            self.df,
            axis = 0,
            ndigits = 0,
            base = 100,
            apportioned_rounding = True,
        )
        summed = result.iloc[:-1, 0].sum()
        self.assertEqual(summed, 100.0)

    def test_dataframe_apportioned_false_may_not_sum_to_base(self):
        """Test that apportioned=False may not sum to base due to rounding"""
        result = pcts.as_percentages(
            self.df,
            axis = 0,
            ndigits = 0,
            base = 100,
            apportioned_rounding = False,
        )

        summed = result.iloc[:-1, 0].sum()
        self.assertEqual(summed, 99.0)
