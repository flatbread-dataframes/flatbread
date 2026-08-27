import unittest

import pandas as pd

from flatbread import DEFAULTS
from flatbread.output.html.display import DisplayConfig


# region _extract_margin_labels
class TestExtractMarginLabels(unittest.TestCase):
    def test_picks_up_config_totals_label(self):
        labels = DisplayConfig._extract_margin_labels(DEFAULTS.config, None)
        totals_label = DEFAULTS['transforms']['totals']['label']
        self.assertIn(totals_label, labels)

    def test_picks_up_config_subtotals_label(self):
        labels = DisplayConfig._extract_margin_labels(DEFAULTS.config, None)
        subtotals_label = DEFAULTS['transforms']['subtotals']['label']
        self.assertIn(subtotals_label, labels)

    def test_merges_config_and_tracked(self):
        attrs = {'flatbread': {'labels': {'aggregation': {'Mean'}}}}
        labels = DisplayConfig._extract_margin_labels(DEFAULTS.config, attrs)
        self.assertIn('Mean', labels)
        self.assertIn(DEFAULTS['transforms']['totals']['label'], labels)

    def test_empty_defaults_returns_tracked_only(self):
        attrs = {'flatbread': {'labels': {'totals': {'Total'}}}}
        labels = DisplayConfig._extract_margin_labels({}, attrs)
        self.assertEqual(labels, {'Total'})

    def test_no_attrs_returns_config_only(self):
        labels = DisplayConfig._extract_margin_labels(DEFAULTS.config, None)
        expected = {
            DEFAULTS['transforms']['totals']['label'],
            DEFAULTS['transforms']['subtotals']['label'],
        }
        self.assertEqual(labels, expected)


# region from_defaults
class TestDisplayConfigFromDefaults(unittest.TestCase):
    def test_margin_labels_include_config_labels(self):
        config = DisplayConfig.from_defaults(DEFAULTS.config)
        totals_label = DEFAULTS['transforms']['totals']['label']
        self.assertIn(totals_label, config.margin_labels)

    def test_margin_labels_include_tracked_attrs(self):
        attrs = {'flatbread': {'labels': {'aggregation': {'Mean'}}}}
        config = DisplayConfig.from_defaults(DEFAULTS.config, data_attrs=attrs)
        self.assertIn('Mean', config.margin_labels)


# region add_margin_labels
class TestAddMarginLabels(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'a': [1, 2, 3]})

    def test_adds_to_existing(self):
        self.df.pita.set_margin_labels('Totals')
        self.df.pita.add_margin_labels('Mean')
        labels = self.df.pita._config.margin_labels
        self.assertIn('Totals', labels)
        self.assertIn('Mean', labels)

    def test_add_multiple(self):
        self.df.pita.add_margin_labels('Mean', 'Median')
        labels = self.df.pita._config.margin_labels
        self.assertIn('Mean', labels)
        self.assertIn('Median', labels)

    def test_set_replaces_all(self):
        self.df.pita.add_margin_labels('Mean')
        self.df.pita.set_margin_labels('Totals')
        labels = self.df.pita._config.margin_labels
        self.assertNotIn('Mean', labels)
        self.assertIn('Totals', labels)

    def test_returns_self(self):
        result = self.df.pita.add_margin_labels('Mean')
        self.assertIs(result, self.df.pita)


if __name__ == '__main__':
    unittest.main()