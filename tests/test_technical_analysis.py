import unittest

import numpy as np
import pandas as pd

from technical_analysis import (
    DOWN, UP, WAIT, analyze_prices, completed_session_cutoff, evaluate_signals,
    indicator_signals, wilder_average,
)


def sample_prices():
    dates = pd.bdate_range(end="2025-01-10", periods=201)
    close = np.linspace(100, 150, len(dates))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.02,
            "Low": close * 0.98,
            "Close": close,
            "Volume": 100_000,
        },
        index=dates,
    )


class TechnicalAnalysisTests(unittest.TestCase):
    def test_market_cutoff_uses_exchange_local_time(self):
        before = pd.Timestamp("2025-01-10T15:29Z")
        after = pd.Timestamp("2025-01-10T15:30Z")
        self.assertEqual(str(completed_session_cutoff(True, before)), "2025-01-09")
        self.assertEqual(str(completed_session_cutoff(True, after)), "2025-01-10")
        self.assertEqual(str(completed_session_cutoff(False, after)), "2025-01-09")

    def test_signal_thresholds_and_disagreement(self):
        indicators = pd.DataFrame(
            {
                "Close": [101, 101, 99, 99, 101, 99, 101, 101, 100],
                "SMA_200": [100] * 9,
                "MACD_Hist": [1, 1, -1, -1, 1, -1, -1, 0, 1],
                "RSI": [55, 70, 45, 30, 71, 29, 55, 55, 50],
            }
        )
        self.assertEqual(indicator_signals(indicators).tolist(), [UP, UP, DOWN, DOWN] + [WAIT] * 5)

    def test_history_uses_next_close_and_excludes_unresolved_signal(self):
        indicators = pd.DataFrame(
            {
                "Close": [101, 103, 99, 98, 102],
                "SMA_200": [100] * 5,
                "MACD_Hist": [1, 1, -1, -1, 1],
                "RSI": [55, 55, 45, 50, 55],
                "Return": [0.01] * 5,
                "ATR": [2] * 5,
            }
        )
        result = evaluate_signals(indicators, indicator_signals(indicators))
        self.assertEqual(result, {"sessions": 4, "signals": 3, "hits": 2, "accuracy": 2 / 3})

    def test_wilder_average_seeds_and_restarts_after_missing_data(self):
        values = pd.Series([1.0, 3.0, 4.0, np.nan, 2.0, 4.0, 7.0])
        expected = pd.Series([np.nan, 2.0, 3.0, np.nan, np.nan, 3.0, 5.0])
        pd.testing.assert_series_equal(wilder_average(values, 2), expected)

    def test_analysis_excludes_incomplete_session_and_reports_stale_data(self):
        prices = sample_prices()
        _, summary = analyze_prices(prices, False, pd.Timestamp("2025-01-10T20:00Z"))
        self.assertEqual(str(summary["Son Seans"]), "2025-01-09")
        self.assertEqual(summary["RSI"], 100)
        self.assertEqual(summary["Sinyal"], WAIT)

        _, stale_summary = analyze_prices(prices, False, pd.Timestamp("2025-02-01T22:00Z"))
        self.assertEqual(stale_summary["Sinyal"], "VERİ ESKİ")

    def test_insufficient_history_is_reported(self):
        with self.assertRaisesRegex(ValueError, "en az 200"):
            analyze_prices(sample_prices().iloc[-199:], False)
