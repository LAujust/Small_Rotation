import unittest

import pandas as pd

from strategy import (
    CASH,
    CHINEXT_NAME,
    CSI100_NAME,
    BacktestConfig,
    performance_metrics,
    run_backtest,
)


class StrategyTests(unittest.TestCase):
    def test_signal_is_applied_one_day_later(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=8)
        prices = pd.DataFrame(
            {
                CSI100_NAME: [100, 100, 110, 121, 133.1, 146.41, 161.051, 177.1561],
                CHINEXT_NAME: [100] * 8,
            },
            index=dates,
        )
        result = run_backtest(
            prices,
            BacktestConfig(
                start_date="2020-01-01", end_date="2020-01-31", lookback=2
            ),
        )

        self.assertEqual(result.loc[dates[2], "signal"], CSI100_NAME)
        self.assertEqual(result.loc[dates[2], "position"], CASH)
        self.assertEqual(result.loc[dates[3], "position"], CSI100_NAME)
        self.assertAlmostEqual(result.loc[dates[3], "strategy_return"], 0.1)

    def test_both_negative_momenta_hold_cash(self) -> None:
        dates = pd.bdate_range("2020-01-01", periods=6)
        prices = pd.DataFrame(
            {
                CSI100_NAME: [100, 99, 98, 97, 96, 95],
                CHINEXT_NAME: [100, 98, 96, 94, 92, 90],
            },
            index=dates,
        )
        result = run_backtest(
            prices,
            BacktestConfig(
                start_date="2020-01-01", end_date="2020-01-31", lookback=2
            ),
        )
        self.assertTrue((result["position"] == CASH).all())
        self.assertAlmostEqual(result["strategy_nav"].iloc[-1], 1.0)

    def test_drawdown_metric(self) -> None:
        returns = pd.Series([0.0, 0.1, -0.2, 0.1])
        nav = (1.0 + returns).cumprod()
        metrics = performance_metrics(returns, nav)
        self.assertAlmostEqual(metrics["max_drawdown"], -0.2)
        self.assertAlmostEqual(metrics["total_return"], nav.iloc[-1] - 1.0)


if __name__ == "__main__":
    unittest.main()
