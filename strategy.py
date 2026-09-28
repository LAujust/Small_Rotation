"""Backtest the super 20/80 rotation strategy.

The strategy compares 20-trading-day momentum for the CSI 100 and ChiNext
indexes. It holds the stronger index when that index has positive momentum;
otherwise it holds cash. A signal observed at today's close becomes the
position for the next trading day, avoiding look-ahead bias.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRADING_DAYS = 252
STRATEGY_NAME = "Super 20/80 Rotation"
CSI100_NAME = "CSI 100"
CHINEXT_NAME = "ChiNext"
CASH = "Cash"


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str = "2015-01-01"
    end_date: str = date.today().isoformat()
    lookback: int = 20
    initial_capital: float = 500_000.0
    risk_free_rate: float = 0.0
    transaction_cost_bps: float = 0.0


def load_close(path: Path, output_name: str) -> pd.Series:
    frame = pd.read_csv(path, usecols=["date", "close"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna().drop_duplicates("date").sort_values("date")
    if frame.empty:
        raise ValueError(f"No usable price rows in {path}")
    return frame.set_index("date")["close"].rename(output_name)


def build_price_frame(csi100_path: Path, chinext_path: Path) -> pd.DataFrame:
    prices = pd.concat(
        [load_close(csi100_path, CSI100_NAME), load_close(chinext_path, CHINEXT_NAME)],
        axis=1,
        join="inner",
    ).dropna()
    if prices.empty:
        raise ValueError("The two index files have no common trading dates")
    return prices


def run_backtest(prices: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    if config.lookback <= 0:
        raise ValueError("lookback must be positive")
    if config.initial_capital <= 0:
        raise ValueError("initial_capital must be positive")

    prices = prices.sort_index().copy()
    momentum = prices.pct_change(config.lookback, fill_method=None)

    signal = pd.Series(CASH, index=prices.index, dtype="object")
    choose_csi100 = (momentum[CSI100_NAME] > momentum[CHINEXT_NAME]) & (
        momentum[CSI100_NAME] > 0
    )
    choose_chinext = (momentum[CHINEXT_NAME] > momentum[CSI100_NAME]) & (
        momentum[CHINEXT_NAME] > 0
    )
    signal.loc[choose_csi100] = CSI100_NAME
    signal.loc[choose_chinext] = CHINEXT_NAME

    # The close-to-close return at t uses the signal known at close t-1.
    position = signal.shift(1, fill_value=CASH)
    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)
    strategy_returns = pd.Series(0.0, index=prices.index, name="strategy_return")
    strategy_returns.loc[position == CSI100_NAME] = asset_returns.loc[
        position == CSI100_NAME, CSI100_NAME
    ]
    strategy_returns.loc[position == CHINEXT_NAME] = asset_returns.loc[
        position == CHINEXT_NAME, CHINEXT_NAME
    ]

    position_change = position.ne(position.shift(1, fill_value=CASH))
    cost_rate = config.transaction_cost_bps / 10_000
    if cost_rate:
        strategy_returns = strategy_returns - position_change.astype(float) * cost_rate

    result = pd.DataFrame(index=prices.index)
    result[CSI100_NAME] = prices[CSI100_NAME]
    result[CHINEXT_NAME] = prices[CHINEXT_NAME]
    result["csi100_momentum"] = momentum[CSI100_NAME]
    result["chinext_momentum"] = momentum[CHINEXT_NAME]
    result["signal"] = signal
    result["position"] = position
    result["position_changed"] = position_change
    result["strategy_return"] = strategy_returns
    result["csi100_return"] = asset_returns[CSI100_NAME]
    result["chinext_return"] = asset_returns[CHINEXT_NAME]

    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    result = result.loc[(result.index >= start) & (result.index <= end)].copy()
    if result.empty:
        raise ValueError(f"No common observations between {config.start_date} and {config.end_date}")

    for return_column, nav_column in [
        ("strategy_return", "strategy_nav"),
        ("csi100_return", "csi100_nav"),
        ("chinext_return", "chinext_nav"),
    ]:
        # Rebase all series to exactly 1.0 at the beginning of the requested window.
        returns = result[return_column].copy()
        returns.iloc[0] = 0.0
        result[return_column] = returns
        result[nav_column] = (1.0 + returns).cumprod()

    result["strategy_equity"] = result["strategy_nav"] * config.initial_capital
    return result


def performance_metrics(
    returns: pd.Series,
    nav: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict[str, float | int]:
    returns = returns.fillna(0.0)
    observations = len(returns)
    years = observations / TRADING_DAYS
    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    annual_return = (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0 if years else np.nan
    annual_volatility = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    daily_risk_free = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS) - 1.0
    excess_returns = returns - daily_risk_free
    return_std = returns.std(ddof=1)
    sharpe = (
        excess_returns.mean() / return_std * np.sqrt(TRADING_DAYS)
        if return_std > 0
        else np.nan
    )
    downside = np.minimum(excess_returns, 0.0)
    downside_deviation = np.sqrt(np.mean(np.square(downside))) * np.sqrt(TRADING_DAYS)
    sortino = (
        excess_returns.mean() * TRADING_DAYS / downside_deviation
        if downside_deviation > 0
        else np.nan
    )
    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = drawdown.min()
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.nan

    return {
        "observations": observations,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "daily_win_rate": (returns > 0).mean(),
        "active_day_win_rate": (returns[returns != 0] > 0).mean(),
    }


def build_metrics(result: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    series = {
        STRATEGY_NAME: ("strategy_return", "strategy_nav"),
        CSI100_NAME: ("csi100_return", "csi100_nav"),
        CHINEXT_NAME: ("chinext_return", "chinext_nav"),
    }
    metrics = {
        name: performance_metrics(
            result[return_column], result[nav_column], config.risk_free_rate
        )
        for name, (return_column, nav_column) in series.items()
    }
    frame = pd.DataFrame(metrics).T
    frame.index.name = "series"
    frame.loc[STRATEGY_NAME, "position_changes"] = int(result["position_changed"].sum())
    frame.loc[STRATEGY_NAME, "invested_ratio"] = (result["position"] != CASH).mean()
    return frame


def build_annual_returns(result: pd.DataFrame) -> pd.DataFrame:
    daily_returns = result[["strategy_return", "csi100_return", "chinext_return"]].rename(
        columns={
            "strategy_return": STRATEGY_NAME,
            "csi100_return": CSI100_NAME,
            "chinext_return": CHINEXT_NAME,
        }
    )
    annual = (1.0 + daily_returns).groupby(daily_returns.index.year).prod() - 1.0
    annual.index.name = "year"
    return annual


def build_trades(result: pd.DataFrame) -> pd.DataFrame:
    changes = result.loc[
        result["position_changed"], ["position", CSI100_NAME, CHINEXT_NAME]
    ].copy()
    changes.index.name = "date"
    return changes.rename(columns={"position": "new_position"})


def save_chart(result: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=160)
    ax.plot(result.index, result["strategy_nav"], label=STRATEGY_NAME, linewidth=2.0)
    ax.plot(result.index, result["csi100_nav"], label=CSI100_NAME, linewidth=1.2)
    ax.plot(result.index, result["chinext_nav"], label=CHINEXT_NAME, linewidth=1.2)
    ax.set_title("Super 20/80 Rotation Backtest")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net value (start = 1.0)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="超级二八轮动回测")
    parser.add_argument("--csi100", type=Path, default=root / "csi100_index.csv")
    parser.add_argument("--chinext", type=Path, default=root / "chinext_index.csv")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--initial-capital", type=float, default=500_000.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    return parser.parse_args()


def format_metrics_for_display(metrics: pd.DataFrame) -> pd.DataFrame:
    displayed = metrics.copy()
    percent_columns = [
        "total_return",
        "annual_return",
        "annual_volatility",
        "max_drawdown",
        "daily_win_rate",
        "active_day_win_rate",
        "invested_ratio",
    ]
    for column in percent_columns:
        if column in displayed:
            displayed[column] = displayed[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.2%}"
            )
    for column in ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]:
        displayed[column] = displayed[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )
    return displayed


def main() -> None:
    args = parse_args()
    config = BacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        lookback=args.lookback,
        initial_capital=args.initial_capital,
        risk_free_rate=args.risk_free_rate,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    prices = build_price_frame(args.csi100, args.chinext)
    result = run_backtest(prices, config)
    metrics = build_metrics(result, config)
    annual_returns = build_annual_returns(result)
    trades = build_trades(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        args.output_dir / "backtest_daily.csv", encoding="utf-8-sig", float_format="%.10f"
    )
    metrics.to_csv(
        args.output_dir / "performance_metrics.csv", encoding="utf-8-sig", float_format="%.10f"
    )
    annual_returns.to_csv(
        args.output_dir / "annual_returns.csv", encoding="utf-8-sig", float_format="%.10f"
    )
    trades.to_csv(args.output_dir / "trades.csv", encoding="utf-8-sig", float_format="%.4f")
    save_chart(result, args.output_dir / "super_28_rotation.png")

    print(f"Backtest: {result.index[0].date()} to {result.index[-1].date()} ({len(result)} days)")
    print(format_metrics_for_display(metrics).to_string())
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
