"""Download index data used by the super 20/80 rotation backtest.

AkShare/Sina is the default source; Eastmoney and BaoStock are optional. The
extra data before 2015 is the warm-up period for the 20-trading-day signal.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import baostock as bs
import pandas as pd


INDEXES = {
    # filename: (BaoStock code, Eastmoney security id, AkShare symbol)
    "csi100_index.csv": ("sh.000903", "1.000903", "sh000903"),  # 中证100/中证A100
    "chinext_index.csv": ("sz.399006", "0.399006", "sz399006"),  # 创业板指
}
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"


def download_baostock(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    result = bs.query_history_k_data_plus(
        code,
        FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",
    )
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock query failed for {code}: {result.error_msg}")

    rows: list[list[str]] = []
    while result.next():
        rows.append(result.get_row_data())
    if not rows:
        raise RuntimeError(f"BaoStock returned no rows for {code}")

    frame = pd.DataFrame(rows, columns=result.fields)
    frame["date"] = pd.to_datetime(frame["date"])
    numeric_columns = [column for column in frame.columns if column not in {"date", "code"}]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["close"]).sort_values("date").drop_duplicates("date")
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame


def download_eastmoney(secid: str, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "secid": secid,
        "klt": "101",
        "fqt": "0",
        "lmt": "10000",
        "beg": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    data = payload.get("data")
    if payload.get("rc") != 0 or not data or not data.get("klines"):
        raise RuntimeError(f"Eastmoney returned no rows for {secid}: {payload}")

    kline_columns = [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "pctChg",
        "change",
        "turn",
    ]
    rows = [item.split(",") for item in data["klines"]]
    frame = pd.DataFrame(rows, columns=kline_columns)
    frame.insert(1, "code", secid)
    frame.insert(6, "preclose", frame["close"].shift(1))
    numeric_columns = [column for column in frame.columns if column not in {"date", "code"}]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["close"]).sort_values("date").drop_duplicates("date")
    return frame[["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"]]


def download_akshare(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    # Import lazily so users selecting another source do not need AkShare installed.
    import akshare as ak

    frame = ak.stock_zh_index_daily(symbol=symbol).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    frame = frame.loc[(frame["date"] >= start) & (frame["date"] <= end)].copy()
    if frame.empty:
        raise RuntimeError(f"AkShare returned no rows for {symbol}")
    frame.insert(1, "code", symbol)
    frame["preclose"] = frame["close"].shift(1)
    frame["amount"] = float("nan")
    frame["pctChg"] = frame["close"].pct_change(fill_method=None) * 100
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame[["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载超级二八轮动所需指数行情")
    parser.add_argument("--start-date", default="2014-11-01", help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", default=date.today().isoformat(), help="结束日期 YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--source",
        choices=["akshare", "eastmoney", "baostock"],
        default="akshare",
        help="行情源（默认 akshare；Eastmoney/BaoStock 可作为备选）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "baostock":
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {login.error_msg}")

    try:
        for filename, (baostock_code, eastmoney_secid, akshare_symbol) in INDEXES.items():
            if args.source == "baostock":
                frame = download_baostock(baostock_code, args.start_date, args.end_date)
                source_code = baostock_code
            elif args.source == "eastmoney":
                frame = download_eastmoney(eastmoney_secid, args.start_date, args.end_date)
                source_code = eastmoney_secid
            else:
                frame = download_akshare(akshare_symbol, args.start_date, args.end_date)
                source_code = akshare_symbol
            output_path = args.output_dir / filename
            frame.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(
                f"{source_code}: {len(frame)} rows, {frame['date'].iloc[0]} to "
                f"{frame['date'].iloc[-1]} -> {output_path}"
            )
    finally:
        if args.source == "baostock":
            bs.logout()


if __name__ == "__main__":
    main()
