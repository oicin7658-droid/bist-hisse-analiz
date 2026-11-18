from datetime import time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
SIGNAL_COLUMNS = ["Return", "Close", "SMA_200", "MACD_Hist", "RSI", "ATR"]
UP = "YÜKSELİŞ SİNYALİ"
DOWN = "DÜŞÜŞ SİNYALİ"
WAIT = "BEKLE"


def completed_session_cutoff(is_bist, now=None):
    """Allow a buffer for publication of the exchange's daily close."""
    zone = ZoneInfo("Europe/Istanbul" if is_bist else "America/New_York")
    local_now = pd.Timestamp.now(tz=zone) if now is None else pd.Timestamp(now).tz_convert(zone)
    cutoff = time(18, 30) if is_bist else time(16, 30)
    return local_now.date() - timedelta(days=local_now.time() < cutoff)


def prepare_history(prices):
    if prices is None or prices.empty:
        raise ValueError("Fiyat verisi alınamadı. Sembolü ve Yahoo Finance erişimini kontrol edin.")
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise ValueError("Fiyat verisinde geçersiz seans tarihi var.")
    prices = prices.copy()
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    if not prices.columns.is_unique or not set(OHLCV).issubset(prices.columns):
        raise ValueError("Fiyat verisi gerekli OHLCV sütunlarını içermiyor.")
    df = prices.loc[:, OHLCV].apply(pd.to_numeric, errors="coerce").sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    valid = (
        np.isfinite(df).all(axis=1)
        & df[["Open", "High", "Low", "Close"]].gt(0).all(axis=1)
        & df["Volume"].ge(0)
        & df["High"].ge(df[["Open", "Close", "Low"]].max(axis=1))
        & df["Low"].le(df[["Open", "Close"]].min(axis=1))
    )
    return df.where(valid, axis=0)


def wilder_average(values, window):
    groups = values.isna().cumsum()
    seed = values.rolling(window).mean()
    position = values.notna().groupby(groups).cumsum()
    inputs = seed.where(position.eq(window), values.where(position.gt(window)))
    return inputs.groupby(groups).transform(
        lambda series: series.ewm(alpha=1 / window, adjust=False).mean()
    )


def add_indicators(df):
    df = df.copy()
    # Temel İndikatörler
    df["Return"] = df["Close"].pct_change(fill_method=None)
    # Wilder RSI with a 14-session initial average.
    delta = df["Close"].diff()
    gain = wilder_average(delta.clip(lower=0), 14)
    loss = wilder_average(-delta.clip(upper=0), 14)
    rsi = 100 - (100 / (1 + gain / loss))
    df["RSI"] = rsi.mask((gain == 0) & (loss == 0), 50)

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands & ATR
    sma_20 = df["Close"].rolling(window=20).mean()
    std_20 = df["Close"].rolling(window=20).std()
    df["Upper_Band"] = sma_20 + (std_20 * 2)
    df["Lower_Band"] = sma_20 - (std_20 * 2)

    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df["ATR"] = wilder_average(true_range, 14)

    # 200 SMA
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    return df


def indicator_signals(history):
    close = history["Close"]
    sma = history["SMA_200"]
    macd = history["MACD_Hist"]
    rsi = history["RSI"]
    up = (close > sma) & (macd > 0) & (rsi > 50) & (rsi <= 70)
    down = (close < sma) & (macd < 0) & (rsi >= 30) & (rsi < 50)
    return pd.Series(np.select([up, down], [UP, DOWN], default=WAIT), index=history.index)


def evaluate_signals(history, signals):
    next_close = history["Close"].shift(-1)
    eligible = next_close.notna() & history[SIGNAL_COLUMNS].notna().all(axis=1)
    move = next_close[eligible] - history.loc[eligible, "Close"]
    tested = signals[eligible]
    up = tested.eq(UP)
    down = tested.eq(DOWN)
    count = int((up | down).sum())
    hits = int(((up & move.gt(0)) | (down & move.lt(0))).sum())
    return {
        "sessions": len(tested),
        "signals": count,
        "hits": hits,
        "accuracy": hits / count if count else None,
    }


def analyze_prices(prices, is_bist, now=None):
    cutoff = completed_session_cutoff(is_bist, now)
    history = prepare_history(prices)
    history = history.loc[history.index.date <= cutoff]
    df = add_indicators(history).replace([np.inf, -np.inf], np.nan)
    if len(df) < 200 or df.iloc[-1][SIGNAL_COLUMNS].isna().any():
        raise ValueError("Göstergeleri hesaplamak için en az 200 geçerli günlük seans gerekiyor.")

    signals = indicator_signals(df)
    latest = df.iloc[-1]
    close = float(latest["Close"])
    pivot = (latest["High"] + latest["Low"] + close) / 3
    support = 2 * pivot - latest["High"]
    resistance = 2 * pivot - latest["Low"]
    last_session = df.index[-1].date()
    signal = "VERİ ESKİ" if (cutoff - last_session).days > 3 else signals.iloc[-1]

    return df, {
        "Son Seans": last_session,
        "Son Fiyat": round(close, 2),
        "Günlük Değişim (%)": round(float(latest["Return"]) * 100, 2),
        "Destek S1": round(float(support), 2),
        "Direnç R1": round(float(resistance), 2),
        "ATR (14)": round(float(latest["ATR"]), 2),
        "Sinyal": signal,
        "SMA 200": round(float(latest["SMA_200"]), 2),
        "MACD Histogramı": round(float(latest["MACD_Hist"]), 4),
        "RSI": round(float(latest["RSI"]), 1),
        "Test": evaluate_signals(df, signals),
    }
