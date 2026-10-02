import warnings
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st
import yfinance as yf
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. SABİTLER VE LİSTELER
# ==============================================================================
# Telegram credentials are loaded from Streamlit secrets or environment variables.
def get_secret(name, env_name):
    try:
        return st.secrets.get(name, os.getenv(env_name, ""))
    except Exception:
        return os.getenv(env_name, "")


DEFAULT_TELEGRAM_TOKEN = get_secret("TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_TOKEN")
DEFAULT_TELEGRAM_CHAT_ID = get_secret("TELEGRAM_CHAT_ID", "TELEGRAM_CHAT_ID")

BIST_30 = [
    "AKBNK", "ALARK", "ASELS", "BIMAS", "BRSAN", "DOAS", "EKGYO", "ENKAI", 
    "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", "KCHOL", "KONTR", 
    "KOZAL", "KRDMD", "ODAS", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", 
    "TCELL", "THYAO", "TOASO", "TUPRS", "YKBNK", "ASTOR"
]

BIST_BANKA = ["AKBNK", "GARAN", "ISCTR", "YKBNK", "VAKBN", "HALKB", "SKBNK", "TSKB", "ALBRK"]

BIST_SANAYI_TEKNO = [
    "ASELS", "EREGL", "TUPRS", "FROTO", "TOASO", "SASA", "SISE", "BRSAN", 
    "ASTOR", "KONTR", "MIATK", "REEDR", "KCAER", "HEKTS", "ALARK", "VESBE",
    "ARCLK", "EGEEN", "KORDS", "TAVHL"
]

BIST_TUM_POPULER = list(set(BIST_30 + BIST_BANKA + BIST_SANAYI_TEKNO + [
    "BIMAS", "CCHOL", "DOAS", "ENKAI", "GUBRF", "KOZAL", "KRDMD", "ODAS", 
    "PETKM", "PGSUS", "SAHOL", "TCELL", "THYAO", "ARCLK", "MAVI", "TKFEN", "SOKM",
    "MAGEN", "LOGIN", "BETA", "GWIND", "SMRTG", "EUPWR", "ALFAS", "KLYSN"
]))

# ==============================================================================
# 2. SAYFA YAPILANDIRMASI
# ==============================================================================
st.set_page_config(
    page_title="PRO BIST & US Yapay Zeka Platformu 550+",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ PRO BİST Yüksek Hassasiyetli Yapay Zeka Platformu (Tam Sürüm)")
st.caption(
    "Netleştirilmiş Sinyaller | Trend Öngörü Algoritması | Gelişmiş Backtest Engine | Canlı Plotly Grafik | Risk Yönetimi"
)
st.divider()

# ==============================================================================
# 3. VERİ TEDARİK VE İŞLEME MOTORU (YFINANCE + STOOQ API)
# ==============================================================================
def get_stock_data_hybrid(symbol_input, period="3y", interval="1d", is_bist=True):
    """
    Veriyi yfinance ile çeker, hata durumunda Stooq API yedeğine düşer.
    """
    symbol = f"{symbol_input}.IS" if is_bist and not symbol_input.endswith(".IS") else symbol_input
    
    # Primary Source: Yahoo Finance
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is not None and not df.empty and len(df) > 100:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception:
        pass

    # Secondary Source: Stooq API
    try:
        stooq_symbol = f"{symbol_input}.PL" if is_bist else symbol_input
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
        df_stooq = pd.read_csv(url)
        if not df_stooq.empty and "Close" in df_stooq.columns:
            df_stooq["Date"] = pd.to_datetime(df_stooq["Date"])
            df_stooq.set_index("Date", inplace=True)
            df_stooq.sort_index(inplace=True)
            return df_stooq
    except Exception:
        pass

    return None

# ==============================================================================
# 4. TELEGRAM ENTEGRASYONU
# ==============================================================================
def send_telegram_message(bot_token, chat_id, message):
    if not bot_token or not chat_id:
        return False, "Bot Token veya Chat ID eksik."
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=5)
        res_data = response.json()
        return (True, "Başarılı") if res_data.get("ok") else (False, res_data.get("description", "Bilinmeyen hata"))
    except Exception as e:
        return False, str(e)

# ==============================================================================
# 5. MATEMATİKSEL İNDİKATÖR KÜTÜPHANESİ
# ==============================================================================
def calculate_indicators(df):
    """
    Gelişmiş teknik indikatör hesaplama havuzu.
    """
    df = df.copy()
    
    # Temel Fiyat Değişimi
    df["Return"] = df["Close"].pct_change()
    
    # Hareketli Ortalamalar
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    # Bollinger Bantları
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    df["BB_Std"] = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (df["BB_Std"] * 2)
    df["BB_Lower"] = df["BB_Middle"] - (df["BB_Std"] * 2)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

    # RSI & Stokastik RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    rsi_min = df["RSI"].rolling(14).min()
    rsi_max = df["RSI"].rolling(14).max()
    df["Stoch_RSI"] = (df["RSI"] - rsi_min) / (rsi_max - rsi_min + 1e-9)

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # ATR (Average True Range)
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df["ATR"] = np.max(ranges, axis=1).rolling(14).mean()
    df["ATR_PCT"] = df["ATR"] / df["Close"]

    # Williams %R
    highest_high = df["High"].rolling(14).max()
    lowest_low = df["Low"].rolling(14).min()
    df["Williams_R"] = -100 * ((highest_high - df["Close"]) / (highest_high - lowest_low + 1e-9))

    # On-Balance Volume (OBV)
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

    # Chaikin Money Flow (CMF)
    mf_multiplier = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-9)
    mf_volume = mf_multiplier * df["Volume"]
    df["CMF"] = mf_volume.rolling(20).sum() / (df["Volume"].rolling(20).sum() + 1e-9)

    # Kırılım seviyeleri: güncel mum, önceki 20 seansın seviyelerine dahil değildir.
    df["Breakout_Resistance_20"] = df["High"].rolling(20).max().shift(1)
    df["Breakout_Support_20"] = df["Low"].rolling(20).min().shift(1)
    df["Volume_Avg_20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio_20"] = df["Volume"] / (df["Volume_Avg_20"] + 1e-9)
    df["Breakout_Up"] = (df["Close"] > df["Breakout_Resistance_20"]) & (df["Volume_Ratio_20"] >= 1.5)
    df["Breakout_Down"] = (df["Close"] < df["Breakout_Support_20"]) & (df["Volume_Ratio_20"] >= 1.5)

    # Smart Market Structure: confirmed pivot breakout signals.
    pivot_len = 5
    pivot_window = 2 * pivot_len + 1
    pivot_high_candidate = df["High"].eq(df["High"].rolling(pivot_window, center=True).max()).shift(pivot_len).fillna(False)
    pivot_low_candidate = df["Low"].eq(df["Low"].rolling(pivot_window, center=True).min()).shift(pivot_len).fillna(False)
    df["Structure_Resistance"] = df["High"].where(pivot_high_candidate).ffill()
    df["Structure_Support"] = df["Low"].where(pivot_low_candidate).ffill()
    df["Smart_Long"] = (df["Close"] > df["Structure_Resistance"]) & (df["Close"].shift(1) <= df["Structure_Resistance"].shift(1))
    df["Smart_Short"] = (df["Close"] < df["Structure_Support"]) & (df["Close"].shift(1) >= df["Structure_Support"].shift(1))

    # Institutional Flow contributes only its lower-liquidity SELL signal.
    inst_pivot_len = 10
    inst_window = 2 * inst_pivot_len + 1
    inst_low_candidate = df["Low"].eq(df["Low"].rolling(inst_window, center=True).min()).shift(inst_pivot_len).fillna(False)
    df["Institutional_Lower_Liquidity"] = df["Low"].where(inst_low_candidate).ffill()
    df["Institutional_Sell"] = (df["Close"] < df["Institutional_Lower_Liquidity"]) & (df["Close"].shift(1) >= df["Institutional_Lower_Liquidity"].shift(1))
    df["Merged_Long"] = df["Smart_Long"]
    df["Merged_Short"] = df["Smart_Short"] | df["Institutional_Sell"]
    return df




def normalize_symbol_list(raw_text):
    """Virgul, bosluk ve satir sonu ile ayrilmis sembolleri tekilleştirir."""
    if not raw_text:
        return []
    normalized=[]
    seen=set()
    for token in str(raw_text).replace("\n", ",").replace(";", ",").split(","):
        symbol=token.strip().upper()
        if not symbol:
            continue
        symbol=symbol.replace(".IS", "")
        if symbol not in seen:
            normalized.append(symbol)
            seen.add(symbol)
    return normalized


def validate_market_data(df):
    """OHLCV girdisini analizden once denetler ve sorunlari raporlar."""
    required=["Open", "High", "Low", "Close", "Volume"]
    missing=[column for column in required if column not in df.columns]
    if missing:
        return False, ["Eksik alanlar: " + ", ".join(missing)]
    problems=[]
    if df.empty:
        return False, ["Veri tablosu bos."]
    if not df.index.is_monotonic_increasing:
        problems.append("Tarih sirasi duzeltildi.")
    if df.index.has_duplicates:
        problems.append("Tekrarlanan tarihler temizlendi.")
    numeric=df[required].apply(pd.to_numeric, errors="coerce")
    if numeric["Close"].isna().any():
        problems.append("Gecersiz kapanis fiyatlari bulundu.")
    if (numeric[["Open", "High", "Low", "Close"]] <= 0).any().any():
        problems.append("Sifir veya negatif fiyat kayitlari bulundu.")
    if (numeric["Volume"] < 0).any():
        problems.append("Negatif hacim degerleri bulundu.")
    return True, problems


def clean_market_data(df):
    """Veriyi siralar, tekrarli satirlari kaldirir ve OHLCV'yi sayisala cevirir."""
    if df is None:
        return None
    result=df.copy()
    result=result[~result.index.duplicated(keep="last")]
    result=result.sort_index()
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column in result.columns:
            result[column]=pd.to_numeric(result[column], errors="coerce")
    result=result.dropna(subset=["Open", "High", "Low", "Close"])
    result["Volume"]=result["Volume"].fillna(0).clip(lower=0)
    return result


def market_regime_snapshot(row):
    """Son mum icin trend, volatilite ve hacim rejimini ozetler."""
    close=float(row.get("Close", np.nan))
    atr_pct=float(row.get("ATR_PCT", np.nan))
    adx=float(row.get("ADX", np.nan))
    rsi=float(row.get("RSI", np.nan))
    volume_ratio=float(row.get("Volume_Ratio_20", np.nan))
    if pd.notna(close) and close > float(row.get("SMA_200", np.nan)):
        trend="Uzun vadeli yukselis"
    elif pd.notna(close) and close < float(row.get("SMA_200", np.nan)):
        trend="Uzun vadeli dusus"
    else:
        trend="Uzun vadeli yon belirsiz"
    if pd.isna(adx):
        momentum="ADX icin veri yetersiz"
    elif adx >= 25:
        momentum="Belirgin trend"
    else:
        momentum="Zayif veya yatay trend"
    if pd.isna(atr_pct):
        volatility="Volatilite verisi yetersiz"
    elif atr_pct >= 0.05:
        volatility="Yuksek volatilite"
    elif atr_pct <= 0.015:
        volatility="Dusuk volatilite"
    else:
        volatility="Orta volatilite"
    if pd.isna(rsi):
        momentum += "; RSI verisi yetersiz"
    elif rsi >= 70:
        momentum += "; RSI asiri alima yakin"
    elif rsi <= 30:
        momentum += "; RSI asiri satima yakin"
    if pd.notna(volume_ratio) and volume_ratio >= 1.5:
        volume_state="Hacim canli"
    else:
        volume_state="Hacim normal veya dusuk"
    return {"Trend Rejimi":trend, "Momentum Rejimi":momentum, "Volatilite Rejimi":volatility, "Hacim Rejimi":volume_state}


def score_bucket_backtest(df, threshold=60, horizons=(5, 10)):
    """Agirlikli skor esigini kullanan tarihsel sinyallerin yon isabetini hesaplar."""
    scored=[]
    for _, row in df.iterrows():
        scored.append(score_signal_row(row))
    score_table=pd.DataFrame(scored, index=df.index)
    close=df["Close"].astype(float)
    report=[]
    for horizon in horizons:
        forward=close.shift(-horizon)/close-1
        buy_mask=(score_table["buy_score"] >= threshold) & (score_table["buy_score"] > score_table["sell_score"])
        sell_mask=(score_table["sell_score"] >= threshold) & (score_table["sell_score"] > score_table["buy_score"])
        for direction, mask, returns in (("AL", buy_mask, forward), ("SAT", sell_mask, -forward)):
            observed=returns[mask].dropna()
            report.append({
                "Vade":horizon,
                "Yon":direction,
                "Esik":threshold,
                "Sinyal":int(observed.size),
                "Yon Isabeti (%)":round(float((observed>0).mean()*100),1) if len(observed) else None,
                "Ortalama Hareket (%)":round(float(observed.mean()*100),2) if len(observed) else None,
                "En Iyi (%)":round(float(observed.max()*100),2) if len(observed) else None,
                "En Kotu (%)":round(float(observed.min()*100),2) if len(observed) else None,
            })
    return pd.DataFrame(report)


def make_scan_csv(scan_rows):
    """Tarama sonuclarini indirilebilir UTF-8 CSV olarak hazirlar."""
    if not scan_rows:
        return b""
    frame=pd.DataFrame(scan_rows)
    keep=[column for column in ["Hisse", "Sinyal", "Teknik Sinyal", "AL Puanı", "SAT Puanı", "Son Fiyat", "Trend Yönü", "Model Başarı Oranı (%)"] if column in frame.columns]
    if not keep:
        keep=list(frame.columns)
    return frame[keep].to_csv(index=False).encode("utf-8-sig")


def add_advanced_indicators(df):
    """Sinyal puanlamasi icin ek trend, momentum, para akisi ve volatilite olcutleri."""
    df = df.copy()
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float).fillna(0)

    # Stochastic oscillator
    stoch_low = low.rolling(14, min_periods=14).min()
    stoch_high = high.rolling(14, min_periods=14).max()
    df["Stoch_K"] = 100 * (close - stoch_low) / (stoch_high - stoch_low + 1e-9)
    df["Stoch_D"] = df["Stoch_K"].rolling(3, min_periods=3).mean()

    # Directional movement and ADX
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    prev_close = close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr14 = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    df["DI_Plus"] = 100 * plus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / (atr14 + 1e-9)
    df["DI_Minus"] = 100 * minus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / (atr14 + 1e-9)
    dx = 100 * (df["DI_Plus"] - df["DI_Minus"]).abs() / (df["DI_Plus"] + df["DI_Minus"] + 1e-9)
    df["ADX"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # Money Flow Index
    typical = (high + low + close) / 3
    raw_flow = typical * volume
    positive_flow = raw_flow.where(typical > typical.shift(1), 0.0).rolling(14, min_periods=14).sum()
    negative_flow = raw_flow.where(typical < typical.shift(1), 0.0).rolling(14, min_periods=14).sum()
    money_ratio = positive_flow / (negative_flow + 1e-9)
    df["MFI"] = 100 - 100 / (1 + money_ratio)

    # Rolling VWAP and rate of change
    df["VWAP_20"] = (typical * volume).rolling(20, min_periods=20).sum() / (volume.rolling(20, min_periods=20).sum() + 1e-9)
    df["ROC_10"] = close.pct_change(10) * 100

    # Donchian levels exclude the current candle
    df["Donchian_Upper_20"] = high.rolling(20, min_periods=20).max().shift(1)
    df["Donchian_Lower_20"] = low.rolling(20, min_periods=20).min().shift(1)

    # Ichimoku conversion/base lines (unshifted values used only as current references)
    df["Ichimoku_Tenkan"] = (high.rolling(9, min_periods=9).max() + low.rolling(9, min_periods=9).min()) / 2
    df["Ichimoku_Kijun"] = (high.rolling(26, min_periods=26).max() + low.rolling(26, min_periods=26).min()) / 2

    # Supertrend direction. Bands are updated bar by bar to avoid future data.
    st_atr = atr14
    midpoint = (high + low) / 2
    basic_upper = midpoint + 3.0 * st_atr
    basic_lower = midpoint - 3.0 * st_atr
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    trend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i == 0 or pd.isna(st_atr.iloc[i]):
            trend.iloc[i] = np.nan
            direction.iloc[i] = 0
            continue
        prev_upper = final_upper.iloc[i - 1]
        prev_lower = final_lower.iloc[i - 1]
        if pd.isna(prev_upper):
            final_upper.iloc[i] = basic_upper.iloc[i]
        elif basic_upper.iloc[i] < prev_upper or close.iloc[i - 1] > prev_upper:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = prev_upper
        if pd.isna(prev_lower):
            final_lower.iloc[i] = basic_lower.iloc[i]
        elif basic_lower.iloc[i] > prev_lower or close.iloc[i - 1] < prev_lower:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = prev_lower
        previous_direction = direction.iloc[i - 1]
        if previous_direction <= 0 and close.iloc[i] > final_upper.iloc[i]:
            direction.iloc[i] = 1
        elif previous_direction >= 0 and close.iloc[i] < final_lower.iloc[i]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = previous_direction
        trend.iloc[i] = final_lower.iloc[i] if direction.iloc[i] > 0 else final_upper.iloc[i]
    df["Supertrend"] = trend
    df["Supertrend_Direction"] = direction

    return df


def score_signal_row(row):
    """Tek bir mum icin agirlikli AL/SAT puanini ve gerekceleri dondurur."""
    buy_score = 0
    sell_score = 0
    buy_reasons = []
    sell_reasons = []

    def value(name, default=np.nan):
        item = row.get(name, default)
        try:
            return float(item) if pd.notna(item) else default
        except (TypeError, ValueError):
            return default

    def add(condition, points, side, reason):
        nonlocal buy_score, sell_score
        if condition:
            if side == "buy":
                buy_score += points
                buy_reasons.append(f"+{points}: {reason}")
            else:
                sell_score += points
                sell_reasons.append(f"+{points}: {reason}")

    close = value("Close")
    add(value("EMA_9") > value("EMA_21"), 9, "buy", "EMA 9, EMA 21 uzerinde")
    add(value("EMA_9") < value("EMA_21"), 9, "sell", "EMA 9, EMA 21 altinda")
    add(close > value("SMA_200"), 9, "buy", "Fiyat 200 gunluk ortalama uzerinde")
    add(close < value("SMA_200"), 9, "sell", "Fiyat 200 gunluk ortalama altinda")
    add(value("MACD") > value("MACD_Signal"), 8, "buy", "MACD sinyal cizgisinin uzerinde")
    add(value("MACD") < value("MACD_Signal"), 8, "sell", "MACD sinyal cizgisinin altinda")
    add(50 <= value("RSI", 50) <= 70, 7, "buy", "RSI yukselis bolgesinde")
    add(value("RSI", 50) < 50, 7, "sell", "RSI 50 altinda")
    add(value("ADX") >= 20 and value("DI_Plus") > value("DI_Minus"), 8, "buy", "ADX trendi ve DI+ ustunlugu")
    add(value("ADX") >= 20 and value("DI_Minus") > value("DI_Plus"), 8, "sell", "ADX trendi ve DI- ustunlugu")
    add(value("CMF") > 0, 6, "buy", "CMF para girisine isaret ediyor")
    add(value("CMF") < 0, 6, "sell", "CMF para cikisina isaret ediyor")
    add(value("MFI", 50) > 50, 5, "buy", "MFI alici tarafinda")
    add(value("MFI", 50) < 50, 5, "sell", "MFI satici tarafinda")
    add(close > value("VWAP_20"), 7, "buy", "Fiyat 20 gunluk VWAP uzerinde")
    add(close < value("VWAP_20"), 7, "sell", "Fiyat 20 gunluk VWAP altinda")
    add(value("Supertrend_Direction") > 0, 8, "buy", "Supertrend yukari yonlu")
    add(value("Supertrend_Direction") < 0, 8, "sell", "Supertrend asagi yonlu")
    add(value("Ichimoku_Tenkan") > value("Ichimoku_Kijun"), 5, "buy", "Ichimoku donus cizgisi baz cizginin ustunde")
    add(value("Ichimoku_Tenkan") < value("Ichimoku_Kijun"), 5, "sell", "Ichimoku donus cizgisi baz cizginin altinda")
    smart_long = row.get("Smart_Long", False)
    smart_short = row.get("Smart_Short", False)
    institutional_sell = row.get("Institutional_Sell", False)
    smart_long = bool(smart_long) if pd.notna(smart_long) else False
    smart_short = bool(smart_short) if pd.notna(smart_short) else False
    institutional_sell = bool(institutional_sell) if pd.notna(institutional_sell) else False
    add(smart_long, 12, "buy", "Smart Market Structure yukari kirilimi")
    add(smart_short, 12, "sell", "Smart Market Structure asagi kirilimi")
    add(institutional_sell, 12, "sell", "Kurumsal alt likidite kirilimi")
    add(value("Volume_Ratio_20", 0) >= 1.5 and close > value("Close", close), 4, "buy", "Hacim ortalamanin 1.5 kati")
    add(value("Breakout_Down", 0) == 1, 4, "sell", "Yuksek hacimli 20 seans asagi kirilimi")

    buy_score = min(int(buy_score), 100)
    sell_score = min(int(sell_score), 100)
    return {
        "buy_score": buy_score,
        "sell_score": sell_score,
        "buy_reasons": buy_reasons,
        "sell_reasons": sell_reasons,
    }


def build_signal_backtest(df, horizons=(5, 10)):
    """Gecmis AL/SAT sinyallerinin sonraki kapanislardaki yon performansini ozetler."""
    rows = []
    close = df["Close"].astype(float)
    for horizon in horizons:
        future_return = close.shift(-horizon) / close - 1
        buy_returns = future_return[df["Merged_Long"].fillna(False)].dropna()
        sell_returns = -future_return[df["Merged_Short"].fillna(False)].dropna()
        for label, values in (("AL", buy_returns), ("SAT", sell_returns)):
            rows.append({
                "Vade (seans)": horizon,
                "Yon": label,
                "Sinyal Sayisi": int(len(values)),
                "Yon Isabeti (%)": round(float((values > 0).mean() * 100), 1) if len(values) else None,
                "Ortalama Hareket (%)": round(float(values.mean() * 100), 2) if len(values) else None,
                "Medyan Hareket (%)": round(float(values.median() * 100), 2) if len(values) else None,
            })
    return pd.DataFrame(rows)


def format_signal_reasons(reasons, empty_text="Bu mumda puan kazandiran kosul yok."):
    """Sinyal gerekcelerini Streamlit ve Telegram icin okunabilir metne cevirir."""
    return "\n".join(reasons) if reasons else empty_text


def analyze_breakout_levels(df):
    """Kırılımı trend, momentum ve hacim doğrulamalarıyla sınıflandırır."""
    last = df.iloc[-1]
    close = float(last["Close"])
    resistance = last.get("Breakout_Resistance_20", np.nan)
    support = last.get("Breakout_Support_20", np.nan)
    atr = float(last.get("ATR", np.nan))
    volume_ratio = float(last.get("Volume_Ratio_20", 0.0)) if pd.notna(last.get("Volume_Ratio_20", np.nan)) else 0.0
    if pd.isna(resistance) or pd.isna(support):
        return {"status": "Yetersiz veri", "resistance": None, "support": None,
                "volume_ratio": None, "confirmations": 0, "checks": {}}

    resistance, support = float(resistance), float(support)
    checks = {
        "20 seans direnci üzerinde kapanış": close > resistance,
        "Hacim en az 1,5 kat": volume_ratio >= 1.5,
        "EMA 9, EMA 21 üzerinde": float(last["EMA_9"]) > float(last["EMA_21"]),
        "Fiyat 200 günlük ortalama üzerinde": close > float(last["SMA_200"]),
        "MACD sinyal üzerinde ve pozitif": float(last["MACD"]) > float(last["MACD_Signal"]) and float(last["MACD"]) > 0,
        "RSI 50-70 bandında": 50 <= float(last["RSI"]) <= 70,
        "CMF pozitif": float(last["CMF"]) > 0,
    }
    passed = sum(checks.values())
    if all(checks.values()):
        status = "GÜÇLÜ YÜKSELİŞ TEYİDİ (7/7)"
    elif checks["20 seans direnci üzerinde kapanış"] and checks["Hacim en az 1,5 kat"] and passed >= 5:
        status = f"YÜKSELİŞ TEYİDİ ( {passed}/7 )"
    elif close > resistance:
        status = f"KIRILIM VAR, TEYİT EKSİK ({passed}/7)"
    elif close < support:
        status = "AŞAĞI KIRILIM"
    elif close >= resistance - 0.5 * atr:
        status = "DİRENÇ BÖLGESİNE YAKIN"
    elif close <= support + 0.5 * atr:
        status = "DESTEK BÖLGESİNE YAKIN"
    else:
        status = "20 SEANSLIK BANT İÇİNDE"
    return {"status": status, "resistance": resistance, "support": support,
            "volume_ratio": volume_ratio, "confirmations": passed, "checks": checks}

# ==============================================================================
# 6. GELİŞMİŞ PLOTLY GÖRSELLEŞTİRME HARİTASI
# ==============================================================================
def render_custom_plotly_chart(df, symbol_name):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f"{symbol_name} Fiyat, Bollinger & Hareketli Ortalamalar",
            "Hacim & CMF (Chaikin Money Flow)",
            "MACD & İvme Osilatörü"
        ),
        row_width=[0.20, 0.20, 0.60],
    )

    # 1. Row: Candlestick & Overlays
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Fiyat",
        ),
        row=1, col=1,
    )

    if "EMA_9" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_9"], line=dict(color="cyan", width=1), name="EMA 9"), row=1, col=1)
    if "EMA_21" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_21"], line=dict(color="yellow", width=1), name="EMA 21"), row=1, col=1)
    if "SMA_200" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"], line=dict(color="orange", width=1.5), name="200 SMA"), row=1, col=1)
    
    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], line=dict(color="gray", width=0.8, dash="dash"), name="BB Üst"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], line=dict(color="gray", width=0.8, dash="dash"), name="BB Alt"), row=1, col=1)
    if "Breakout_Resistance_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["Breakout_Resistance_20"], line=dict(color="magenta", width=1, dash="dot"), name="20 Seans Direnci"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Breakout_Support_20"], line=dict(color="lime", width=1, dash="dot"), name="20 Seans Desteği"), row=1, col=1)
    if "Breakout_Up" in df.columns and df["Breakout_Up"].any():
        points = df[df["Breakout_Up"]]
        fig.add_trace(go.Scatter(x=points.index, y=points["Close"], mode="markers", marker=dict(color="lime", size=11, symbol="triangle-up"), name="Hacimli Yukarı Kırılım"), row=1, col=1)
    if "Breakout_Down" in df.columns and df["Breakout_Down"].any():
        points = df[df["Breakout_Down"]]
        fig.add_trace(go.Scatter(x=points.index, y=points["Close"], mode="markers", marker=dict(color="red", size=11, symbol="triangle-down"), name="Hacimli Aşağı Kırılım"), row=1, col=1)

    # 2. Row: Hacim & CMF
    vol_colors = ["green" if c >= o else "red" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Hacim"), row=2, col=1)
    if "CMF" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["CMF"] * df["Volume"].max(), line=dict(color="purple", width=1.2), name="CMF Ölçekli"), row=2, col=1)

    # 3. Row: MACD
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="lightblue", width=1.2), name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], line=dict(color="coral", width=1.2), name="Sinyal"), row=3, col=1)
        hist_colors = ["green" if val >= 0 else "red" for val in df["MACD_Hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], marker_color=hist_colors, name="Histogram"), row=3, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=800,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 7. YAPAY ZEKA MODELLERİ VE TAHMİN MOTORU
# ==============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def analiz_hesapla(symbol_input, model_tercihi="XGBoost", is_bist=True, sermaye_input=100000, signal_threshold=60):
    df = get_stock_data_hybrid(symbol_input, is_bist=is_bist)
    df = clean_market_data(df)
    if df is None or len(df) < 150:
        return None, "Canlı veri çekilemedi veya veri hacmi yetersiz."

    data_is_valid, data_quality_notes = validate_market_data(df)
    if not data_is_valid:
        return None, "Piyasa verisi kullanilabilir degil: " + "; ".join(data_quality_notes)

    # Indikator processing
    df_ind = calculate_indicators(df)
    df_ind = add_advanced_indicators(df_ind)

    # Hedef Oluşturma (%0.5 üzeri 1 gün sonrası artış)
    df_ind["Target"] = np.where(df_ind["Close"].shift(-1) > df_ind["Close"] * 1.005, 1, 0)
    df_cleaned = df_ind.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df_cleaned) < 150:
        return None, "Indikatorlar sonrasi kullanilabilir gecmis 150 mumdan az."

    features = [
        "Return", "RSI", "Stoch_RSI", "MACD", "MACD_Hist",
        "ATR_PCT", "BB_Width", "Williams_R", "CMF", "Stoch_K",
        "Stoch_D", "ADX", "DI_Plus", "DI_Minus", "MFI", "ROC_10",
        "Supertrend_Direction"
    ]
    
    X = df_cleaned[features]
    y = df_cleaned["Target"]

    # Zaman Serisine Uygun Train / Test Ayrımı
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:-1]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:-1]

    # Model Seçimi ve Eğitimi
    if model_tercihi == "XGBoost":
        model = XGBClassifier(
            n_estimators=180, 
            max_depth=4, 
            learning_rate=0.03, 
            random_state=42, 
            eval_metric="logloss"
        )
    else:
        model = RandomForestClassifier(
            n_estimators=180, 
            max_depth=5, 
            random_state=42
        )

    model.fit(X_train, y_train)

    # Performans Metrikleri
    test_preds = model.predict(X_test)
    acc_score = accuracy_score(y_test, test_preds) * 100
    prec_score = precision_score(y_test, test_preds, zero_division=0) * 100
    rec_score = recall_score(y_test, test_preds, zero_division=0) * 100
    f1 = f1_score(y_test, test_preds, zero_division=0) * 100

    # Güncel Veri Tahmini
    latest_data = X.iloc[[-1]]
    prediction = model.predict(latest_data)[0]
    prob = model.predict_proba(latest_data)[0]

    latest_close = float(df_cleaned["Close"].iloc[-1])
    prev_close = float(df_cleaned["Close"].iloc[-2])
    change_pct = ((latest_close - prev_close) / prev_close) * 100

    latest_ema9 = float(df_cleaned["EMA_9"].iloc[-1])
    latest_ema21 = float(df_cleaned["EMA_21"].iloc[-1])
    latest_sma200 = float(df_cleaned["SMA_200"].iloc[-1])
    latest_atr = float(df_cleaned["ATR"].iloc[-1])
    kirilim = analyze_breakout_levels(df_cleaned)
    merged_signal = "AL" if bool(df_cleaned["Merged_Long"].iloc[-1]) else ("SAT" if bool(df_cleaned["Merged_Short"].iloc[-1]) else "SİNYAL YOK")
    signal_scores = score_signal_row(df_cleaned.iloc[-1])
    if signal_scores["buy_score"] >= signal_threshold and signal_scores["buy_score"] > signal_scores["sell_score"]:
        technical_signal = "GÜÇLÜ AL" if signal_scores["buy_score"] >= 75 else "AL ADAYI"
    elif signal_scores["sell_score"] >= signal_threshold and signal_scores["sell_score"] > signal_scores["buy_score"]:
        technical_signal = "GÜÇLÜ SAT" if signal_scores["sell_score"] >= 75 else "SAT ADAYI"
    else:
        technical_signal = "BEKLE"
    signal_history = score_bucket_backtest(df_cleaned, threshold=signal_threshold)
    hit_5d = signal_history[(signal_history["Vade"] == 5) & signal_history["Yon Isabeti (%)"].notna()]
    hit_5d_rate = float(hit_5d["Yon Isabeti (%)"].mean()) if not hit_5d.empty else None
    regime = market_regime_snapshot(df_cleaned.iloc[-1])

    # Trend Yönü Kararı
    if latest_close > latest_sma200 and latest_ema9 > latest_ema21:
        trend_durumu = "GÜÇLÜ YÜKSELİŞ (BOĞA)"
    elif latest_close < latest_sma200 and latest_ema9 < latest_ema21:
        trend_durumu = "GÜÇLÜ DÜŞÜŞ (AYI)"
    else:
        trend_durumu = "YATAY / KARARSIZ"

    # Net Sinyal Üretimi
    if prediction == 1 and prob[1] >= 0.62 and trend_durumu == "GÜÇLÜ YÜKSELİŞ (BOĞA)" and kirilim["confirmations"] == 7:
        net_sinyal = "🚀 GÜÇLÜ AL (7/7 TEYİTLİ)"
    elif prediction == 1 and prob[1] >= 0.55 and kirilim["confirmations"] >= 5 and kirilim["checks"].get("20 seans direnci üzerinde kapanış", False) and kirilim["checks"].get("Hacim en az 1,5 kat", False):
        net_sinyal = f"📈 AL (TEYİTLİ {kirilim['confirmations']}/7)"
    elif prediction == 0 and prob[0] > 0.62:
        net_sinyal = "🛑 GÜÇLÜ SAT / BEKLE"
    else:
        net_sinyal = "⏳ TEYİT YOK / BEKLE"

    # Fiyat Projeksiyonları
    direction_factor = (prob[1] - 0.5) * 2
    est_1d = latest_close + (direction_factor * latest_atr * 0.8)
    est_1w = latest_close + (direction_factor * latest_atr * 2.5)

    # Pivot Seviyeleri
    latest_high = float(df_cleaned["High"].iloc[-1])
    latest_low = float(df_cleaned["Low"].iloc[-1])
    pivot = (latest_high + latest_low + latest_close) / 3.0
    support_1 = (2 * pivot) - latest_high
    resistance_1 = (2 * pivot) - latest_low
    stop_loss = round(latest_close - (1.5 * latest_atr), 2)
    take_profit = round(latest_close + (2.5 * latest_atr), 2)

    # Risk Yönetimi & Pozisyon Büyüklüğü
    risk_tutari = sermaye_input * 0.02  # Portföyün %2 riski
    hisse_risk = max(latest_close - stop_loss, 0.01)
    alınabilir_adet = int(risk_tutari / hisse_risk)
    toplam_pozisyon_degeri = round(alınabilir_adet * latest_close, 2)

    ozet = {
        "Hisse": symbol_input.replace(".IS", ""),
        "Son Fiyat": round(latest_close, 2),
        "Günlük Değişim (%)": round(change_pct, 2),
        "Sinyal": net_sinyal,
        "Birleşik Yapı Sinyali": merged_signal,
        "Teknik Sinyal": technical_signal,
        "AL Puanı": signal_scores["buy_score"],
        "SAT Puanı": signal_scores["sell_score"],
        "AL Gerekçeleri": format_signal_reasons(signal_scores["buy_reasons"]),
        "SAT Gerekçeleri": format_signal_reasons(signal_scores["sell_reasons"]),
        "5 Seans Yön İsabeti (%)": round(hit_5d_rate, 1) if hit_5d_rate is not None else None,
        "Sinyal Performans Özeti": signal_history,
        "Piyasa Rejimi": regime,
        "Veri Kalitesi Notları": data_quality_notes,
        "Trend Yönü": trend_durumu,
        "Yükseliş İhtimali (%)": round(prob[1] * 100, 1),
        "Model Başarı Oranı (%)": round(acc_score, 1),
        "Sinyal Hassasiyeti (%)": round(prec_score, 1),
        "Recall (%)": round(rec_score, 1),
        "F1 Skor (%)": round(f1, 1),
        "Tahmin 1 Gun": round(est_1d, 2),
        "Tahmin 1 Hafta": round(est_1w, 2),
        "İdeal Alış": round(latest_close - (0.5 * latest_atr), 2),
        "Stop-Loss": stop_loss,
        "Kar Al (Take Profit)": take_profit,
        "Destek S1": round(support_1, 2),
        "Direnç R1": round(resistance_1, 2),
        "Kırılım Analizi": kirilim["status"],
        "20 Seans Direnci": round(kirilim["resistance"], 2) if kirilim["resistance"] is not None else None,
        "20 Seans Desteği": round(kirilim["support"], 2) if kirilim["support"] is not None else None,
        "Hacim Ortalamasına Oran": round(kirilim["volume_ratio"], 2) if kirilim["volume_ratio"] is not None else None,
        "Yükseliş Teyit Sayısı": kirilim["confirmations"],
        "Yükseliş Teyitleri": kirilim["checks"],
        "Onerilen Adet": alınabilir_adet,
        "Pozisyon Maliyeti": toplam_pozisyon_degeri
    }

    return df_cleaned, ozet

# ==============================================================================
# 8. YAN MENÜ (SIDEBAR) VE AYARLAR
# ==============================================================================
st.sidebar.header("⚙️ Genel Sistem Ayarları")

secilen_model = st.sidebar.selectbox("🤖 Algoritma Tipi", ["XGBoost", "Random Forest"])
piyasa = st.sidebar.radio("Piyasa Seçimi", ["BIST (Türk Borsası)", "ABD Borsaları"])
sermaye = st.sidebar.number_input("Toplam Portföy Büyüklüğü:", value=100000, step=10000)
sinyal_esigi = st.sidebar.slider("Teknik Sinyal Minimum Puanı", 40, 85, 60, 5)

if piyasa == "BIST (Türk Borsası)":
    varsayilan_hisse = "ASELS"
    para_birimi = "TL"
    is_bist_flag = True
else:
    varsayilan_hisse = "AAPL"
    para_birimi = "$"
    is_bist_flag = False

st.sidebar.divider()
st.sidebar.subheader("📲 Telegram Entegrasyonu")
telegram_token = st.sidebar.text_input("Bot Token:", value=DEFAULT_TELEGRAM_TOKEN, type="password")
telegram_chat_id = st.sidebar.text_input("Chat ID:", value=DEFAULT_TELEGRAM_CHAT_ID)

if st.sidebar.button("🔔 Telegram Bağlantı Testi"):
    ok, msg = send_telegram_message(
        telegram_token,
        telegram_chat_id,
        "🚀 *PRO BIST Platformu*\nTelegram bildirim servisi aktif!",
    )
    if ok:
        st.sidebar.success("Test mesajı iletildi!")
    else:
        st.sidebar.error(f"Hata: {msg}")

# ==============================================================================
# 9. ARAYÜZ VE SEKMELER
# ==============================================================================
tab_analiz, tab_toplu, tab_risk = st.tabs([
    "🔍 Tekil Hisse & Derin Analiz", 
    "📊 Toplu BIST Taraması", 
    "🛡️ Risk & Portföy Yönetimi"
])

# ------------------------------------------------------------------------------
# TAB 1: TEKİL HİSSE DERİN ANALİZ
# ------------------------------------------------------------------------------
with tab_analiz:
    hisse_kod = st.text_input(
        "Hisse Kodunu Girin (Örn: THYAO, EREGL, NVDA, ASTOR):", 
        value=varsayilan_hisse
    ).strip().upper()

    if hisse_kod:
        with st.spinner(f"{hisse_kod} detaylı yapay zeka analizinden geçiyor..."):
            df_data, ozet_veri = analiz_hesapla(
                hisse_kod, 
                model_tercihi=secilen_model, 
                is_bist=is_bist_flag,
                sermaye_input=sermaye,
                signal_threshold=sinyal_esigi
            )

        if df_data is None:
            st.error(f"Hata: {ozet_veri}")
        else:
            # Üst Metrik Kartları
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Son Fiyat", f"{ozet_veri['Son Fiyat']} {para_birimi}", delta=f"%{ozet_veri['Günlük Değişim (%)']}")
            c2.metric("Sinyal Durumu", ozet_veri["Sinyal"])
            c3.metric("Trend Filtresi", ozet_veri["Trend Yönü"])
            c4.metric("Teknik Sinyal", ozet_veri["Teknik Sinyal"], delta=f"AL {ozet_veri['AL Puanı']} / SAT {ozet_veri['SAT Puanı']}")

            st.subheader("🧭 Teknik Sinyal Puanlaması")
            sc1, sc2 = st.columns(2)
            sc1.metric("AL puanı", f"{ozet_veri['AL Puanı']}/100")
            sc2.metric("SAT puanı", f"{ozet_veri['SAT Puanı']}/100")
            reason_left, reason_right = st.columns(2)
            with reason_left:
                st.markdown("**AL gerekçeleri**")
                st.markdown(ozet_veri["AL Gerekçeleri"].replace("\n", "  \n"))
            with reason_right:
                st.markdown("**SAT gerekçeleri**")
                st.markdown(ozet_veri["SAT Gerekçeleri"].replace("\n", "  \n"))
            st.caption(f"AL/SAT aday eşiği: {sinyal_esigi}/100. Puanlar teknik koşulların ağırlıklı toplamıdır; olasılık veya getiri garantisi değildir.")

            st.divider()
            st.subheader("📚 Geçmiş Sinyal Performansı")
            perf = ozet_veri["Sinyal Performans Özeti"]
            st.caption("Geçmiş AL/SAT olaylarından sonra fiyatın belirtilen vadede sinyal yönünde hareket etme oranı; işlem maliyetleri dahil değildir.")
            st.dataframe(perf, use_container_width=True, hide_index=True)
            st.write("**Piyasa rejimi:**", " · ".join(ozet_veri["Piyasa Rejimi"].values()))
            if ozet_veri["Veri Kalitesi Notları"]:
                with st.expander("Veri kalitesi notlari"):
                    st.write("\n".join(ozet_veri["Veri Kalitesi Notları"]))

            # Tahmin Seviyeleri
            col_t1, col_t2, col_t3 = st.columns(3)
            col_t1.info(f"**Yükseliş Olasılığı:** `%{ozet_veri['Yükseliş İhtimali (%)']}`")
            col_t2.success(f"**1 Günlük Tahmin:** `{ozet_veri['Tahmin 1 Gun']} {para_birimi}`")
            col_t3.success(f"**1 Haftalık Tahmin:** `{ozet_veri['Tahmin 1 Hafta']} {para_birimi}`")

            # Risk ve Destek/Direnç
            col_d1, col_d2, col_d3 = st.columns(3)
            col_d1.warning(f"**Kar Al (Take Profit):** `{ozet_veri['Kar Al (Take Profit)']} {para_birimi}`")
            col_d2.error(f"**Stop-Loss Seviyesi:** `{ozet_veri['Stop-Loss']} {para_birimi}`")
            col_d3.markdown(f"**Destek S1 / Direnç R1:** `{ozet_veri['Destek S1']} / {ozet_veri['Direnç R1']}`")

            st.divider()

            st.subheader("📍 Önemli Kırılım ve Yükseliş Teyitleri")
            k1, k2, k3 = st.columns(3)
            k1.metric("20 Seans Direnci", f"{ozet_veri['20 Seans Direnci']} {para_birimi}" if ozet_veri["20 Seans Direnci"] is not None else "—")
            k2.metric("20 Seans Desteği", f"{ozet_veri['20 Seans Desteği']} {para_birimi}" if ozet_veri["20 Seans Desteği"] is not None else "—")
            k3.metric("Kırılım Durumu", ozet_veri["Kırılım Analizi"], delta=f"Hacim/ortalama: {ozet_veri['Hacim Ortalamasına Oran']}x" if ozet_veri["Hacim Ortalamasına Oran"] is not None else None)
            st.write(f"**Yükseliş teyidi:** {ozet_veri['Yükseliş Teyit Sayısı']}/7 koşul")
            teyit_sol, teyit_sag = st.columns(2)
            teyit_listesi = list(ozet_veri["Yükseliş Teyitleri"].items())
            for idx, (ad, gecti) in enumerate(teyit_listesi):
                hedef = teyit_sol if idx < 4 else teyit_sag
                hedef.write(f"{'✅' if gecti else '▫️'} {ad}")
            st.caption("Teyit koşulları sinyali seçici yapar, ancak fiyat hareketini kesinleştirmez veya getiri garantisi vermez.")

            # Telegram Sinyal Butonu
            if st.button("📲 Sinyali Telegram Grubuna İlet"):
                mesaj = f"""
🎯 *YAPAY ZEKA DETAYLI ANALİZ RAPORU*
📈 *Hisse:* `{ozet_veri['Hisse']}`
💵 *Son Fiyat:* `{ozet_veri['Son Fiyat']} {para_birimi}` (%{ozet_veri['Günlük Değişim (%)']})
🚥 *Sinyal:* *{ozet_veri['Sinyal']}*
🧭 *Trend Yönü:* `{ozet_veri['Trend Yönü']}`
📊 *Backtest Doğruluk:* `%{ozet_veri['Model Başarı Oranı (%)']}`
📍 *Kırılım:* `{ozet_veri['Kırılım Analizi']}`
✅ *Yükseliş teyidi:* `{ozet_veri['Yükseliş Teyit Sayısı']}/7`
🔺 *20 Seans direnci:* `{ozet_veri['20 Seans Direnci']}`
🔻 *20 Seans desteği:* `{ozet_veri['20 Seans Desteği']}`

🔮 *1 Günlük Hedef:* `{ozet_veri['Tahmin 1 Gun']} {para_birimi}`
🔮 *1 Haftalık Hedef:* `{ozet_veri['Tahmin 1 Hafta']} {para_birimi}`
🎯 *Kar Al:* `{ozet_veri['Kar Al (Take Profit)']} {para_birimi}`
🛑 *Stop-Loss:* `{ozet_veri['Stop-Loss']} {para_birimi}`
🛡️ *Önerilen Adet:* `{ozet_veri['Onerilen Adet']}` Adet
                """
                ok, res = send_telegram_message(telegram_token, telegram_chat_id, mesaj)
                if ok:
                    st.success("Analiz Telegram'a aktarıldı!")
                else:
                    st.error(f"Aktarım Hatası: {res}")

            st.divider()
            render_custom_plotly_chart(df_data, ozet_veri["Hisse"])

# ------------------------------------------------------------------------------
# TAB 2: TOPLU BIST TARAMA SEKMESİ
# ------------------------------------------------------------------------------
with tab_toplu:
    st.subheader("📊 Otomatik BİST Tarama Paneli")
    
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    liste_metni = ", ".join(BIST_TUM_POPULER)
    
    if col_k1.button("🏆 Tüm Popüler BİST"):
        liste_metni = ", ".join(BIST_TUM_POPULER)
    if col_k2.button("🌟 BIST 30 Hisseleri"):
        liste_metni = ", ".join(BIST_30)
    if col_k3.button("🏦 Bankacılık Sektörü"):
        liste_metni = ", ".join(BIST_BANKA)
    if col_k4.button("🏭 Sanayi ve Teknoloji"):
        liste_metni = ", ".join(BIST_SANAYI_TEKNO)

    girilen_hisseler = st.text_area("Taranacak Sembol Listesi:", value=liste_metni, height=120)

    auto_telegram = st.checkbox("⚡ '🚀 GÜÇLÜ AL' sinyallerini anında Telegram'a gönder", value=True)

    if st.button("🔍 Sinyal Taramasını Başlat", type="primary"):
        h_list = normalize_symbol_list(girilen_hisseler)
        if not h_list:
            st.warning("Tarama icin en az bir sembol girin.")
            st.stop()
        tarama_sonuc = []
        bar = st.progress(0)

        for idx, h in enumerate(h_list):
            _, oz = analiz_hesapla(
                h, 
                model_tercihi=secilen_model, 
                is_bist=is_bist_flag, 
                sermaye_input=sermaye,
                signal_threshold=sinyal_esigi
            )
            if oz and isinstance(oz, dict):
                tarama_sonuc.append(oz)

                if auto_telegram and "GÜÇLÜ AL" in oz["Sinyal"]:
                    msg = (
                        f"⚡ *GÜÇLÜ AL SİNYALİ YAKALANDI:* `{oz['Hisse']}`\n"
                        f"💵 Fiyat: `{oz['Son Fiyat']}` {para_birimi} | Trend: `{oz['Trend Yönü']}`\n"
                        f"🔮 1 Günlük Tahmin: `{oz['Tahmin 1 Gun']}` | Başarı Oranı: `%{oz['Model Başarı Oranı (%)']}`\n"
                        f"🛑 Stop: `{oz['Stop-Loss']}` | 🎯 Kar Al: `{oz['Kar Al (Take Profit)']}`"
                    )
                    send_telegram_message(telegram_token, telegram_chat_id, msg)

            bar.progress((idx + 1) / len(h_list))

        bar.empty()
        if tarama_sonuc:
            df_res = pd.DataFrame(tarama_sonuc).sort_values(by="Yükseliş İhtimali (%)", ascending=False)
            sutunlar = [
                "Hisse", "Sinyal", "Teknik Sinyal", "AL Puanı", "SAT Puanı", "Trend Yönü", "Yükseliş İhtimali (%)", 
                "Model Başarı Oranı (%)", "Sinyal Hassasiyeti (%)", "Yükseliş Teyit Sayısı", "Kırılım Analizi", "Son Fiyat", 
                "Tahmin 1 Gun", "Kar Al (Take Profit)", "Stop-Loss"
            ]
            st.dataframe(df_res[sutunlar], use_container_width=True)
            st.download_button("⬇️ Tarama CSV indir", data=make_scan_csv(tarama_sonuc), file_name="sinyal_taramasi.csv", mime="text/csv")

# ------------------------------------------------------------------------------
# TAB 3: RİSK VE PORTFÖY HESAPLAYICI
# ------------------------------------------------------------------------------
with tab_risk:
    st.subheader("🛡️ Otomatik Pozisyon ve Risk Büyüklüğü Yönetimi")
    st.write(
        "Portföyünüzün tek bir işlemde yüksek kayıplara uğramasını engellemek amacıyla, "
        "ATR bazlı stop seviyesine göre işlem büyüklüğü %2 risk kuralı ile hesaplanır."
    )

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        risk_hisse = st.text_input("Hesaplanacak Hisse Kodunu Girin:", value="THYAO").strip().upper()
        risk_sermaye = st.number_input("Kullanılabilir Toplam Sermaye (TL/$):", value=sermaye, step=5000)
        max_risk_orani = st.slider("İşlem Başı Maksimum Risk Oranı (%)", 0.5, 5.0, 2.0, 0.1)

    if risk_hisse:
        _, r_ozet = analiz_hesapla(risk_hisse, model_tercihi=secilen_model, is_bist=is_bist_flag, sermaye_input=risk_sermaye, signal_threshold=sinyal_esigi)
        if r_ozet and isinstance(r_ozet, dict):
            with r_col2:
                st.info(f"**Hisse:** `{r_ozet['Hisse']}`")
                st.write(f"**Giriş Fiyatı:** `{r_ozet['Son Fiyat']}` {para_birimi}")
                st.write(f"**Hesaplanan Stop Loss:** `{r_ozet['Stop-Loss']}` {para_birimi}")
                
                riske_edilen_para = risk_sermaye * (max_risk_orani / 100)
                birim_risk = max(r_ozet['Son Fiyat'] - r_ozet['Stop-Loss'], 0.01)
                onerilen_adet = int(riske_edilen_para / birim_risk)
                toplam_tutar = onerilen_adet * r_ozet['Son Fiyat']

                st.success(f"📌 **Önerilen İşlem Adedi:** `{onerilen_adet}` Adet")
                st.warning(f"💼 **Gerekli Toplam Pozisyon Büyüklüğü:** `{round(toplam_tutar, 2)}` {para_birimi}")
                st.error(f"🛑 **Maksimum Göze Alınan Kayıp:** `{round(riske_edilen_para, 2)}` {para_birimi}")
        else:
            st.error(f"Hata: {r_ozet}")

