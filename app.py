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

    return df


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
def analiz_hesapla(symbol_input, model_tercihi="XGBoost", is_bist=True, sermaye_input=100000):
    df = get_stock_data_hybrid(symbol_input, is_bist=is_bist)
    if df is None or len(df) < 150:
        return None, "Canlı veri çekilemedi veya veri hacmi yetersiz."

    # İndikatör İşleme
    df_ind = calculate_indicators(df)

    # Hedef Oluşturma (%0.5 üzeri 1 gün sonrası artış)
    df_ind["Target"] = np.where(df_ind["Close"].shift(-1) > df_ind["Close"] * 1.005, 1, 0)
    df_cleaned = df_ind.replace([np.inf, -np.inf], np.nan).dropna()

    features = [
        "Return", "RSI", "Stoch_RSI", "MACD", "MACD_Hist", 
        "ATR_PCT", "BB_Width", "Williams_R", "CMF"
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
                sermaye_input=sermaye
            )

        if df_data is None:
            st.error(f"Hata: {ozet_veri}")
        else:
            # Üst Metrik Kartları
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Son Fiyat", f"{ozet_veri['Son Fiyat']} {para_birimi}", delta=f"%{ozet_veri['Günlük Değişim (%)']}")
            c2.metric("Sinyal Durumu", ozet_veri["Sinyal"])
            c3.metric("Trend Filtresi", ozet_veri["Trend Yönü"])
            c4.metric("Backtest Başarısı", f"%{ozet_veri['Model Başarı Oranı (%)']}")

            st.divider()

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
        h_list = [h.strip().upper() for h in girilen_hisseler.split(",") if h.strip()]
        tarama_sonuc = []
        bar = st.progress(0)

        for idx, h in enumerate(h_list):
            _, oz = analiz_hesapla(
                h, 
                model_tercihi=secilen_model, 
                is_bist=is_bist_flag, 
                sermaye_input=sermaye
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
                "Hisse", "Sinyal", "Trend Yönü", "Yükseliş İhtimali (%)", 
                "Model Başarı Oranı (%)", "Sinyal Hassasiyeti (%)", "Yükseliş Teyit Sayısı", "Kırılım Analizi", "Son Fiyat", 
                "Tahmin 1 Gun", "Kar Al (Take Profit)", "Stop-Loss"
            ]
            st.dataframe(df_res[sutunlar], use_container_width=True)

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
        _, r_ozet = analiz_hesapla(risk_hisse, model_tercihi=secilen_model, is_bist=is_bist_flag, sermaye_input=risk_sermaye)
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
