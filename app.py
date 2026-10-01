import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
import streamlit as st
import yfinance as yf
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# --- SABİT TELEGRAM BİLGİLERİNİZ ---
DEFAULT_TELEGRAM_TOKEN = "8898496727:AAEaArWqlYX92vLGfJUW1nHzUL-cWFC2otQ"
DEFAULT_TELEGRAM_CHAT_ID = "1840616371"

# --- GENİŞ BİST HİSSE LİSTELERİ ---
BIST_30 = [
    "AKBNK", "ALARK", "ASELS", "BIMAS", "BRSAN", "DOAS", "EKGYO", "ENKAI", 
    "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", "KCHOL", "KONTR", 
    "KOZAL", "KRDMD", "ODAS", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", 
    "TCELL", "THYAO", "TOASO", "TUPRS", "YKBNK", "ASTOR"
]

BIST_BANKA = ["AKBNK", "GARAN", "ISCTR", "YKBNK", "VAKBN", "HALKB", "SKBNK", "TSKB"]

BIST_SANAYI_TEKNO = [
    "ASELS", "EREGL", "TUPRS", "FROTO", "TOASO", "SASA", "SISE", "BRSAN", 
    "ASTOR", "KONTR", "MIATK", "REEDR", "KCAER", "HEKTS", "ALARK", "VESBE"
]

BIST_TUM_POPULER = list(set(BIST_30 + BIST_BANKA + BIST_SANAYI_TEKNO))

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="PRO BIST & US Yapay Zeka Platformu",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- BAŞLIK ---
st.title("⚡ PRO BİST Yüksek Hassasiyetli Yapay Zeka Platformu")
st.caption(
    "Trend Öngörü Algoritması | Model Başarı Oranı (Backtest) | Alternatif Canlı Veri Akışı"
)
st.divider()


# --- ALTERNATİF CANLI VERİ ÇEKME (STOOQ / YFINANCE HYBRID) ---
def get_stock_data_hybrid(symbol_input, is_bist=True):
    """BİST verilerini öncelikle yfinance, hata durumunda Stooq API üzerinden çeker."""
    symbol = f"{symbol_input}.IS" if is_bist and not symbol_input.endswith(".IS") else symbol_input
    
    # 1. Yöntem: Yahoo Finance
    try:
        df = yf.download(symbol, period="3y", progress=False)
        if df is not None and not df.empty and len(df) > 150:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception:
        pass

    # 2. Yöntem (Yedek): Stooq API (Özellikle BIST ve ABD için anlık veri akışı)
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


# --- TELEGRAM MESAJ GÖNDERME FONKSİYONU ---
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


# --- ÖZEL PLOTLY CANLI GRAFİK FONKSİYONU ---
def render_custom_plotly_chart(df, symbol_name):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f"{symbol_name} Canlı Fiyat, EMA & Trend Kanalları", "MACD & İvme"),
        row_width=[0.25, 0.75],
    )

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

    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="blue", width=1.5), name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], line=dict(color="red", width=1.5), name="Sinyal"), row=2, col=1)
        colors = ["green" if val >= 0 else "red" for val in df["MACD_Hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], marker_color=colors, name="Histogram"), row=2, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# --- TEKNİK ANALİZ, GELİŞMİŞ TREND VE BAŞARI ORANI HESAPLAMA ---
@st.cache_data(ttl=1800, show_spinner=False)
def analiz_hesapla(symbol_input, model_tercihi="XGBoost", is_bist=True):
    df = get_stock_data_hybrid(symbol_input, is_bist=is_bist)
    if df is None or len(df) < 150:
        return None, "Canlı veri çekilemedi veya veri hacmi yetersiz."

    # İndikatörler
    df["Return"] = df["Close"].pct_change()
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    # RSI & Stoch RSI
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

    # ATR & Trend Filtreleri
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df["ATR"] = np.max(ranges, axis=1).rolling(14).mean()
    df["ATR_PCT"] = df["ATR"] / df["Close"]

    # Hedef Değişken (Yarın %1'den fazla yükselecek mi?)
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"] * 1.005, 1, 0)
    df_cleaned = df.replace([np.inf, -np.inf], np.nan).dropna()

    features = ["Return", "RSI", "Stoch_RSI", "MACD", "MACD_Hist", "ATR_PCT"]
    X = df_cleaned[features]
    y = df_cleaned["Target"]

    # Zaman Serisi Split (Geçmiş Başarı Oranı Testi - Backtest)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:-1]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:-1]

    if model_tercihi == "XGBoost":
        model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, random_state=42, eval_metric="logloss")
    else:
        model = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)

    model.fit(X_train, y_train)

    # Başarı Metrikleri Hesaplama
    test_preds = model.predict(X_test)
    acc_score = accuracy_score(y_test, test_preds) * 100
    prec_score = precision_score(y_test, test_preds, zero_division=0) * 100

    # Son Durum Tahmini
    latest_data = X.iloc[[-1]]
    prediction = model.predict(latest_data)[0]
    prob = model.predict_proba(latest_data)[0]

    latest_close = float(df_cleaned["Close"].iloc[-1])
    latest_ema9 = float(df_cleaned["EMA_9"].iloc[-1])
    latest_ema21 = float(df_cleaned["EMA_21"].iloc[-1])
    latest_sma200 = float(df_cleaned["SMA_200"].iloc[-1])
    latest_stoch = float(df_cleaned["Stoch_RSI"].iloc[-1])
    latest_atr = float(df_cleaned["ATR"].iloc[-1])

    # Trend Durumu
    if latest_close > latest_sma200 and latest_ema9 > latest_ema21:
        trend_durumu = "GÜÇLÜ YÜKSELİŞ (BOĞA)"
    elif latest_close < latest_sma200 and latest_ema9 < latest_ema21:
        trend_durumu = "GÜÇLÜ DÜŞÜŞ (AYI)"
    else:
        trend_durumu = "YATAY / KARARSIZ"

    # Sinyal Güçlendirme Matrisi
    if prediction == 1 and prob[1] > 0.65 and trend_durumu == "GÜÇLÜ YÜKSELİŞ (BOĞA)":
        net_sinyal = "🚀 GÜÇLÜ AL"
    elif prediction == 1 and prob[1] > 0.55:
        net_sinyal = "📈 AL (TEDBİRLİ)"
    elif prediction == 0 and prob[0] > 0.65:
        net_sinyal = "🛑 GÜÇLÜ SAT / NAKİTTE KAL"
    else:
        net_sinyal = "⏳ NÖTR / BEKLE"

    # Fiyat Tahminleri
    direction_factor = (prob[1] - 0.5) * 2
    est_1d = latest_close + (direction_factor * latest_atr * 0.8)
    est_1w = latest_close + (direction_factor * latest_atr * 2.5)

    ozet = {
        "Hisse": symbol_input.replace(".IS", ""),
        "Son Fiyat": round(latest_close, 2),
        "Trend Yönü": trend_durumu,
        "Sinyal": net_sinyal,
        "Model Yükseliş İhtimali (%)": round(prob[1] * 100, 1),
        "Geçmiş Model Doğruluğu (%)": round(acc_score, 1),
        "Sinyal Hassasiyeti (%)": round(prec_score, 1),
        "1 Günlük Tahmin": round(est_1d, 2),
        "1 Haftalık Tahmin": round(est_1w, 2),
        "Stop-Loss": round(latest_close - (1.5 * latest_atr), 2),
    }

    return df_cleaned, ozet


# --- YAN MENÜ ---
st.sidebar.header("⚙️ Ayarlar")
secilen_model = st.sidebar.selectbox("🤖 Yapay Zeka Modeli", ["XGBoost", "Random Forest"])
piyasa = st.sidebar.radio("Piyasa Seçimi", ["BIST (Türk Borsası)", "ABD Borsaları"])
is_bist_flag = True if piyasa == "BIST (Türk Borsası)" else False

telegram_token = st.sidebar.text_input("Bot Token:", value=DEFAULT_TELEGRAM_TOKEN, type="password")
telegram_chat_id = st.sidebar.text_input("Chat ID:", value=DEFAULT_TELEGRAM_CHAT_ID)

# --- TEKİL ANALİZ SEKMESİ ---
st.subheader("🔍 Yüksek Hassasiyetli Hisse Analizi")
hisse_kod = st.text_input("Hisse Sembolü Girin (Örn: THYAO, EREGL, NVDA):", value="THYAO").strip().upper()

if hisse_kod:
    with st.spinner(f"{hisse_kod} detaylı analiz ediliyor..."):
        df_data, ozet_veri = analiz_hesapla(hisse_kod, model_tercihi=secilen_model, is_bist=is_bist_flag)

    if df_data is None:
        st.error(f"Hata: {ozet_veri}")
    else:
        # Metrik Kartları
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Son Fiyat", f"{ozet_veri['Son Fiyat']} TL")
        c2.metric("Sinyal Netliği", ozet_veri["Sinyal"])
        c3.metric("Öngörülen Trend", ozet_veri["Trend Yönü"])
        c4.metric("Model Başarı Oranı (Acc)", f"%{ozet_veri['Geçmiş Model Doğruluğu (%)']}")

        st.divider()

        # Detaylı Tahmin & Hassasiyet
        col_t1, col_t2, col_t3 = st.columns(3)
        col_t1.info(f"**Yükseliş İhtimali:** `%{ozet_veri['Model Yükseliş İhtimali (%)']}`")
        col_t2.success(f"**1 Günlük Tahmin:** `{ozet_veri['1 Günlük Tahmin']} TL`")
        col_t3.success(f"**1 Haftalık Tahmin:** `{ozet_veri['1 Haftalık Tahmin']} TL`")

        # Plotly Grafiği
        render_custom_plotly_chart(df_data, ozet_veri["Hisse"])
