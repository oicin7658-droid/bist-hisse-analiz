import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# --- SABİT TELEGRAM BİLGİLERİNİZ ---
DEFAULT_TELEGRAM_TOKEN = "8898496727:AAEaArWqlYX92vLGfJUW1nHzUL-cWFC2otQ"
DEFAULT_TELEGRAM_CHAT_ID = "1840616371"

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="PRO BIST & US Yapay Zeka Platformu",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- SESSION STATE ---
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["ASELS", "THYAO", "TUPRS", "EREGL", "GARAN"]

# --- BAŞLIK ---
st.title("⚡ PRO Hisse Analiz & Canlı Takip Platformu")
st.caption(
    "Yapay Zeka Sinyalleri | TradingView Canlı Grafikleri | Telegram Entegre Sistem"
)
st.divider()


# --- TELEGRAM MESAJ GÖNDERME FONKSİYONU ---
def send_telegram_message(bot_token, chat_id, message):
    """Telegram Bot API üzerinden belirlenen Chat ID'ye mesaj gönderir."""
    if not bot_token or not chat_id:
        return False, "Bot Token veya Chat ID eksik."

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}

    try:
        response = requests.post(url, json=payload, timeout=5)
        res_data = response.json()
        if res_data.get("ok"):
            return True, "Başarılı"
        else:
            return False, res_data.get("description", "Bilinmeyen hata")
    except Exception as e:
        return False, str(e)


# --- TRADINGVIEW HTML WIDGET FONKSİYONU ---
def render_tradingview_widget(symbol_code, is_bist=True):
    tv_symbol = f"BIST:{symbol_code}" if is_bist else symbol_code

    html_code = f"""
    <div class="tradingview-widget-container" style="height:100%;width:100%">
      <div id="tradingview_chart" style="height:550px;width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "autosize": true,
        "symbol": "{tv_symbol}",
        "interval": "D",
        "timezone": "Europe/Istanbul",
        "theme": "dark",
        "style": "1",
        "locale": "tr",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_chart"
      }}
      );
      </script>
    </div>
    """
    components.html(html_code, height=560)


# --- TEKNİK ANALİZ VE YAPAY ZEKA FONKSİYONU ---
@st.cache_data(ttl=3600, show_spinner=False)
def analiz_hesapla(
    symbol_input,
    model_tercihi="XGBoost",
    is_bist=True,
    start_date="2021-01-01",
):
    symbol = (
        f"{symbol_input}.IS"
        if is_bist and not symbol_input.endswith(".IS")
        else symbol_input
    )

    try:
        df = yf.download(symbol, start=start_date, progress=False)
        if df.empty or len(df) < 200:
            return None, "Yetersiz veri veya geçersiz hisse kodu."

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Temel İndikatörler
        df["Return"] = df["Close"].pct_change()
        df["Volume_Change"] = df["Volume"].pct_change()
        df["Vol_SMA_Ratio"] = df["Volume"] / (
            df["Volume"].rolling(window=20).mean() + 1e-9
        )

        # RSI & Stoch RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))

        rsi_min = df["RSI"].rolling(window=14).min()
        rsi_max = df["RSI"].rolling(window=14).max()
        df["Stoch_RSI"] = (df["RSI"] - rsi_min) / (rsi_max - rsi_min + 1e-9)

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
        df["Bollinger_PCT"] = (df["Close"] - df["Lower_Band"]) / (
            df["Upper_Band"] - df["Lower_Band"] + 1e-9
        )

        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df["ATR"] = true_range.rolling(14).mean()
        df["ATR_PCT"] = df["ATR"] / df["Close"]

        # 200 SMA
        df["SMA_200"] = df["Close"].rolling(window=200).mean()
        df["Trend_200_Ratio"] = df["Close"] / (df["SMA_200"] + 1e-9)

        # Target (Yarınki kapanış bugünkünden yüksek mi?)
        df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
        df_cleaned = df.replace([np.inf, -np.inf], np.nan).dropna()

        features = [
            "Return",
            "Volume_Change",
            "Vol_SMA_Ratio",
            "RSI",
            "Stoch_RSI",
            "MACD",
            "MACD_Hist",
            "Bollinger_PCT",
            "ATR_PCT",
            "Trend_200_Ratio",
        ]

        X = df_cleaned[features]
        y = df_cleaned["Target"]

        if model_tercihi == "XGBoost":
            model = XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
            )
        else:
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
            )

        model.fit(X.iloc[:-1], y.iloc[:-1])

        latest_data = X.iloc[[-1]]
        prediction = model.predict(latest_data)[0]
        prob = model.predict_proba(latest_data)[0]

        latest_close = float(df_cleaned["Close"].iloc[-1])
        prev_close = float(df_cleaned["Close"].iloc[-2])
        change_pct = ((latest_close - prev_close) / prev_close) * 100

        latest_high = float(df_cleaned["High"].iloc[-1])
        latest_low = float(df_cleaned["Low"].iloc[-1])
        latest_stoch = float(df_cleaned["Stoch_RSI"].iloc[-1])
        latest_atr = float(df_cleaned["ATR"].iloc[-1])
        trend_ok = float(df_cleaned["Trend_200_Ratio"].iloc[-1]) > 0.98

        pivot = (latest_high + latest_low + latest_close) / 3.0
        support_1 = (2 * pivot) - latest_high
        resistance_1 = (2 * pivot) - latest_low

        ideal_entry = round(latest_close - (0.5 * latest_atr), 2)
        if ideal_entry < support_1:
            ideal_entry = round(support_1, 2)

        stop_loss = round(ideal_entry - (1.5 * latest_atr), 2)

        sinyal = (
            "GÜÇLÜ AL"
            if (prediction == 1 and latest_stoch < 0.85 and trend_ok)
            else "NÖTR / BEKLE"
        )

        ozet = {
            "Hisse": symbol_input.replace(".IS", ""),
            "Model": model_tercihi,
            "Son Fiyat": round(latest_close, 2),
            "Günlük Değişim (%)": round(change_pct, 2),
            "İdeal Alış": ideal_entry,
            "Destek S1": round(support_1, 2),
            "Direnç R1": round(resistance_1, 2),
            "Stop-Loss": stop_loss,
            "Sinyal": sinyal,
            "Yükseliş İhtimali (%)": round(prob[1] * 100, 1),
            "RSI": round(float(df_cleaned["RSI"].iloc[-1]), 1),
            "StochRSI": round(latest_stoch, 2),
        }

        return df_cleaned, ozet
    except Exception as e:
        return None, str(e)


# --- YAN MENÜ (SIDEBAR) ---
st.sidebar.header("⚙️ Genel Ayarlar")

secilen_model = st.sidebar.selectbox(
    "🤖 Yapay Zeka Modeli", ["XGBoost", "Random Forest"]
)

piyasa = st.sidebar.radio(
    "Piyasa Seçimi", ["BIST (Türk Borsası)", "ABD Borsaları (S&P 500 / Nasdaq)"]
)

if piyasa == "BIST (Türk Borsası)":
    varsayilan_hisse = "ASELS"
    para_birimi = "TL"
    is_bist_flag = True
else:
    varsayilan_hisse = "AAPL"
    para_birimi = "$"
    is_bist_flag = False

# --- TELEGRAM AYARLARI ---
st.sidebar.divider()
st.sidebar.subheader("📲 Telegram Bildirim Ayarları")
telegram_token = st.sidebar.text_input(
    "Bot Token:", value=DEFAULT_TELEGRAM_TOKEN, type="password"
)
telegram_chat_id = st.sidebar.text_input(
    "Chat ID:", value=DEFAULT_TELEGRAM_CHAT_ID
)

if st.sidebar.button("🔔 Test Mesajı Gönder"):
    ok, msg = send_telegram_message(
        telegram_token,
        telegram_chat_id,
        "🚀 *PRO BIST & US Platformu*\nTelegram bot bağlantınız başarıyla sağlandı!",
    )
    if ok:
        st.sidebar.success("Test mesajı Telegram hesabınıza gönderildi!")
    else:
        st.sidebar.error(f"Mesaj gönderilemedi: {msg}")

# --- SEKMELER ---
tab_analiz, tab_watchlist, tab_toplu = st.tabs(
    [
        "🔍 Tekil Hisse & Canlı Grafik",
        "⭐ Takip Listem Canlı Özet",
        "📊 Toplu BIST Taraması",
    ]
)

# ==============================================================================
# SEKME 1: TEKİL HİSSE ANALİZİ
# ==============================================================================
with tab_analiz:
    hisse_kod = (
        st.text_input(
            "Hisse Sembolü Girin (Örn: THYAO, EREGL, NVDA):",
            value=varsayilan_hisse,
        )
        .strip()
        .upper()
    )

    if hisse_kod:
        with st.spinner(f"{hisse_kod} analiz ediliyor..."):
            df_data, ozet_veri = analiz_hesapla(
                hisse_kod,
                model_tercihi=secilen_model,
                is_bist=is_bist_flag,
                start_date="2021-01-01",
            )

        if df_data is None:
            st.error(f"Hata: {ozet_veri}")
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(
                "Son Fiyat",
                f"{ozet_veri['Son Fiyat']} {para_birimi}",
                delta=f"%{ozet_veri['Günlük Değişim (%)']}",
            )
            c2.metric(
                "İdeal Alış Fiyatı", f"{ozet_veri['İdeal Alış']} {para_birimi}"
            )
            c3.metric(
                "Destek / Direnç",
                f"{ozet_veri['Destek S1']} / {ozet_veri['Direnç R1']}",
            )
            c4.metric(
                "Stop-Loss", f"{ozet_veri['Stop-Loss']} {para_birimi}"
            )
            c5.metric("Model Sinyali", ozet_veri["Sinyal"])

            # Telegram Sinyal Butonu
            if st.button("📲 Bu Analizi Telegram'a Gönder"):
                mesaj = f"""
🎯 *YAPAY ZEKA HİSSE SİNYALİ*
📈 *Hisse:* `{ozet_veri['Hisse']}`
💵 *Son Fiyat:* `{ozet_veri['Son Fiyat']} {para_birimi}` (%{ozet_veri['Günlük Değişim (%)']})
🚥 *Sinyal:* *{ozet_veri['Sinyal']}*
📊 *Yükseliş İhtimali:* `%{ozet_veri['Yükseliş İhtimali (%)']}`

🎯 *İdeal Alış:* `{ozet_veri['İdeal Alış']} {para_birimi}`
🛑 *Stop-Loss:* `{ozet_veri['Stop-Loss']} {para_birimi}`
🛡️ *Destek:* `{ozet_veri['Destek S1']}` | *Direnç:* `{ozet_veri['Direnç R1']}`
                """
                ok, res = send_telegram_message(
                    telegram_token, telegram_chat_id, mesaj
                )
                if ok:
                    st.success("Analiz Telegram hesabınıza gönderildi!")
                else:
                    st.error(f"Gönderilemedi: {res}")

            st.divider()
            render_tradingview_widget(hisse_kod, is_bist=is_bist_flag)

# ==============================================================================
# SEKME 3: TOPLU TARAMA VE OTOMATİK TELEGRAM BİLDİRİMİ
# ==============================================================================
with tab_toplu:
    st.subheader("📊 Toplu Hisse Taraması & Otomatik Sinyal Gönderimi")
    varsayilan_metin = "ASELS, TUPRS, THYAO, GARAN, AKBNK, EREGL, BIMAS, SISE, KCHOL, SAHOL, YKBNK, PETKM"
    girilen_hisseler = st.text_area(
        "Taranacak Hisse Kodları:", value=varsayilan_metin, height=100
    )

    auto_telegram = st.checkbox(
        "⚡ Tarama sırasında yakalanan 'GÜÇLÜ AL' sinyallerini otomatik Telegram'a gönder",
        value=True,
    )

    if st.button("🔍 Taramayı Başlat", type="primary"):
        h_list = [
            h.strip().upper() for h in girilen_hisseler.split(",") if h.strip()
        ]
        tarama_sonuc = []
        bar = st.progress(0)

        for idx, h in enumerate(h_list):
            _, oz = analiz_hesapla(
                h,
                model_tercihi=secilen_model,
                is_bist=is_bist_flag,
                start_date="2021-01-01",
            )
            if oz and isinstance(oz, dict):
                tarama_sonuc.append(oz)

                # Otomatik Telegram Bildirimi
                if auto_telegram and oz["Sinyal"] == "GÜÇLÜ AL":
                    msg = f"⚡ *GÜÇLÜ AL SİNYALİ:* `{oz['Hisse']}`\n💵 Fiyat: `{oz['Son Fiyat']}` | 🎯 İdeal Alış: `{oz['İdeal Alış']}`\n📊 Yükseliş İhtimali: `%{oz['Yükseliş İhtimali (%)']}`"
                    send_telegram_message(
                        telegram_token, telegram_chat_id, msg
                    )

            bar.progress((idx + 1) / len(h_list))

        bar.empty()
        if tarama_sonuc:
            df_res = pd.DataFrame(tarama_sonuc).sort_values(
                by="Yükseliş İhtimali (%)", ascending=False
            )
            st.dataframe(df_res, use_container_width=True)
