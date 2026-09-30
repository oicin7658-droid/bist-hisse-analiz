import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="PRO BIST & US Yapay Zeka & Canlı Grafik Platformu",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- SESSION STATE (TAKİP LİSTESİ İÇİN MEMORY) ---
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["ASELS", "THYAO", "TUPRS", "EREGL", "GARAN"]

# --- BAŞLIK ---
st.title("⚡ PRO Hisse Analiz & Canlı Takip Platformu")
st.caption(
    "Yapay Zeka Sinyalleri | TradingView Canlı Grafikleri | Kişisel Takip Listesi | Toplu Tarama"
)
st.divider()


# --- TRADINGVIEW HTML WIDGET FONKSİYONU ---
def render_tradingview_widget(symbol_code, is_bist=True):
    # TradingView BIST için 'BIST:ASELS', ABD için 'NASDAQ:AAPL' veya 'NYSE:TSLA' bekler
    tv_symbol = f"BIST:{symbol_code}" if is_bist else symbol_code

    html_code = f"""
    <!-- TradingView Widget BEGIN -->
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
    <!-- TradingView Widget END -->
    """
    components.html(html_code, height=560)


# --- TEKNİK ANALİZ VE YAPAY ZEKA FONKSİYONU ---
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

        # İndikatörler
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

        # Target
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
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
            )
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
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

# --- TAKİP LİSTESİ YÖNETİMİ ---
st.sidebar.divider()
st.sidebar.subheader("📌 Takip Listem")
yeni_hisse = st.sidebar.text_input("Listeye Hisse Ekle:").strip().upper()
if st.sidebar.button("➕ Ekle") and yeni_hisse:
    if yeni_hisse not in st.session_state.watchlist:
        st.session_state.watchlist.append(yeni_hisse)
        st.sidebar.success(f"{yeni_hisse} eklendi!")

st.sidebar.caption("Mevcut Listeniz: " + ", ".join(st.session_state.watchlist))
if st.sidebar.button("🗑️ Listeyi Temizle"):
    st.session_state.watchlist = []
    st.rerun()


# --- SEKMELER (TABS) ---
tab_analiz, tab_watchlist, tab_toplu = st.tabs(
    [
        "🔍 Tekil Hisse & Canlı Grafik",
        "⭐ Takip Listem Canlı Özet",
        "📊 Toplu BIST Taraması",
    ]
)

# ==============================================================================
# SEKME 1: TEKİL HİSSE VE CANLI TRADINGVIEW GRAFİĞİ
# ==============================================================================
with tab_analiz:
    hisse_kod = (
        st.text_input(
            "Hisse Sembolü Girin (Örn: THYAO, EREGL, NVDA, TSLA):",
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
            # Metrik Kartları
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

            st.markdown(
                f"**🤖 Model ({secilen_model}) Tahmin Güveni:** `%{ozet_veri['Yükseliş İhtimali (%)']}` | **RSI:** `{ozet_veri['RSI']}` | **StochRSI:** `{ozet_veri['StochRSI']}`"
            )
            st.divider()

            # GRAFİK TÜRÜ SEÇİMİ
            grafik_turu = st.radio(
                "Grafik Türünü Seçin:",
                [
                    "📈 TradingView Canlı & İnteraktif Grafik",
                    "📊 Plotly İndikatörlü Teknik Grafik",
                ],
                horizontal=True,
            )

            if "TradingView" in grafik_turu:
                st.subheader(f"📺 TradingView Canlı Ekranı - {hisse_kod}")
                render_tradingview_widget(hisse_kod, is_bist=is_bist_flag)

            else:
                # Plotly İndikatörlü Grafik
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.7, 0.3],
                )
                fig.add_trace(
                    go.Candlestick(
                        x=df_data.index,
                        open=df_data["Open"],
                        high=df_data["High"],
                        low=df_data["Low"],
                        close=df_data["Close"],
                        name="Fiyat",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_data.index,
                        y=df_data["SMA_200"],
                        line=dict(color="orange", width=1.5),
                        name="200 SMA",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_hline(
                    y=ozet_veri["İdeal Alış"],
                    line_dash="dash",
                    line_color="green",
                    annotation_text="İdeal Alış",
                    row=1,
                    col=1,
                )
                fig.add_hline(
                    y=ozet_veri["Destek S1"],
                    line_dash="dot",
                    line_color="blue",
                    annotation_text="Destek S1",
                    row=1,
                    col=1,
                )
                fig.add_hline(
                    y=ozet_veri["Direnç R1"],
                    line_dash="dot",
                    line_color="red",
                    annotation_text="Direnç R1",
                    row=1,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=df_data.index,
                        y=df_data["RSI"],
                        line=dict(color="purple", width=1.5),
                        name="RSI (14)",
                    ),
                    row=2,
                    col=1,
                )
                fig.add_hline(
                    y=70, line_dash="dash", line_color="red", row=2, col=1
                )
                fig.add_hline(
                    y=30, line_dash="dash", line_color="green", row=2, col=1
                )

                fig.update_layout(
                    height=550,
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# SEKME 2: TAKİP LİSTEM CANLI ÖZET
# ==============================================================================
with tab_watchlist:
    st.subheader("⭐ Kişisel Takip Listenizdeki Hisseler")

    if not st.session_state.watchlist:
        st.info(
            "Takip listeniz henüz boş. Sol taraftaki menüden hisse ekleyebilirsiniz."
        )
    else:
        wl_sonuclar = []
        with st.spinner("Takip listeniz güncelleniyor..."):
            for h in st.session_state.watchlist:
                _, ozet = analiz_hesapla(
                    h,
                    model_tercihi=secilen_model,
                    is_bist=is_bist_flag,
                    start_date="2021-01-01",
                )
                if ozet and isinstance(ozet, dict):
                    wl_sonuclar.append(ozet)

        if wl_sonuclar:
            df_wl = pd.DataFrame(wl_sonuclar)

            st.dataframe(
                df_wl[
                    [
                        "Hisse",
                        "Son Fiyat",
                        "Günlük Değişim (%)",
                        "İdeal Alış",
                        "Sinyal",
                        "Yükseliş İhtimali (%)",
                        "RSI",
                    ]
                ].style.highlight_max(
                    axis=0,
                    subset=["Yükseliş İhtimali (%)"],
                    color="darkgreen",
                ),
                use_container_width=True,
            )

# ==============================================================================
# SEKME 3: TOPLU BIST TARAMASI
# ==============================================================================
with tab_toplu:
    st.subheader("📊 Toplu Hisse Taraması")
    varsayilan_metin = "ASELS, TUPRS, THYAO, GARAN, AKBNK, EREGL, BIMAS, SISE, KCHOL, SAHOL, YKBNK, PETKM"
    girilen_hisseler = st.text_area(
        "Taranacak Hisse Kodları:", value=varsayilan_metin, height=100
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
                is_bist=True,
                start_date="2021-01-01",
            )
            if oz and isinstance(oz, dict):
                tarama_sonuc.append(oz)
            bar.progress((idx + 1) / len(h_list))

        bar.empty()
        if tarama_sonuc:
            df_res = pd.DataFrame(tarama_sonuc).sort_values(
                by="Yükseliş İhtimali (%)", ascending=False
            )
            st.dataframe(df_res, use_container_width=True)
