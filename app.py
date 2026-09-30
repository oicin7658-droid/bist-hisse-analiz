import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# Sayfa Yapılandırması
st.set_page_config(
    page_title="BIST & US Hisse Analiz Platformu",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- BAŞLIK VE AÇIKLAMA ---
st.title("📈 Yapay Zeka Destekli Hisse Analiz Platformu")
st.markdown(
    "Makine Öğrenmesi (Random Forest), Gelişmiş Teknik İndikatörler ve Pivot Seviyeleri ile Hisse Analizi"
)
st.divider()

# --- ÖRNEK BIST POPÜLER HİSSE LİSTESİ ---
VARSAYILAN_BIST_LISTESI = [
    "ASELS",
    "TUPRS",
    "THYAO",
    "GARAN",
    "AKBNK",
    "EREGL",
    "BIMAS",
    "SISE",
    "KCHOL",
    "SAHOL",
    "YKBNK",
    "PETKM",
]


# --- TEKNİK HESAPLAMA VE MODEL FONKSİYONU ---
def analiz_hesapla(symbol_input, is_bist=True, start_date="2021-01-01"):
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

        # Getiri ve Hacim Oranları
        df["Return"] = df["Close"].pct_change()
        df["Volume_Change"] = df["Volume"].pct_change()
        df["Vol_SMA_Ratio"] = df["Volume"] / (
            df["Volume"].rolling(window=20).mean() + 1e-9
        )

        # 1. RSI (14)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))

        # 2. Stochastic RSI
        rsi_min = df["RSI"].rolling(window=14).min()
        rsi_max = df["RSI"].rolling(window=14).max()
        df["Stoch_RSI"] = (df["RSI"] - rsi_min) / (rsi_max - rsi_min + 1e-9)

        # 3. MACD
        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema_12 - ema_26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        # 4. Bollinger Bands
        sma_20 = df["Close"].rolling(window=20).mean()
        std_20 = df["Close"].rolling(window=20).std()
        df["Upper_Band"] = sma_20 + (std_20 * 2)
        df["Lower_Band"] = sma_20 - (std_20 * 2)
        df["Bollinger_PCT"] = (df["Close"] - df["Lower_Band"]) / (
            df["Upper_Band"] - df["Lower_Band"] + 1e-9
        )

        # 5. ATR (Volatilite)
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df["ATR"] = true_range.rolling(14).mean()
        df["ATR_PCT"] = df["ATR"] / df["Close"]

        # 6. Trend Oranı (200 Günlük Ortalama)
        df["SMA_200"] = df["Close"].rolling(window=200).mean()
        df["Trend_200_Ratio"] = df["Close"] / (df["SMA_200"] + 1e-9)

        # Target (Gelecek Gün Pozitif mi?)
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
            "Son Fiyat": round(latest_close, 2),
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


# --- SEKMELER (TABS) ---
tab1, tab2 = st.tabs(
    ["🔍 Tekil Hisse Detaylı Analizi", "📊 Toplu Hisse Taraması"]
)

# ==============================================================================
# SEKME 1: TEKİL HİSSE ANALİZİ
# ==============================================================================
with tab1:
    st.sidebar.header("⚙️ Tekil Analiz Parametreleri")

    piyasa = st.sidebar.radio(
        "Piyasa Seçimi",
        ["BIST (Türk Borsası)", "ABD Borsaları (S&P 500 / Nasdaq)"],
    )

    if piyasa == "BIST (Türk Borsası)":
        varsayilan_hisse = "ASELS"
        para_birimi = "TL"
    else:
        varsayilan_hisse = "AAPL"
        para_birimi = "$"

    hisse_kod = (
        st.sidebar.text_input("Hisse Sembolü (Kod):", value=varsayilan_hisse)
        .strip()
        .upper()
    )

    baslangic_tarihi = st.sidebar.date_input(
        "Veri Başlangıç Tarihi", pd.to_datetime("2021-01-01")
    )

    analiz_butonu = st.sidebar.button("🚀 Analizi Başlat", type="primary")

    if analiz_butonu or hisse_kod:
        is_bist_market = piyasa == "BIST (Türk Borsası)"
        with st.spinner(f"{hisse_kod} hissesi analiz ediliyor..."):
            df_data, ozet_veri = analiz_hesapla(
                hisse_kod, is_bist_market, baslangic_tarihi
            )

        if df_data is None:
            st.error(f"Hata: {ozet_veri}")
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(
                "Son Fiyat", f"{ozet_veri['Son Fiyat']} {para_birimi}"
            )
            c2.metric(
                "İdeal Alış Fiyatı", f"{ozet_veri['İdeal Alış']} {para_birimi}"
            )
            c3.metric(
                "Destek (S1) / Direnç (R1)",
                f"{ozet_veri['Destek S1']} / {ozet_veri['Direnç R1']}",
            )
            c4.metric(
                "Stop-Loss", f"{ozet_veri['Stop-Loss']} {para_birimi}"
            )
            c5.metric("Model Sinyali", ozet_veri["Sinyal"])

            st.subheader(
                f"🎯 Tahmin Güveni: %{ozet_veri['Yükseliş İhtimali (%)']} | RSI: {ozet_veri['RSI']} | StochRSI: {ozet_veri['StochRSI']}"
            )
            st.divider()

            # Plotly Grafiği
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
# SEKME 2: TOPLU HİSSE TARAMASI
# ==============================================================================
with tab2:
    st.subheader("📋 Çoklu BIST Hisse Taraması")
    st.markdown(
        "Aşağıdaki listeyi düzenleyebilir veya doğrudan **'Taramayı Başlat'** butonuna basarak tüm hisseleri analiz edebilirsiniz."
    )

    varsayilan_metin = ", ".join(VARSAYILAN_BIST_LISTESI)
    girilen_hisseler = st.text_area(
        "Taranacak Hisse Kodları (Virgülle Ayırın):",
        value=varsayilan_metin,
        height=100,
    )

    tarama_butonu = st.button("🔍 Taramayı Başlat", type="primary")

    if tarama_butonu:
        hisse_listesi = [
            h.strip().upper() for h in girilen_hisseler.split(",") if h.strip()
        ]

        if not hisse_listesi:
            st.warning("Lütfen en az bir hisse kodu girin.")
        else:
            sonuclar = []
            ilerleme_cubugu = st.progress(0)
            durum_metni = st.empty()

            for i, h_kodu in enumerate(hisse_listesi):
                durum_metni.text(
                    f"[{i+1}/{len(hisse_listesi)}] Analiz Ediliyor: {h_kodu}..."
                )
                _, ozet = analiz_hesapla(
                    h_kodu, is_bist=True, start_date="2021-01-01"
                )

                if ozet and isinstance(ozet, dict):
                    sonuclar.append(ozet)

                ilerleme_cubugu.progress((i + 1) / len(hisse_listesi))

            durum_metni.empty()
            ilerleme_cubugu.empty()

            if sonuclar:
                df_tarama = pd.DataFrame(sonuclar)

                # Yükseliş İhtimaline göre sırala
                df_tarama = df_tarama.sort_values(
                    by="Yükseliş İhtimali (%)", ascending=False
                )

                # Sadece GÜÇLÜ AL Verenleri Filtreleme Seçeneği
                sadece_al = st.checkbox(
                    "Sadece 'GÜÇLÜ AL' Sinyali Verenleri Göster", value=False
                )
                if sadece_al:
                    df_tarama_goster = df_tarama[
                        df_tarama["Sinyal"] == "GÜÇLÜ AL"
                    ]
                else:
                    df_tarama_goster = df_tarama

                st.success(
                    f"Tarama Tamamlandı! Toplam {len(df_tarama_goster)} hisse listelendi."
                )

                # Tablo Gösterimi
                st.dataframe(
                    df_tarama_goster.style.highlight_max(
                        axis=0,
                        subset=["Yükseliş İhtimali (%)"],
                        color="darkgreen",
                    ),
                    use_container_width=True,
                )

                # CSV İndirme
                csv_tarama = df_tarama.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 Toplu Tarama Sonuçlarını İndir (CSV)",
                    data=csv_tarama,
                    file_name="toplu_bist_tarama_sonuclari.csv",
                    mime="text/csv",
                )
            else:
                st.error("Hiçbir hisse için veri çekilemedi.")