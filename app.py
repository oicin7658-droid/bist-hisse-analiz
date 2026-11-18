import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf
from technical_analysis import DOWN, UP, analyze_prices, completed_session_cutoff

# --- SABİT TELEGRAM BİLGİLERİNİZ ---
DEFAULT_TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEFAULT_TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="BIST ve ABD Teknik Analiz",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- BAŞLIK ---
st.title("BIST ve ABD Teknik Analiz")
st.caption(
    "Günlük teknik sinyaller · İnteraktif grafikler · Telegram bildirimleri"
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
    except (requests.RequestException, ValueError):
        return False, "Telegram bağlantısı kurulamadı veya yanıt okunamadı."


# --- ÖZEL INTERAKTİF PLOTLY GRAFİK FONKSİYONU ---
def render_custom_plotly_chart(df, symbol_name):
    """TradingView yerine geçen Candlestick, Bollinger, SMA200 ve MACD Plotly Grafiği."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f"{symbol_name} Fiyat & İndikatörler", "MACD"),
        row_width=[0.25, 0.75],
    )

    # Mum Grafiği (Candlestick)
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Fiyat",
        ),
        row=1,
        col=1,
    )

    # 200 SMA
    if "SMA_200" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_200"],
                line=dict(color="orange", width=1.5),
                name="200 SMA",
            ),
            row=1,
            col=1,
        )

    # Bollinger Bantları
    if "Upper_Band" in df.columns and "Lower_Band" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Upper_Band"],
                line=dict(color="gray", width=1, dash="dash"),
                name="Üst Bollinger",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Lower_Band"],
                line=dict(color="gray", width=1, dash="dash"),
                name="Alt Bollinger",
            ),
            row=1,
            col=1,
        )

    # MACD & Histogram
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                line=dict(color="blue", width=1.5),
                name="MACD",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_Signal"],
                line=dict(color="red", width=1.5),
                name="Sinyal",
            ),
            row=2,
            col=1,
        )
        colors = [
            "green" if val >= 0 else "red" for val in df["MACD_Hist"].fillna(0)
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["MACD_Hist"],
                marker_color=colors,
                name="Histogram",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, width="stretch")


@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(symbol, session_cutoff):
    prices = yf.download(
        symbol, period="5y", interval="1d", auto_adjust=True,
        ignore_tz=True, threads=False, progress=False, timeout=10,
    )
    if prices is None or prices.empty:
        raise ValueError("Fiyat verisi alınamadı. Sembolü ve Yahoo Finance erişimini kontrol edin.")
    return prices.loc[prices.index.date <= session_cutoff]


def analiz_hesapla(symbol, is_bist, uploaded=None):
    if is_bist and not symbol.endswith(".IS"):
        symbol = f"{symbol}.IS"
    try:
        if uploaded is None:
            prices = download_prices(symbol, completed_session_cutoff(is_bist))
        else:
            prices = pd.read_csv(uploaded)
            if "Date" not in prices:
                raise ValueError("CSV dosyasında Date sütunu bulunmalı.")
            prices.index = pd.to_datetime(prices.pop("Date"), format="%Y-%m-%d")
            if prices.index.hasnans:
                raise ValueError("CSV dosyasındaki her satırın geçerli bir tarihi olmalı.")
        history, summary = analyze_prices(prices, is_bist)
    except (ValueError, OSError) as exc:
        return None, str(exc)
    summary["Hisse"] = symbol
    return history, summary


def format_summary(summary, currency):
    return (
        f"*{summary['Hisse']}* | {summary['Son Seans']} kapanışı\n"
        f"Fiyat: {summary['Son Fiyat']} {currency}\n"
        f"Sinyal: {summary['Sinyal']}\n"
        f"RSI: {summary['RSI']} | SMA 200: {summary['SMA 200']} | MACD hist.: {summary['MACD Histogramı']}\n"
        f"Destek / direnç: {summary['Destek S1']} / {summary['Direnç R1']}"
    )


# --- YAN MENÜ (SIDEBAR) ---
st.sidebar.header("⚙️ Genel Ayarlar")
with st.sidebar.expander("Teknik sinyal kuralları", expanded=True):
    st.markdown(
        "**Yükseliş:** Kapanış SMA 200 üstünde, MACD histogramı pozitif ve 50 < RSI ≤ 70.\n\n"
        "**Düşüş:** Kapanış SMA 200 altında, MACD histogramı negatif ve 30 ≤ RSI < 50.\n\n"
        "**Bekle:** Göstergeler bu koşulları birlikte sağlamıyor."
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
tab_analiz, tab_toplu = st.tabs(
    [
        "🔍 Tekil Hisse & İnteraktif Grafik",
        "📊 Toplu Hisse Taraması",
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

    uploaded = st.file_uploader(
        "Günlük fiyat CSV dosyası (isteğe bağlı)", type=["csv"],
        help="Date (YYYY-MM-DD), Open, High, Low, Close ve Volume sütunları gerekir. Seçilen sembole ait düzeltilmiş günlük OHLC verisini yükleyin.",
    )
    if hisse_kod:
        with st.spinner(f"{hisse_kod} analiz ediliyor..."):
            df_data, ozet_veri = analiz_hesapla(hisse_kod, is_bist_flag, uploaded)

        if df_data is None:
            st.error(ozet_veri)
        else:
            st.caption(f"Son tamamlanmış seans {ozet_veri['Son Seans']:%d.%m.%Y} · Günlük, düzeltilmiş fiyatlar")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Son Kapanış",
                f"{ozet_veri['Son Fiyat']} {para_birimi}",
                delta=f"{ozet_veri['Günlük Değişim (%)']}%",
            )
            c2.metric("Destek / Direnç", f"{ozet_veri['Destek S1']} / {ozet_veri['Direnç R1']}")
            c3.metric("ATR (14 seans)", f"{ozet_veri['ATR (14)']} {para_birimi}")
            c4.metric("Teknik Sinyal", ozet_veri["Sinyal"])

            s1, s2, s3 = st.columns(3)
            s1.metric("SMA 200", ozet_veri["SMA 200"])
            s2.metric("MACD histogramı", ozet_veri["MACD Histogramı"])
            s3.metric("RSI (14)", ozet_veri["RSI"])

            test = ozet_veri["Test"]
            if test["signals"]:
                st.caption(
                    f"Geçmiş yön testi · {test['sessions']} değerlendirilebilir seans · "
                    f"{test['signals']} yön sinyalinden {test['hits']} isabet (%{test['accuracy'] * 100:.1f}). "
                    "Her sinyal bir sonraki seansın kapanışına göre değerlendirilir."
                )
            else:
                st.caption("Geçmiş yön testi için sonucu bilinen yükseliş veya düşüş sinyali bulunamadı.")

            if st.button("📲 Bu Analizi Telegram'a Gönder"):
                ok, res = send_telegram_message(
                    telegram_token, telegram_chat_id, format_summary(ozet_veri, para_birimi)
                )
                if ok:
                    st.success("Analiz Telegram hesabınıza gönderildi!")
                else:
                    st.error(f"Gönderilemedi: {res}")

            render_custom_plotly_chart(df_data, ozet_veri["Hisse"])

# ==============================================================================
# SEKME 2: TOPLU TARAMA VE OTOMATİK TELEGRAM BİLDİRİMİ
# ==============================================================================
with tab_toplu:
    st.subheader("📊 Toplu Hisse Taraması & Otomatik Sinyal Gönderimi")
    if is_bist_flag:
        varsayilan_metin = "ASELS, TUPRS, THYAO, GARAN, AKBNK, EREGL, BIMAS, SISE, KCHOL, SAHOL, YKBNK, PETKM"
        piyasa_etiketi = "BIST"
    else:
        # NASDAQ'ta işlem gören, yaygın takip edilen şirketler için başlangıç listesi.
        varsayilan_metin = "AAPL, MSFT, NVDA, AMZN, GOOGL, META, AVGO, TSLA, COST, NFLX, AMD, INTC, QCOM, TXN, AMAT, MU, LRCX, KLAC, ADI, ARM, PLTR, PANW, CRWD, SNPS, CDNS, ASML, MELI, CSCO, AMGN, GILD, ISRG, BKNG, SBUX, MDLZ, PEP, CHTR, TMUS, HON, ADP, CMCSA, REGN, MAR, ORLY, CTAS, DASH, MSTR, APP, COIN, SHOP, ZS"
        piyasa_etiketi = "NASDAQ / ABD"
    girilen_hisseler = st.text_area(
        f"{piyasa_etiketi} için taranacak semboller (virgül veya satır ile ayırın):", value=varsayilan_metin, height=120
    )

    auto_telegram = st.checkbox(
        "Tarama sırasında yön sinyallerini Telegram'a gönder",
        value=False,
        disabled=not (telegram_token and telegram_chat_id),
    )

    if st.button("🔍 Taramayı Başlat", type="primary"):
        symbols = (item.strip().upper() for item in girilen_hisseler.replace("\n", ",").split(","))
        h_list = list(dict.fromkeys(symbol for symbol in symbols if symbol))
        if not h_list:
            st.warning("Taramak için en az bir sembol girin.")
        else:
            results = []
            failures = []
            bar = st.progress(0)
            for idx, symbol in enumerate(h_list, 1):
                _, summary = analiz_hesapla(symbol, is_bist_flag)
                if isinstance(summary, str):
                    failures.append(f"{symbol}: {summary}")
                else:
                    if auto_telegram and summary["Sinyal"] in (UP, DOWN):
                        ok, message = send_telegram_message(
                            telegram_token, telegram_chat_id, format_summary(summary, para_birimi)
                        )
                        if not ok:
                            st.warning(f"{symbol} bildirimi gönderilemedi: {message}")
                    results.append({key: value for key, value in summary.items() if key != "Test"})
                bar.progress(idx / len(h_list))
            bar.empty()

            if results:
                table = pd.DataFrame(results).set_index("Hisse").sort_index()
                st.dataframe(table, width="stretch")
            if failures:
                st.warning(f"{len(failures)} sembol analiz edilemedi.")
                with st.expander("Analiz hataları", expanded=not results):
                    for failure in failures:
                        st.text(failure)
