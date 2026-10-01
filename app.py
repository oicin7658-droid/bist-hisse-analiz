import sys
import os
import time
import datetime
import math
import logging
import json
import requests
from io import StringIO
import pandas as pd
import numpy as np
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None

# ==============================================================================
# STREAMLIT SAYFA YAPILANDIRMASI
# ==============================================================================
st.set_page_config(
    page_title="Gelişmiş BIST & Hisse Kırılım Analiz Paneli",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# LOGGING VE SİSTEM YAPILANDIRMASI
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AdvancedStockBreakoutAnalyzer")

# ==============================================================================
# KONFİGÜRASYON
# ==============================================================================
CONFIG = {
    "REQUEST_TIMEOUT": 20,
    "MAX_RETRIES": 3,
    "RETRY_BACKOFF": 2.0,
    "LOOKBACK_PERIOD_DAYS": 365,
    "BREAKOUT_VOLUME_FACTOR": 1.5,
    "RSI_PERIOD": 14,
    "BOLI_PERIOD": 20,
    "BOLI_STD_DEV": 2.0,
    "ATR_PERIOD": 14,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "USER_AGENTS": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ]
}

# ==============================================================================
# GELİŞMİŞ HTTP İSTEMCİSİ (401 / RATE LIMIT ÇÖZÜCÜ)
# ==============================================================================
class RobustHTTPClient:
    def __init__(self, timeout: int = CONFIG["REQUEST_TIMEOUT"]):
        self.timeout = timeout
        self.session = requests.Session()
        self._initialize_headers()

    def _initialize_headers(self):
        self.session.headers.update({
            "User-Agent": CONFIG["USER_AGENTS"][0],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
            "Connection": "keep-alive"
        })

    def get_json(self, url: str) -> dict:
        for attempt in range(CONFIG["MAX_RETRIES"]):
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code in (401, 403):
                    logger.warning(f"HTTP {response.status_code} alındı. Çerezler yenileniyor... URL: {url}")
                    self.session.get("https://finance.yahoo.com", timeout=self.timeout)
            except Exception as e:
                logger.error(f"Bağlantı hatası: {str(e)} (Deneme {attempt + 1})")
            time.sleep(CONFIG["RETRY_BACKOFF"] * (attempt + 1))
        return {}

# ==============================================================================
# CANLI VE TARİHSEL VERİ ÇEKİCİ (STREAMLIT CACHE DESTEKLİ)
# ==============================================================================
class LiveMarketDataFetcher:
    def __init__(self):
        self.http_client = RobustHTTPClient()

    def fetch_yahoo_live_ticker(self, symbol: str) -> dict:
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d"
        data = self.http_client.get_json(url)

        if not data or "chart" not in data or not data["chart"].get("result"):
            logger.info(f"[{symbol}] Yahoo v8 verisi alınamadı. Stooq yedek kanalına geçiliyor.")
            return self._fetch_stooq_fallback(symbol)

        try:
            result = data["chart"]["result"][0]
            meta = result.get("meta", {})
            indicators = result.get("indicators", {}).get("quote", [{}])[0]

            closes = [c for c in indicators.get("close", []) if c is not None]
            opens = [o for o in indicators.get("open", []) if o is not None]
            highs = [h for h in indicators.get("high", []) if h is not None]
            lows = [l for l in indicators.get("low", []) if l is not None]
            volumes = [v for v in indicators.get("volume", []) if v is not None]

            if not closes:
                logger.warning(f"[{symbol}] Kapanış verisi boş. Stooq yedek kanalına geçiliyor.")
                return self._fetch_stooq_fallback(symbol)

            current_price = meta.get("regularMarketPrice", closes[-1])
            prev_close = meta.get("chartPreviousClose", closes[0])
            price_change = current_price - prev_close
            change_percent = (price_change / prev_close) * 100 if prev_close else 0.0

            return {
                "symbol": str(symbol),
                "current_price": float(current_price),
                "open": float(opens[0]) if opens else float(current_price),
                "high": float(max(highs)) if highs else float(current_price),
                "low": float(min(lows)) if lows else float(current_price),
                "volume": int(sum(volumes)) if volumes else 0,
                "previous_close": float(prev_close),
                "change": float(price_change),
                "change_percent": float(change_percent),
                "timestamp": datetime.datetime.now().isoformat(),
                "currency": meta.get("currency", "TRY"),
                "data_source": "Yahoo_Finance_Direct"
            }
        except Exception as e:
            logger.error(f"[{symbol}] Canlı veri işleme hatası: {str(e)}")
            return self._fetch_stooq_fallback(symbol)

    def _fetch_stooq_fallback(self, symbol: str) -> dict:
        clean_symbol = symbol.lower().replace(".is", ".tr")
        if "." not in clean_symbol:
            clean_symbol += ".us"

        url = f"https://stooq.com/q/l/?s={clean_symbol}&f=sdohv&e=json"
        try:
            res = requests.get(url, timeout=10)
            if res.status_code == 200:
                json_data = res.json()
                symbols_list = json_data.get("symbols", [])
                if symbols_list:
                    item = symbols_list[0]
                    close_val = float(item.get("close", 0) or 0)
                    open_val = float(item.get("open", 0) or 0)
                    if close_val == 0:
                        return {}
                    return {
                        "symbol": str(symbol),
                        "current_price": close_val,
                        "open": open_val,
                        "high": float(item.get("high", close_val) or close_val),
                        "low": float(item.get("low", close_val) or close_val),
                        "volume": int(item.get("volume", 0) or 0),
                        "previous_close": open_val,
                        "change": close_val - open_val,
                        "change_percent": ((close_val - open_val) / open_val * 100) if open_val > 0 else 0.0,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "currency": "TRY" if ".is" in symbol.lower() else "USD",
                        "data_source": "Stooq_Backup_Engine"
                    }
        except Exception as e:
            logger.error(f"[{symbol}] Stooq yedek hatası: {str(e)}")
        return {}

    def fetch_historical_ohlcv(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        Dinamik güncel tarih hesaplayarak 'No data found' hatasını çözen indirme.
        """
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)

        if yf is not None:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
                
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.columns = [str(c).lower() for c in df.columns]
                    df = df.dropna(subset=['close'])
                    return df
            except Exception as e:
                logger.warning(f"[{symbol}] yfinance SDK hatası: {str(e)}. Direct CSV deneniyor...")

        return self._download_yahoo_csv_fallback(symbol, start_date, end_date)

    def _download_yahoo_csv_fallback(self, symbol: str, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
        start_ts = int(start_date.timestamp())
        end_ts = int(end_date.timestamp())
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={start_ts}&period2={end_ts}&interval=1d&events=history"
        
        try:
            res = self.http_client.session.get(url, timeout=15)
            if res.status_code == 200 and "Date" in res.text:
                df = pd.read_csv(StringIO(res.text))
                df.columns = [str(c).lower() for c in df.columns]
                if "date" in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                df = df.dropna(subset=['close'])
                return df
        except Exception as e:
            logger.error(f"[{symbol}] Direct CSV yedeği indirilemedi: {str(e)}")
        
        return pd.DataFrame()

# ==============================================================================
# TEKNİK ANALİZ MOTORU
# ==============================================================================
class BreakoutAnalysisEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_indicators()

    def _prepare_indicators(self):
        if self.df.empty or len(self.df) < 20:
            return
            
        # Hareketli Ortalamalar (SMA)
        self.df['sma_20'] = self.df['close'].rolling(20).mean()
        self.df['sma_50'] = self.df['close'].rolling(50).mean()
        self.df['sma_200'] = self.df['close'].rolling(200).mean()
        
        # Üstel Hareketli Ortalamalar (EMA)
        self.df['ema_12'] = self.df['close'].ewm(span=12, adjust=False).mean()
        self.df['ema_26'] = self.df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD Hesaplaması
        self.df['macd'] = self.df['ema_12'] - self.df['ema_26']
        self.df['macd_signal'] = self.df['macd'].ewm(span=CONFIG["MACD_SIGNAL"], adjust=False).mean()
        self.df['macd_hist'] = self.df['macd'] - self.df['macd_signal']

        # RSI (Relative Strength Index)
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(CONFIG["RSI_PERIOD"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(CONFIG["RSI_PERIOD"]).mean()
        rs = gain / loss.replace(0, np.nan)
        self.df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)

        # Bollinger Bantları (Bollinger Bands)
        self.df['boli_middle'] = self.df['close'].rolling(CONFIG["BOLI_PERIOD"]).mean()
        self.df['boli_std'] = self.df['close'].rolling(CONFIG["BOLI_PERIOD"]).std()
        self.df['boli_upper'] = self.df['boli_middle'] + (self.df['boli_std'] * CONFIG["BOLI_STD_DEV"])
        self.df['boli_lower'] = self.df['boli_middle'] - (self.df['boli_std'] * CONFIG["BOLI_STD_DEV"])
        self.df['boli_bandwidth'] = (self.df['boli_upper'] - self.df['boli_lower']) / self.df['boli_middle']

        # ATR (Average True Range)
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        self.df['atr'] = true_range.rolling(CONFIG["ATR_PERIOD"]).mean()

        # Hacim İndikatörleri
        self.df['vol_sma_20'] = self.df['volume'].rolling(20).mean()
        self.df['vol_ratio'] = self.df['volume'] / self.df['vol_sma_20'].replace(0, np.nan)

    def get_processed_dataframe(self) -> pd.DataFrame:
        """İndikatörleri hesaplanmış DataFrame'i döndürür."""
        return self.df

    def analyze_breakout_signals(self) -> dict:
        if len(self.df) < 20:
            return {
                "breakout_detected": False, 
                "breakout_score": 0, 
                "reason": "Yetersiz zaman serisi verisi.",
                "volume_spike": False,
                "volume_ratio": 0.0,
                "current_rsi": 50.0,
                "bollinger_break": False,
                "macd_bullish": False
            }
            
        last_row = self.df.iloc[-1]
        prev_row = self.df.iloc[-2]
        
        current_price = float(last_row['close'])
        current_volume = float(last_row['volume'])
        avg_volume = float(last_row['vol_sma_20']) if not pd.isna(last_row['vol_sma_20']) else 1.0
        
        volume_spike = bool(current_volume > (avg_volume * CONFIG["BREAKOUT_VOLUME_FACTOR"]))
        volume_ratio = float(current_volume / avg_volume) if avg_volume > 0 else 0.0
        
        bollinger_break = bool(current_price > float(last_row['boli_upper'])) if not pd.isna(last_row.get('boli_upper')) else False
        macd_bullish = bool(float(last_row.get('macd', 0)) > float(last_row.get('macd_signal', 0)))
        
        score = 0
        if volume_spike:
            score += 30
        if float(last_row['rsi']) > 55:
            score += 20
        if not pd.isna(last_row['sma_20']) and float(last_row['close']) > float(last_row['sma_20']):
            score += 20
        if bollinger_break:
            score += 15
        if macd_bullish:
            score += 15

        return {
            "breakout_detected": bool(score >= 50),
            "breakout_score": int(score),
            "volume_spike": bool(volume_spike),
            "volume_ratio": round(float(volume_ratio), 2),
            "current_rsi": float(last_row['rsi']),
            "bollinger_break": bollinger_break,
            "macd_bullish": macd_bullish,
            "current_atr": float(last_row.get('atr', 0.0)) if not pd.isna(last_row.get('atr')) else 0.0
        }

# ==============================================================================
# ARAMA VE BİST LİSTE YARDIMCILARI
# ==============================================================================
class BISTRegistryHelper:
    @staticmethod
    def get_popular_bist_tickers() -> list:
        return [
            "THYAO.IS", "GARAN.IS", "KOZAL.IS", "EREGL.IS", "ASELS.IS",
            "AKBNK.IS", "YKBNK.IS", "SISE.IS", "BIMAS.IS", "KCHOL.IS",
            "SASA.IS", "HEKTS.IS", "TUPRS.IS", "PETKM.IS", "SAHOL.IS"
        ]

    @staticmethod
    def format_symbol_input(user_input_str: str) -> list:
        raw_symbols = [s.strip().upper() for s in user_input_str.split(",") if s.strip()]
        formatted = []
        for sym in raw_symbols:
            if not sym.endswith(".IS") and not "." in sym and not sym.startswith("^"):
                #Varsayılan olarak Türk BIST hissesi kontrolü
                formatted.append(f"{sym}.IS")
            else:
                formatted.append(sym)
        return formatted

# ==============================================================================
# STREAMLIT CACHED DATA FETCHING
# ==============================================================================
@st.cache_data(ttl=300)
def get_stock_data(symbol: str):
    fetcher = LiveMarketDataFetcher()
    live_data = fetcher.fetch_yahoo_live_ticker(symbol)
    df_hist = fetcher.fetch_historical_ohlcv(symbol, days=CONFIG["LOOKBACK_PERIOD_DAYS"])
    return live_data, df_hist

# ==============================================================================
# GELİŞMİŞ GÖRSELLEŞTİRME VE TABLO MODÜLLERİ
# ==============================================================================
def render_summary_table(analysis_results: list):
    """
    Tüm hisselerin genel durumunu özet tablo halinde sunar.
    """
    if not analysis_results:
        return
        
    st.subheader("📊 Analiz Edilen Hisselerin Genel Özet Tablosu")
    
    table_data = []
    for res in analysis_results:
        table_data.append({
            "Sembol": res["symbol"],
            "Son Fiyat": f"{res['price']:.2f} {res['currency']}",
            "Günlük Değişim (%)": f"%{res['change_pct']:.2f}",
            "Kırılım Skor": f"{res['score']}/100",
            "Kırılım Sinyali": "🚀 TESPİT EDİLDİ" if res["detected"] else "⚪ NÖTR",
            "Hacim Katı": f"{res['vol_ratio']}x",
            "RSI (14)": f"{res['rsi']:.1f}",
            "Veri Kaynağı": res["source"]
        })
        
    df_summary = pd.DataFrame(table_data)
    st.dataframe(df_summary, use_container_width=True)

def render_technical_details_tab(df_processed: pd.DataFrame, breakout: dict):
    """
    Teknik indikatör detaylarını sekmeler halinde gösterir.
    """
    tab1, tab2, tab3 = st.tabs(["📉 Fiyat ve Ortalamalar", "📊 MACD & RSI", "🛡️ Bollinger & Risk (ATR)"])
    
    with tab1:
        st.write("##### Fiyat, SMA 20 ve SMA 50 Grafiği")
        chart_cols = [c for c in ['close', 'sma_20', 'sma_50', 'sma_200'] if c in df_processed.columns]
        st.line_chart(df_processed[chart_cols].dropna())
        
    with tab2:
        col_r, col_m = st.columns(2)
        with col_r:
            st.write("##### RSI (14) Seviyesi")
            if 'rsi' in df_processed.columns:
                st.line_chart(df_processed['rsi'].dropna())
        with col_m:
            st.write("##### MACD İndikatörü")
            if 'macd' in df_processed.columns and 'macd_signal' in df_processed.columns:
                st.line_chart(df_processed[['macd', 'macd_signal']].dropna())
                
    with tab3:
        st.write("##### Bollinger Bantları")
        boll_cols = [c for c in ['close', 'boli_upper', 'boli_middle', 'boli_lower'] if c in df_processed.columns]
        if boll_cols:
            st.line_chart(df_processed[boll_cols].dropna())
        st.caption(f"Güncel ATR (Oynaklık / Risk Derecesi): **{breakout.get('current_atr', 0.0):.2f}**")

# ==============================================================================
# STREAMLIT UI ARAYÜZÜ
# ==============================================================================
def main():
    st.title("📈 BIST & Küresel Hisse Senedi Kırılım Analizi")
    st.markdown("Hisse verileri önbellek koruması ve yedekli HTTP istemcisi üzerinden güvenle sorgulanır.")

    # Yan Menü (Sidebar) Yapılandırması
    st.sidebar.header("⚙️ Arama ve Analiz Ayarları")
    
    # Popüler BIST Hisseleri Hızlı Seçim Kutusu
    popular_list = BISTRegistryHelper.get_popular_bist_tickers()
    selected_quick = st.sidebar.multiselect("Hızlı Popüler Hisse Ekle:", popular_list, default=["THYAO.IS", "KOZAL.IS"])
    
    default_symbols_str = ", ".join(selected_quick) if selected_quick else "KOZAL.IS, THYAO.IS, GARAN.IS, AAPL"
    user_input = st.sidebar.text_input("Hisse Sembolleri (Virgül ile ayırın):", value=default_symbols_str)

    # Parametre Özelleştirme Alanı
    with st.sidebar.expander("🛠️ İndikatör Hassasiyet Ayarları"):
        vol_factor = st.slider("Hacim Patlaması Çarpanı", min_value=1.1, max_value=3.0, value=CONFIG["BREAKOUT_VOLUME_FACTOR"], step=0.1)
        rsi_thresh = st.slider("RSI Eşik Değeri", min_value=40, max_value=70, value=55, step=1)
        CONFIG["BREAKOUT_VOLUME_FACTOR"] = vol_factor

    symbols = [s.strip().upper() for s in user_input.split(",") if s.strip()]

    if st.sidebar.button("🔄 Analiz Et / Verileri Yenile", type="primary"):
        st.cache_data.clear()
        st.sidebar.success("Önbellek temizlendi ve veriler güncellendi!")

    summary_accumulator = []

    for symbol in symbols:
        with st.expander(f"📌 {symbol} Analiz Sonuçları", expanded=True):
            live_data, df_hist = get_stock_data(symbol)

            if df_hist.empty:
                st.error(f"[{symbol}] için geçerli veri indirilemedi. Sembol delisted olmuş veya Yahoo/Stooq yanıt vermiyor olabilir.")
                continue

            analyzer = BreakoutAnalysisEngine(df_hist)
            breakout = analyzer.analyze_breakout_signals()
            df_processed = analyzer.get_processed_dataframe()

            curr_price = live_data.get("current_price", df_processed['close'].iloc[-1])
            change_pct = live_data.get("change_percent", 0.0)

            # Özet Liste Biriktirme
            summary_accumulator.append({
                "symbol": symbol,
                "price": curr_price,
                "change_pct": change_pct,
                "score": breakout.get("breakout_score", 0),
                "detected": breakout.get("breakout_detected", False),
                "vol_ratio": breakout.get("volume_ratio", 0.0),
                "rsi": breakout.get("current_rsi", 50.0),
                "currency": live_data.get("currency", "TRY"),
                "source": live_data.get("data_source", "Bilinmiyor")
            })

            # Canlı Veri Kartları (Metrics)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Son Fiyat", f"{curr_price:.2f} {live_data.get('currency', 'TRY')}", delta=f"%{change_pct:.2f}")
            col2.metric("Günlük Yüksek", f"{live_data.get('high', df_processed['high'].iloc[-1]):.2f}")
            col3.metric("Günlük Düşük", f"{live_data.get('low', df_processed['low'].iloc[-1]):.2f}")
            col4.metric("Veri Kaynağı", live_data.get("data_source", "Bilinmiyor"))

            st.divider()

            # Kırılım Durumu ve Sinyal Detayı
            if breakout.get("breakout_detected"):
                st.success(f"🚀 **KIRILIM TESPİT EDİLDİ!** (Skor: {breakout.get('breakout_score')}/100)")
            else:
                st.info(f"ℹ️ **Kırılım Belirtisi Yok** (Skor: {breakout.get('breakout_score')}/100)")

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.write(f"**Hacim Katı:** {breakout.get('volume_ratio')}x")
            col_b.write(f"**RSI (14):** {breakout.get('current_rsi', 0):.2f}")
            col_c.write(f"**Bollinger Üst Üstü:** {'EVET ✅' if breakout.get('bollinger_break') else 'HAYIR ❌'}")
            col_d.write(f"**MACD Boğa Pozisyonu:** {'EVET ✅' if breakout.get('macd_bullish') else 'HAYIR ❌'}")

            # Detaylı İndikatör Grafikleri Sekmeleri
            render_technical_details_tab(df_processed, breakout)

            # Ham Veri İnceleme Alanı
            with st.expander("🔍 Son 10 Günlük OHLCV Veri Tablosunu Görüntüle"):
                st.dataframe(df_processed.tail(10).sort_index(ascending=False), use_container_width=True)

    # Genel Özet Tabloyu Sayfa Altına Ekleme
    if summary_accumulator:
        st.divider()
        render_summary_table(summary_accumulator)

# ==============================================================================
# UYGULAMA GİRİŞ NOKTASI
# ==============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as top_level_error:
        logger.critical(f"Uygulama çökme hatası: {str(top_level_error)}")
        st.error(f"Uygulama çalışırken beklenmeyen bir hata oluştu: {str(top_level_error)}")
