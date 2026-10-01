import sys
import os
import time
import datetime
import math
import logging
import json
import requests
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
    "USER_AGENTS": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
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

        if not data or "chart" not in data or not data["chart"]["result"]:
            return self._fetch_stooq_fallback(symbol)

        try:
            result = data["chart"]["result"][0]
            meta = result["meta"]
            indicators = result.get("indicators", {}).get("quote", [{}])[0]

            closes = [c for c in indicators.get("close", []) if c is not None]
            opens = [o for o in indicators.get("open", []) if o is not None]
            highs = [h for h in indicators.get("high", []) if h is not None]
            lows = [l for l in indicators.get("low", []) if l is not None]
            volumes = [v for v in indicators.get("volume", []) if v is not None]

            if not closes:
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
                    close_val = float(item.get("close", 0))
                    open_val = float(item.get("open", 0))
                    return {
                        "symbol": str(symbol),
                        "current_price": close_val,
                        "open": open_val,
                        "high": float(item.get("high", close_val)),
                        "low": float(item.get("low", close_val)),
                        "volume": int(item.get("volume", 0)),
                        "previous_close": open_val,
                        "change": close_val - open_val,
                        "change_percent": ((close_val - open_val) / open_val * 100) if open_val > 0 else 0.0,
                        "timestamp": datetime.datetime.now().isoformat(),
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
                from io import StringIO
                df = pd.read_csv(StringIO(res.text))
                df.columns = [str(c).lower() for c in df.columns]
                if "date" in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
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
        self.df['sma_20'] = self.df['close'].rolling(20).mean()
        self.df['sma_50'] = self.df['close'].rolling(50).mean()
        self.df['sma_200'] = self.df['close'].rolling(200).mean()
        
        # RSI
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        self.df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)

        # Volume SMA
        self.df['vol_sma_20'] = self.df['volume'].rolling(20).mean()

    def analyze_breakout_signals(self) -> dict:
        if len(self.df) < 20:
            return {"breakout_detected": False, "reason": "Yetersiz zaman serisi verisi."}
            
        last_row = self.df.iloc[-1]
        
        current_price = float(last_row['close'])
        current_volume = float(last_row['volume'])
        avg_volume = float(last_row['vol_sma_20']) if not pd.isna(last_row['vol_sma_20']) else 1.0
        
        volume_spike = bool(current_volume > (avg_volume * CONFIG["BREAKOUT_VOLUME_FACTOR"]))
        volume_ratio = float(current_volume / avg_volume) if avg_volume > 0 else 0.0
        
        score = 0
        if volume_spike:
            score += 40
        if float(last_row['rsi']) > 55:
            score += 30
        if float(last_row['close']) > float(last_row['sma_20']):
            score += 30

        return {
            "breakout_detected": bool(score >= 50),
            "breakout_score": int(score),
            "volume_spike": bool(volume_spike),
            "volume_ratio": round(float(volume_ratio), 2),
            "current_rsi": float(last_row['rsi'])
        }

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
# STREAMLIT UI ARAYÜZÜ
# ==============================================================================
def main():
    st.title("📈 BIST & Küresel Hisse Senedi Kırılım Analizi")
    st.markdown("Hisse verileri önbellek koruması ve yedekli HTTP istemcisi üzerinden güvenle sorgulanır.")

    st.sidebar.header("Arama Ayarları")
    default_symbols = "KOZAL.IS, THYAO.IS, GARAN.IS, AAPL"
    user_input = st.sidebar.text_input("Hisse Sembolleri (Virgül ile ayırın):", value=default_symbols)

    symbols = [s.strip().upper() for s in user_input.split(",") if s.strip()]

    if st.sidebar.button("Analiz Et / Yenile", type="primary"):
        st.cache_data.clear()

    for symbol in symbols:
        with st.expander(f"📌 {symbol} Analiz Sonuçları", expanded=True):
            live_data, df_hist = get_stock_data(symbol)

            if df_hist.empty:
                st.error(f"[{symbol}] için geçerli veri indirilemedi. Sembol delisted olmuş veya Yahoo yanıt vermiyor olabilir.")
                continue

            analyzer = BreakoutAnalysisEngine(df_hist)
            breakout = analyzer.analyze_breakout_signals()

            # Canlı Veri Kartları
            col1, col2, col3, col4 = st.columns(4)
            curr = live_data.get("current_price", 0.0)
            pct = live_data.get("change_percent", 0.0)
            col1.metric("Son Fiyat", f"{curr:.2f} {live_data.get('currency', 'TRY')}", delta=f"%{pct:.2f}")
            col2.metric("Günlük Yüksek", f"{live_data.get('high', 0.0):.2f}")
            col3.metric("Günlük Düşük", f"{live_data.get('low', 0.0):.2f}")
            col4.metric("Veri Kaynağı", live_data.get("data_source", "Bilinmiyor"))

            st.divider()

            # Kırılım Durumu
            if breakout.get("breakout_detected"):
                st.success(f"🚀 **KIRILIM TESPİT EDİLDİ!** (Skor: {breakout.get('breakout_score')}/100)")
            else:
                st.info(f"ℹ️ **Kırılım Belirtisi Yok** (Skor: {breakout.get('breakout_score')}/100)")

            col_a, col_b = st.columns(2)
            col_a.write(f"**Hacim Katı:** {breakout.get('volume_ratio')}x")
            col_b.write(f"**RSI (14):** {breakout.get('current_rsi', 0):.2f}")

            # Grafik Gösterimi
            st.line_chart(df_hist[['close', 'sma_20']])

if __name__ == "__main__":
    main()
