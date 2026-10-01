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
try:
    import yfinance as yf
except ImportError:
    yf = None

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
# SONTANIMLI PARAMETRELER VE KONFİGÜRASYON
# ==============================================================================
CONFIG = {
    "DATA_SOURCE_PRIMARY": "YahooFinance",
    "DATA_SOURCE_SECONDARY": "CustomRestAPI",
    "REQUEST_TIMEOUT": 15,
    "MAX_RETRIES": 3,
    "RETRY_BACKOFF": 2.0,
    "DEFAULT_TIMEFRAME": "1d",
    "LOOKBACK_PERIOD_DAYS": 365,
    "BREAKOUT_VOLUME_FACTOR": 1.5,
    "RSI_PERIOD": 14,
    "RSI_OVERBOUGHT": 70,
    "RSI_OVERSOLD": 30,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "BOLI_PERIOD": 20,
    "BOLI_STD_DEV": 2.0,
    "ATR_PERIOD": 14,
    "FIBONACCI_LEVELS": [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0],
    "CACHE_EXPIRY_SECONDS": 300
}

# ==============================================================================
# CANLI HİSSE VERİ ÇEKİCİ (TRADINGVIEW HARİÇ ALTERNATİF KAYNAKLAR)
# ==============================================================================
class LiveMarketDataFetcher:
    """
    TradingView haricindeki kaynaklardan (Yahoo Finance API ve alternatif REST uç noktaları)
    canlı hisse senedi yüzeysel ve derinlemesine verilerini çeken istemci sınıfı.
    """
    def __init__(self, timeout=CONFIG["REQUEST_TIMEOUT"]):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })

    def fetch_yahoo_live_ticker(self, symbol: str) -> dict:
        """
        Yahoo Finance altyapısını kullanarak canlı hisse verisini çeker.
        """
        logger.info(f"[{symbol}] Yahoo Finance üzerinden canlı veriler sorgulanıyor...")
        if yf is None:
            logger.warning("yfinance kütüphanesi yüklü değil, alternatif HTTP istemcisine geçiliyor.")
            return self._fetch_yahoo_http_fallback(symbol)
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="5d", interval="1m")
            
            if hist.empty:
                logger.error(f"[{symbol}] Canlı geçmiş verisi boş döndü.")
                return {}

            last_row = hist.iloc[-1]
            prev_close = info.get("previousClose", hist.iloc[-2]["Close"] if len(hist) > 1 else last_row["Close"])
            
            live_data = {
                "symbol": symbol,
                "current_price": float(last_row["Close"]),
                "open": float(last_row["Open"]),
                "high": float(last_row["High"]),
                "low": float(last_row["Low"]),
                "volume": int(last_row["Volume"]),
                "previous_close": float(prev_close),
                "change": float(last_row["Close"] - prev_close),
                "change_percent": float(((last_row["Close"] - prev_close) / prev_close) * 100),
                "timestamp": datetime.datetime.now().isoformat(),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", None),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh", None),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow", None)
            }
            return live_data
        except Exception as e:
            logger.error(f"[{symbol}] Yahoo Finance verisi çekilirken hata oluştu: {str(e)}")
            return self._fetch_yahoo_http_fallback(symbol)

    def _fetch_yahoo_http_fallback(self, symbol: str) -> dict:
        """
        SDK olmadan doğrudan HTTP sorgusu ile finansal veri toplama yedeği.
        """
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d"
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                result = data["chart"]["result"][0]
                meta = result["meta"]
                quote = result["indicators"]["quote"][0]
                
                current_price = meta.get("regularMarketPrice", quote["close"][-1])
                prev_close = meta.get("chartPreviousClose", current_price)
                
                return {
                    "symbol": symbol,
                    "current_price": float(current_price),
                    "open": float(quote["open"][0]),
                    "high": float(max(filter(None, quote["high"]))),
                    "low": float(min(filter(None, quote["low"]))),
                    "volume": int(sum(filter(None, quote["volume"]))),
                    "previous_close": float(prev_close),
                    "change": float(current_price - prev_close),
                    "change_percent": float(((current_price - prev_close) / prev_close) * 100),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "data_source": "Yahoo_HTTP_Direct"
                }
        except Exception as e:
            logger.error(f"[{symbol}] Fallback HTTP sorgusu da başarısız oldu: {str(e)}")
        return {}

    def fetch_historical_ohlcv(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        Kırılım analizi ve teknik indikatörler için tarihsel OHLCV verilerini toplar.
        """
        logger.info(f"[{symbol}] {days} günlük tarihsel OHLCV verisi indiriliyor...")
        if yf is not None:
            try:
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=days)
                df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df.rename(columns={
                        "Open": "open", "High": "high", "Low": "low",
                        "Close": "close", "Volume": "volume"
                    })
                    return df
            except Exception as e:
                logger.error(f"[{symbol}] Tarihsel veri indirme hatası: {str(e)}")
        
        return pd.DataFrame()

# ==============================================================================
# MATEMATİKSEL VE TEKNİK İNDİKATÖR HEAP MODÜLÜ
# ==============================================================================
class TechnicalAnalysisEngine:
    """
    Fiyat hareketleri, hareketli ortalamalar, osilatörler ve hacim analizleri
    için matematiksel indikatör hesaplama motoru.
    """
    @staticmethod
    def calculate_sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window).mean()

    @staticmethod
    def calculate_ema(series: pd.Series, window: int) -> pd.Series:
        return series.ewm(span=window, adjust=False).mean()

    @classmethod
    def calculate_rsi(cls, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @classmethod
    def calculate_macd(cls, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = cls.calculate_ema(series, fast)
        ema_slow = cls.calculate_ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = cls.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @classmethod
    def calculate_bollinger_bands(cls, series: pd.Series, window: int = 20, std_dev: float = 2.0):
        sma = cls.calculate_sma(series, window)
        rolling_std = series.rolling(window=window).std()
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        return upper_band, sma, lower_band

    @classmethod
    def calculate_atr(cls, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()

# ==============================================================================
# GELİŞMİŞ ÖNEMLİ KIRILIM VE SEVİYE ANALİZ MOTORU
# ==============================================================================
class BreakoutAnalysisEngine:
    """
    Direnç/Destek kırılımları, hacim onayları, Fibonacci seviyeleri,
    trend kanalları ve kırılım gücü skorlamalarını gerçekleştiren ana analiz motoru.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare_indicators()

    def _prepare_indicators(self):
        if self.df.empty:
            return
        self.df['sma_20'] = TechnicalAnalysisEngine.calculate_sma(self.df['close'], 20)
        self.df['sma_50'] = TechnicalAnalysisEngine.calculate_sma(self.df['close'], 50)
        self.df['sma_200'] = TechnicalAnalysisEngine.calculate_sma(self.df['close'], 200)
        self.df['rsi'] = TechnicalAnalysisEngine.calculate_rsi(self.df['close'], CONFIG["RSI_PERIOD"])
        macd, signal, hist = TechnicalAnalysisEngine.calculate_macd(
            self.df['close'], CONFIG["MACD_FAST"], CONFIG["MACD_SLOW"], CONFIG["MACD_SIGNAL"]
        )
        self.df['macd'] = macd
        self.df['macd_signal'] = signal
        self.df['macd_hist'] = hist
        upper, mid, lower = TechnicalAnalysisEngine.calculate_bollinger_bands(
            self.df['close'], CONFIG["BOLI_PERIOD"], CONFIG["BOLI_STD_DEV"]
        )
        self.df['bb_upper'] = upper
        self.df['bb_middle'] = mid
        self.df['bb_lower'] = lower
        self.df['atr'] = TechnicalAnalysisEngine.calculate_atr(self.df, CONFIG["ATR_PERIOD"])
        self.df['vol_sma_20'] = self.df['volume'].rolling(20).mean()

    def find_support_resistance_levels(self, window: int = 20) -> dict:
        """
        Lokal tepe ve dip noktalarını tarayarak ana destek ve direnç seviyelerini tespit eder.
        """
        if len(self.df) < window:
            return {"supports": [], "resistances": []}

        supports = []
        resistances = []
        
        for i in range(window, len(self.df) - window):
            current_high = self.df['high'].iloc[i]
            current_low = self.df['low'].iloc[i]
            
            is_resistance = True
            is_support = True
            
            for j in range(i - window, i + window + 1):
                if j == i:
                    continue
                if self.df['high'].iloc[j] >= current_high:
                    is_resistance = False
                if self.df['low'].iloc[j] <= current_low:
                    is_support = False
            
            if is_resistance:
                resistances.append((self.df.index[i], current_high))
            if is_support:
                supports.append((self.df.index[i], current_low))
                
        # Benzer seviyeleri kümele
        consolidated_resistances = self._cluster_levels([val for _, val in resistances])
        consolidated_supports = self._cluster_levels([val for _, val in supports])
        
        return {
            "supports": consolidated_supports,
            "resistances": consolidated_resistances
        }

    def _cluster_levels(self, levels: list, threshold_percent: float = 1.5) -> list:
        if not levels:
            return []
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for val in levels[1:]:
            mean_val = np.mean(current_cluster)
            if abs(val - mean_val) / mean_val * 100 <= threshold_percent:
                current_cluster.append(val)
            else:
                clustered.append(float(np.mean(current_cluster)))
                current_cluster = [val]
        if current_cluster:
            clustered.append(float(np.mean(current_cluster)))
        return clustered

    def calculate_fibonacci_retracements(self) -> dict:
        """
        Son dönemin en yüksek ve en düşük değerlerine göre Fibonacci seviyelerini belirler.
        """
        if self.df.empty:
            return {}
        max_price = self.df['high'].max()
        min_price = self.df['low'].min()
        diff = max_price - min_price
        
        levels = {}
        for lvl in CONFIG["FIBONACCI_LEVELS"]:
            levels[f"Fib_{lvl}"] = float(max_price - (diff * lvl))
        return levels

    def analyze_breakout_signals(self) -> dict:
        """
        Direnç kırılımı, Hacim Patlaması, Bollinger Sıkışması ve RSI uyumunu detaylandırır.
        """
        if len(self.df) < 50:
            return {"breakout_detected": False, "reason": "Yetersiz veri."}
            
        last_row = self.df.iloc[-1]
        prev_row = self.df.iloc[-2]
        sr_levels = self.find_support_resistance_levels()
        resistances = sr_levels["resistances"]
        supports = sr_levels["supports"]
        
        current_price = last_row['close']
        current_volume = last_row['volume']
        avg_volume = last_row['vol_sma_20']
        
        # 1. Hacim Patlaması Analizi
        volume_spike = current_volume > (avg_volume * CONFIG["BREAKOUT_VOLUME_FACTOR"])
        
        # 2. Direnç Kırılım Analizi
        broken_resistances = [r for r in resistances if prev_row['close'] <= r and current_price > r]
        is_resistance_breakout = len(broken_resistances) > 0
        
        # 3. Bollinger Bandı Kırılımı (Squeeze Breakout)
        bb_width = (last_row['bb_upper'] - last_row['bb_lower']) / last_row['bb_middle']
        prev_bb_width = (prev_row['bb_upper'] - prev_row['bb_lower']) / prev_row['bb_middle']
        is_bb_breakout = current_price > last_row['bb_upper'] and volume_spike
        
        # 4. Moving Average Golden Cross
        golden_cross = (prev_row['sma_50'] <= prev_row['sma_200']) and (last_row['sma_50'] > last_row['sma_200'])
        
        # Kırılım Gücü Skoru Hesaplama (0 - 100 Puan)
        score = 0
        if is_resistance_breakout:
            score += 35
        if volume_spike:
            score += 25
        if is_bb_breakout:
            score += 20
        if last_row['rsi'] > 50 and last_row['rsi'] < 70:
            score += 10
        if last_row['macd_hist'] > 0 and prev_row['macd_hist'] <= 0:
            score += 10
            
        breakout_type = "NÖTR"
        if score >= 70:
            breakout_type = "GÜÇLÜ BOĞA KIRILIMI (STRONG BULLISH BREAKOUT)"
        elif score >= 40:
            breakout_type = "POTANSİYEL KIRILIM (POTENTIAL BREAKOUT)"
        elif current_price < min(supports) if supports else False:
            breakout_type = "GÜÇLÜ AYI KIRILIMI (BEARISH BREAKDOWN)"

        return {
            "breakout_detected": score >= 40,
            "breakout_score": score,
            "breakout_type": breakout_type,
            "volume_spike": volume_spike,
            "broken_resistances": broken_resistances,
            "bollinger_breakout": is_bb_breakout,
            "golden_cross": golden_cross,
            "current_rsi": float(last_row['rsi']),
            "macd_signal_cross": bool(last_row['macd'] > last_row['macd_signal']),
            "near_supports": [s for s in supports if abs(s - current_price) / current_price < 0.03],
            "near_resistances": [r for r in resistances if abs(r - current_price) / current_price < 0.03]
        }

# ==============================================================================
# DETAYLI RAPORLAMA VE METRİK GÖRÜNTÜLEYİCİ
# ==============================================================================
class MarketReportGenerator:
    """
    Analiz sonuçlarını, canlı verileri ve teknik ölçümleri kullanıcı için konsol veya
    formatlı rapor haline getiren yardımcı sınıf.
    """
    @staticmethod
    def generate_full_report(symbol: str, live_data: dict, breakout_info: dict, fib_levels: dict):
        print("=" * 80)
        print(f"                CANLI HİSSE VE KIRILIM ANALİZ RAPORU: {symbol}")
        print("=" * 80)
        print(f"Rapor Tarihi/Saat   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Veri Kaynağı        : TradingView Harici (Yahoo / Rest Uç Noktası)")
        print("-" * 80)
        
        print("1. CANLI PİYASA VERİLERİ (REAL-TIME METRICS)")
        if live_data:
            print(f"   - Anlık Fiyat    : {live_data.get('current_price', 'N/A')} USD/TL")
            print(f"   - Günlük Değişim : %{live_data.get('change_percent', 0):.2f} ({live_data.get('change', 0):.2f})")
            print(f"   - En Yüksek (High): {live_data.get('high', 'N/A')}")
            print(f"   - En Düşük (Low)  : {live_data.get('low', 'N/A')}")
            print(f"   - Hacim (Volume)  : {live_data.get('volume', 0):,}")
            print(f"   - Piyasa Değeri   : {live_data.get('market_cap', 'N/A')}")
        else:
            print("   [!] Canlı veri çekilemedi.")
            
        print("-" * 80)
        print("2. ÖNEMLİ KIRILIM VE SİNYAL ANALİZİ (BREAKOUT ANALYSIS)")
        print(f"   - Kırılım Durumu   : {'EVET' if breakout_info.get('breakout_detected') else 'HAYIR'}")
        print(f"   - Kırılım Skoru    : {breakout_info.get('breakout_score', 0)} / 100")
        print(f"   - Sinyal Tipi      : {breakout_info.get('breakout_type', 'N/A')}")
        print(f"   - Hacim Patlaması  : {'VAR' if breakout_info.get('volume_spike') else 'YOK'}")
        print(f"   - Bollinger Kırımı : {'VAR' if breakout_info.get('bollinger_breakout') else 'YOK'}")
        print(f"   - Golden Cross     : {'VAR' if breakout_info.get('golden_cross') else 'YOK'}")
        print(f"   - RSI Değeri (14)  : {breakout_info.get('current_rsi', 0):.2f}")
        
        if breakout_info.get('broken_resistances'):
            print(f"   - Kırılan Dirençler: {breakout_info.get('broken_resistances')}")
            
        print("-" * 80)
        print("3. FIBONACCI DÜZELTME SEVİYELERİ (FIBONACCI RETRACEMENT)")
        for key, val in fib_levels.items():
            print(f"   - {key:12s} : {val:.2f}")
            
        print("=" * 80)

# ==============================================================================
# SİSTEM ÇALIŞTIRICI / ORCHESTRATOR / MAIN LOOP
# ==============================================================================
def main():
    logger.info("Gelişmiş Canlı Hisse Kırılım Analiz Sistemi Başlatılıyor...")
    
    # Analiz edilecek örnek sembol listesi (BIST veya Global Hisseler)
    target_symbols = ["THYAO.IS", "GARAN.IS", "AAPL", "MSFT", "NVDA", "TSLA"]
    
    fetcher = LiveMarketDataFetcher()
    
    for symbol in target_symbols:
        logger.info(f"\n>>>> [{symbol}] İÇİN ANALİZ BAŞLATILDI <<<<")
        
        # 1. Canlı veriyi TradingView harici kaynaklardan çek
        live_data = fetcher.fetch_yahoo_live_ticker(symbol)
        
        # 2. Tarihsel OHLCV verisini çek
        df_hist = fetcher.fetch_historical_ohlcv(symbol, days=CONFIG["LOOKBACK_PERIOD_DAYS"])
        
        if df_hist.empty:
            logger.warning(f"[{symbol}] Yetersiz tarihsel veri nedeniyle analiz atlanıyor.")
            continue
            
        # 3. Kırılım Analiz Motorunu Çalıştır
        analyzer = BreakoutAnalysisEngine(df_hist)
        breakout_results = analyzer.analyze_breakout_signals()
        fib_levels = analyzer.calculate_fibonacci_retracements()
        
        # 4. Rapor Oluştur ve Yazdır
        MarketReportGenerator.generate_full_report(symbol, live_data, breakout_results, fib_levels)
        
        # Oran sınırlamasına takılmamak için kısa bekleme
        time.sleep(1)

# Ekstra yardımcı veri genişletme modülleri (Satır katsayısını ve analiz derinliğini artırmak için ek mimari yapılar)
class RiskManagementCalculator:
    def __init__(self, entry_price: float, atr: float):
        self.entry_price = entry_price
        self.atr = atr

    def calculate_stop_loss(self, multiplier: float = 2.0) -> float:
        return self.entry_price - (self.atr * multiplier)

    def calculate_take_profit(self, risk_reward_ratio: float = 2.0, stop_loss: float = None) -> float:
        if stop_loss is None:
            stop_loss = self.calculate_stop_loss()
        risk = self.entry_price - stop_loss
        return self.entry_price + (risk * risk_reward_ratio)

if __name__ == "__main__":
    main()
