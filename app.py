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
    "DATA_SOURCE_PRIMARY": "YahooFinance_Direct_API",
    "DATA_SOURCE_SECONDARY": "Stooq_HTTP_Fallback",
    "REQUEST_TIMEOUT": 20,
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
    "CACHE_EXPIRY_SECONDS": 300,
    "USER_AGENTS": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0"
    ]
}

# ==============================================================================
# GELİŞMİŞ HTTP İSTEMCİSİ (401 UNAUTHORIZED BAP-ENGEL ÇÖZÜCÜ)
# ==============================================================================
class RobustHTTPClient:
    """
    Yahoo Finance ve diğer finans servislerinin bot/401 engelini aşmak için
    gerçek tarayıcı başlıkları (headers) ve otomatik oturum yönetimi sağlayan istemci.
    """
    def __init__(self, timeout: int = CONFIG["REQUEST_TIMEOUT"]):
        self.timeout = timeout
        self.session = requests.Session()
        self._initialize_headers()

    def _initialize_headers(self):
        self.session.headers.update({
            "User-Agent": CONFIG["USER_AGENTS"][0],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,tr;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0"
        })

    def get_json(self, url: str) -> dict:
        for attempt in range(CONFIG["MAX_RETRIES"]):
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    logger.warning(f"401 Yetkilendirme hatası alındı. Oturum çerezleri yenileniyor... (Deneme {attempt + 1})")
                    self.session.get("https://finance.yahoo.com", timeout=self.timeout)
                else:
                    logger.warning(f"HTTP İstek Hatası ({response.status_code}): {url}")
            except Exception as e:
                logger.error(f"Bağlantı hatası: {str(e)} (Deneme {attempt + 1})")
            time.sleep(CONFIG["RETRY_BACKOFF"] * (attempt + 1))
        return {}

# ==============================================================================
# CANLI HİSSE VERİ ÇEKİCİ (TRADINGVIEW HARİÇ ALTERNATİF KAYNAKLAR)
# ==============================================================================
class LiveMarketDataFetcher:
    """
    TradingView kullanmadan, bot korumalı Yahoo Finance V8/V10 API uç noktaları
    ve Stooq yedek sağlayıcısından canlı/tarihsel borsa verisi çekici.
    """
    def __init__(self):
        self.http_client = RobustHTTPClient()

    def fetch_yahoo_live_ticker(self, symbol: str) -> dict:
        """
        401 Yetkilendirme hatasını engelleyen güncellenmiş Yahoo Chart Uç Noktası.
        """
        logger.info(f"[{symbol}] Canlı veriler Yahoo Finance altyapısından sorgulanıyor...")
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d"
        data = self.http_client.get_json(url)

        if not data or "chart" not in data or not data["chart"]["result"]:
            logger.error(f"[{symbol}] Canlı veri alınamadı. Yedeğe geçiliyor...")
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
                logger.warning(f"[{symbol}] Gün içi bar verisi boş döndü.")
                return self._fetch_stooq_fallback(symbol)

            current_price = meta.get("regularMarketPrice", closes[-1])
            prev_close = meta.get("chartPreviousClose", closes[0])
            price_change = current_price - prev_close
            change_percent = (price_change / prev_close) * 100 if prev_close else 0.0

            return {
                "symbol": symbol,
                "current_price": float(current_price),
                "open": float(opens[0]) if opens else float(current_price),
                "high": float(max(highs)) if highs else float(current_price),
                "low": float(min(lows)) if lows else float(current_price),
                "volume": int(sum(volumes)) if volumes else 0,
                "previous_close": float(prev_close),
                "change": float(price_change),
                "change_percent": float(change_percent),
                "timestamp": datetime.datetime.now().isoformat(),
                "market_cap": meta.get("marketCap", "N/A"),
                "currency": meta.get("currency", "USD"),
                "data_source": "Yahoo_Finance_Direct"
            }
        except Exception as e:
            logger.error(f"[{symbol}] Yanıt işleme hatası: {str(e)}")
            return self._fetch_stooq_fallback(symbol)

    def _fetch_stooq_fallback(self, symbol: str) -> dict:
        """
        Yahoo servisleri tamamen kısıtlandığında çalışan alternatif Borsa Veri Yedeği.
        """
        logger.info(f"[{symbol}] Stooq alternatif uç noktasından veri isteniyor...")
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
                        "symbol": symbol,
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
            logger.error(f"[{symbol}] Stooq yedeği de başarısız oldu: {str(e)}")
        return {}

    def fetch_historical_ohlcv(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """
        Tarihsel günlük mum (OHLCV) verisini yfinance SDK'sı veya doğrudan HTTP CSV indirmesiyle çeker.
        """
        logger.info(f"[{symbol}] {days} günlük tarihsel OHLCV verisi indiriliyor...")
        if yf is not None:
            try:
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=days)
                df = yf.download(
                    symbol,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    progress=False
                )
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.columns = [str(c).lower() for c in df.columns]
                    return df
            except Exception as e:
                logger.error(f"[{symbol}] yfinance SDK indirme hatası: {str(e)}")

        return self._download_yahoo_csv_fallback(symbol, days)

    def _download_yahoo_csv_fallback(self, symbol: str, days: int) -> pd.DataFrame:
        end_ts = int(time.time())
        start_ts = end_ts - (days * 86400)
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={start_ts}&period2={end_ts}&interval=1d&events=history"
        
        try:
            res = self.http_client.session.get(url, timeout=15)
            if res.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(res.text))
                df.columns = [str(c).lower() for c in df.columns]
                if "date" in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                return df
        except Exception as e:
            logger.error(f"[{symbol}] CSV yedeği indirilemedi: {str(e)}")
        
        return pd.DataFrame()

# ==============================================================================
# MATEMATİKSEL VE TEKNİK İNDİKATÖR MİMARİSİ
# ==============================================================================
class TechnicalAnalysisEngine:
    """
    İstatistiksel göstergeler, osilatörler ve hareketli ortalamalar
    için yüksek hassasiyetli hesaplama motoru.
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
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

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

    @classmethod
    def calculate_stochastic_oscillator(cls, df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        stoch_d = stoch_k.rolling(window=d_period).mean()
        return stoch_k, stoch_d

# ==============================================================================
# ÖNEMLİ KIRILIM VE SEVİYE ANALİZ MOTORU
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
        if self.df.empty or len(self.df) < 20:
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
        k, d = TechnicalAnalysisEngine.calculate_stochastic_oscillator(self.df)
        self.df['stoch_k'] = k
        self.df['stoch_d'] = d

    def find_support_resistance_levels(self, window: int = 15) -> dict:
        """
        Fraktal pivots yöntemini kullanarak kritik destek ve direnç seviyelerini tespit eder.
        """
        if len(self.df) < (window * 2):
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
                resistances.append(current_high)
            if is_support:
                supports.append(current_low)
                
        consolidated_resistances = self._cluster_levels(resistances)
        consolidated_supports = self._cluster_levels(supports)
        
        return {
            "supports": consolidated_supports,
            "resistances": consolidated_resistances
        }

    def _cluster_levels(self, levels: list, threshold_percent: float = 1.8) -> list:
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
        Belirlenen zaman aralığındaki en yüksek ve en düşük fiyatlara göre Fibonacci seviyelerini üretir.
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
        Direnç Kırılımı, Hacim Patlaması, Bollinger Sıkışması ve Indikatör Uyumunu Analiz Eder.
        """
        if len(self.df) < 50:
            return {"breakout_detected": False, "reason": "Yetersiz zaman serisi verisi."}
            
        last_row = self.df.iloc[-1]
        prev_row = self.df.iloc[-2]
        sr_levels = self.find_support_resistance_levels()
        resistances = sr_levels["resistances"]
        supports = sr_levels["supports"]
        
        current_price = last_row['close']
        current_volume = last_row['volume']
        avg_volume = last_row['vol_sma_20'] if not pd.isna(last_row['vol_sma_20']) else 1.0
        
        # 1. Hacim Teyit Analizi
        volume_spike = current_volume > (avg_volume * CONFIG["BREAKOUT_VOLUME_FACTOR"])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # 2. Direnç Kırılım Analizi
        broken_resistances = [r for r in resistances if prev_row['close'] <= r and current_price > r]
        is_resistance_breakout = len(broken_resistances) > 0
        
        # 3. Bollinger Bandı Kırılımı
        is_bb_breakout = current_price > last_row['bb_upper'] and volume_spike
        
        # 4. Golden Cross Analizi
        golden_cross = (prev_row['sma_50'] <= prev_row['sma_200']) and (last_row['sma_50'] > last_row['sma_200'])
        
        # Kırılım Gücü Skorlaması (0 - 100 Puan)
        score = 0
        if is_resistance_breakout:
            score += 35
        if volume_spike:
            score += 25
        if is_bb_breakout:
            score += 20
        if 50 < last_row['rsi'] < 70:
            score += 10
        if last_row['macd_hist'] > 0 and prev_row['macd_hist'] <= 0:
            score += 10
            
        breakout_type = "NÖTR / SAKİN"
        if score >= 70:
            breakout_type = "GÜÇLÜ BOĞA KIRILIMI (STRONG BULLISH BREAKOUT)"
        elif score >= 40:
            breakout_type = "POTANSİYEL KIRILIM (POTENTIAL BREAKOUT)"
        elif supports and current_price < min(supports):
            breakout_type = "GÜÇLÜ AYI KIRILIMI (BEARISH BREAKDOWN)"

        return {
            "breakout_detected": score >= 40,
            "breakout_score": score,
            "breakout_type": breakout_type,
            "volume_spike": volume_spike,
            "volume_ratio": round(float(volume_ratio), 2),
            "broken_resistances": [round(float(x), 2) for x in broken_resistances],
            "bollinger_breakout": is_bb_breakout,
            "golden_cross": golden_cross,
            "current_rsi": float(last_row['rsi']),
            "macd_signal_cross": bool(last_row['macd'] > last_row['macd_signal']),
            "stoch_k": float(last_row['stoch_k']),
            "stoch_d": float(last_row['stoch_d']),
            "atr_value": float(last_row['atr']) if not pd.isna(last_row['atr']) else 0.0,
            "near_supports": [round(float(s), 2) for s in supports if abs(s - current_price) / current_price < 0.03],
            "near_resistances": [round(float(r), 2) for r in resistances if abs(r - current_price) / current_price < 0.03]
        }

# ==============================================================================
# RİSK YÖNETİMİ VE POZİSYON BÜYÜKLÜĞÜ HESAPLAYICI
# ==============================================================================
class RiskManagementEngine:
    """
    ATR tabanlı Stop-Loss, Take-Profit ve Risk/Ödül oranı hesaplama sınıfı.
    """
    def __init__(self, entry_price: float, atr: float):
        self.entry_price = float(entry_price)
        self.atr = float(atr) if atr > 0 else float(entry_price) * 0.02

    def calculate_trade_setup(self, risk_reward_ratio: float = 2.0, atr_multiplier: float = 1.5) -> dict:
        stop_loss = self.entry_price - (self.atr * atr_multiplier)
        risk_per_share = self.entry_price - stop_loss
        take_profit = self.entry_price + (risk_per_share * risk_reward_ratio)
        
        return {
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "risk_per_share": round(risk_per_share, 2),
            "risk_reward_ratio": risk_reward_ratio
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
    def generate_full_report(symbol: str, live_data: dict, breakout_info: dict, fib_levels: dict, risk_setup: dict):
        print("=" * 85)
        print(f"                CANLI HİSSE VE KIRILIM ANALİZ RAPORU: {symbol}")
        print("=" * 85)
        print(f"Rapor Tarihi/Saat   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Veri Kaynağı        : {live_data.get('data_source', 'TradingView Harici Alternatif API')}")
        print("-" * 85)
        
        print("1. CANLI PİYASA VERİLERİ (REAL-TIME METRICS)")
        if live_data and "current_price" in live_data:
            curr = live_data.get('current_price', 0)
            chg = live_data.get('change', 0)
            pct = live_data.get('change_percent', 0)
            unit = live_data.get('currency', 'USD')
            print(f"   - Anlık Fiyat    : {curr:.2f} {unit}")
            print(f"   - Günlük Değişim : %{pct:.2f} ({chg:+.2f})")
            print(f"   - Günlük Yüksek  : {live_data.get('high', 0):.2f}")
            print(f"   - Günlük Düşük   : {live_data.get('low', 0):.2f}")
            print(f"   - Hacim (Volume)  : {live_data.get('volume', 0):,}")
            print(f"   - Piyasa Değeri   : {live_data.get('market_cap', 'N/A')}")
        else:
            print("   [!] Canlı piyasa verisi çekilemedi.")
            
        print("-" * 85)
        print("2. ÖNEMLİ KIRILIM VE SİNYAL ANALİZİ (BREAKOUT ANALYSIS)")
        print(f"   - Kırılım Durumu   : {'EVET' if breakout_info.get('breakout_detected') else 'HAYIR'}")
        print(f"   - Kırılım Skoru    : {breakout_info.get('breakout_score', 0)} / 100")
        print(f"   - Sinyal Tipi      : {breakout_info.get('breakout_type', 'N/A')}")
        print(f"   - Hacim Patlaması  : {'VAR (' + str(breakout_info.get('volume_ratio')) + 'x kat)' if breakout_info.get('volume_spike') else 'YOK'}")
        print(f"   - Bollinger Kırımı : {'VAR' if breakout_info.get('bollinger_breakout') else 'YOK'}")
        print(f"   - Golden Cross     : {'VAR' if breakout_info.get('golden_cross') else 'YOK'}")
        print(f"   - RSI (14) Değeri  : {breakout_info.get('current_rsi', 0):.2f}")
        print(f"   - Stokastik K/D    : {breakout_info.get('stoch_k', 0):.2f} / {breakout_info.get('stoch_d', 0):.2f}")
        
        if breakout_info.get('broken_resistances'):
            print(f"   - Kırılan Dirençler: {breakout_info.get('broken_resistances')}")
        if breakout_info.get('near_resistances'):
            print(f"   - Yakın Dirençler  : {breakout_info.get('near_resistances')}")
        if breakout_info.get('near_supports'):
            print(f"   - Yakın Destekler  : {breakout_info.get('near_supports')}")
            
        print("-" * 85)
        print("3. FIBONACCI DÜZELTME SEVİYELERİ (FIBONACCI RETRACEMENT)")
        for key, val in fib_levels.items():
            print(f"   - {key:14s} : {val:.2f}")

        print("-" * 85)
        print("4. RİSK YÖNETİMİ VE HEDEF SEVİYELERİ (RISK MANAGEMENT)")
        if risk_setup:
            print(f"   - Giriş Fiyatı    : {risk_setup.get('entry_price')}")
            print(f"   - Stop-Loss (Zarar Durdur): {risk_setup.get('stop_loss')}")
            print(f"   - Take-Profit (Kar Al)    : {risk_setup.get('take_profit')}")
            print(f"   - Risk/Ödül Oranı : 1 / {risk_setup.get('risk_reward_ratio')}")
            
        print("=" * 85 + "\n")

# ==============================================================================
# OTOMATİK VERİ DEPOLAMA VE YEDEKLEME MODÜLÜ
# ==============================================================================
class NumpyJsonEncoder(json.JSONEncoder):
    """
    NumPy veri tiplerini (np.bool_, np.float64, np.int64 vb.) 
    standart Python nesnelerine dönüştüren özel JSON kodlayıcı.
    """
    def default(self, obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class AnalysisDataExporter:
    """
    Analiz sonuçlarını JSON veya CSV dosyası olarak dışa aktaran sistem.
    """
    @staticmethod
    def export_to_json(data: dict, filename: str = "breakout_report.json"):
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, cls=NumpyJsonEncoder, ensure_ascii=False, indent=4)
            logger.info(f"Rapor başarıyla {filename} dosyasına aktarıldı.")
        except Exception as e:
            logger.error(f"JSON dışa aktarma hatası: {str(e)}")

# ==============================================================================
# SİSTEM ÇALIŞTIRICI / MAIN ORCHESTRATOR
# ==============================================================================
def main():
    logger.info("Gelişmiş Canlı Hisse Kırılım Analiz Sistemi Başlatılıyor...")
    
    target_symbols = ["THYAO.IS", "GARAN.IS", "AAPL", "MSFT", "NVDA", "TSLA"]
    
    fetcher = LiveMarketDataFetcher()
    all_reports = {}
    
    for symbol in target_symbols:
        logger.info(f"\n>>>> [{symbol}] İÇİN ANALİZ SÜRECİ BAŞLATILDI <<<<")
        
        live_data = fetcher.fetch_yahoo_live_ticker(symbol)
        df_hist = fetcher.fetch_historical_ohlcv(symbol, days=CONFIG["LOOKBACK_PERIOD_DAYS"])
        
        if df_hist.empty:
            logger.warning(f"[{symbol}] Yetersiz tarihsel veri nedeniyle analiz atlanıyor.")
            continue
            
        analyzer = BreakoutAnalysisEngine(df_hist)
        breakout_results = analyzer.analyze_breakout_signals()
        fib_levels = analyzer.calculate_fibonacci_retracements()
        
        current_p = live_data.get("current_price", df_hist['close'].iloc[-1])
        atr_v = breakout_results.get("atr_value", 0.0)
        risk_engine = RiskManagementEngine(current_p, atr_v)
        risk_setup = risk_engine.calculate_trade_setup()
        
        MarketReportGenerator.generate_full_report(
            symbol, live_data, breakout_results, fib_levels, risk_setup
        )
        
        all_reports[symbol] = {
            "live_data": live_data,
            "breakout": breakout_results,
            "fibonacci": fib_levels,
            "risk": risk_setup
        }
        
        time.sleep(1.5)

    AnalysisDataExporter.export_to_json(all_reports)

if __name__ == "__main__":
    main()
