"""
Crypto market data via ccxt.
Supports ticker formats: BTC, ETH, BTC/USDT, ETH/USDT
Default exchange: Binance
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta

# EXCHANGE = ccxt.binance()
EXCHANGE = ccxt.coinbase({
    'proxies': {
        'http': 'http://127.0.0.1:7897',   # 换成你实际的端口
        'https': 'http://127.0.0.1:7897',
    }
})

CRYPTO_SYMBOLS = {
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA",
    "AVAX", "MATIC", "DOT", "LINK", "UNI", "ATOM", "LTC",
}


def normalize_symbol(ticker: str) -> str:
    """Normalize ticker to exchange pair format, e.g. BTC -> BTC/USDT"""
    ticker = ticker.upper().strip()
    if "/" not in ticker:
        return f"{ticker}/USDT"
    return ticker


def is_crypto(ticker: str) -> bool:
    """Return True if ticker is a recognized crypto symbol or pair."""
    t = ticker.upper().strip()
    if "/" in t:
        return True
    base = t.split("-")[0]
    return base in CRYPTO_SYMBOLS


def get_crypto_ohlcv(symbol: str, start_date: str, end_date: str) -> str:
    """
    Fetch daily OHLCV data for a crypto pair from Binance.

    Args:
        symbol: Ticker symbol, e.g. BTC or BTC/USDT
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted string of OHLCV data
    """
    try:
        pair = normalize_symbol(symbol)
        since = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        all_ohlcv = []
        while since < end_ts:
            ohlcv = EXCHANGE.fetch_ohlcv(pair, timeframe="1d", since=since, limit=500)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 86400000

        if not all_ohlcv:
            return f"No data found for {symbol}"

        df = pd.DataFrame(
            all_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime("%Y-%m-%d")
        df = df[df["date"] <= end_date][["date", "open", "high", "low", "close", "volume"]]
        df = df.round(4)

        return f"OHLCV data for {pair} ({start_date} to {end_date}):\n{df.to_string(index=False)}"

    except Exception as e:
        return f"Error fetching crypto data for {symbol}: {str(e)}"


def get_crypto_indicators(symbol: str, indicator: str, curr_date: str, look_back_days: int = 30) -> str:
    """
    Compute technical indicators from Binance OHLCV data.

    Args:
        symbol: Ticker symbol, e.g. BTC or BTC/USDT
        indicator: Indicator name, e.g. rsi, macd, sma, ema, boll, atr
        curr_date: Current trading date in yyyy-mm-dd format
        look_back_days: Number of days to look back (default 30)

    Returns:
        Formatted string of indicator values
    """
    try:
        pair = normalize_symbol(symbol)
        end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=look_back_days + 50)

        since = int(start_dt.timestamp() * 1000)
        ohlcv = EXCHANGE.fetch_ohlcv(pair, timeframe="1d", since=since, limit=look_back_days + 50)

        if not ohlcv:
            return f"No data for {symbol}"

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime("%Y-%m-%d")
        df = df[df["date"] <= curr_date].tail(look_back_days)
        close = df["close"]

        ind = indicator.lower().strip()
        result = ""

        if ind == "rsi":
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss
            df["rsi"] = (100 - 100 / (1 + rs)).round(2)
            result = df[["date", "close", "rsi"]].dropna().to_string(index=False)

        elif ind == "macd":
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = (ema12 - ema26).round(4)
            signal = macd.ewm(span=9).mean().round(4)
            df["macd"] = macd
            df["signal"] = signal
            df["histogram"] = (macd - signal).round(4)
            result = df[["date", "close", "macd", "signal", "histogram"]].dropna().to_string(index=False)

        elif ind in ("sma", "close_50_sma"):
            period = 50 if "50" in ind else 20
            df[f"sma_{period}"] = close.rolling(period).mean().round(4)
            result = df[["date", "close", f"sma_{period}"]].dropna().to_string(index=False)

        elif ind in ("ema", "close_10_ema"):
            period = 10 if "10" in ind else 20
            df[f"ema_{period}"] = close.ewm(span=period).mean().round(4)
            result = df[["date", "close", f"ema_{period}"]].dropna().to_string(index=False)

        elif ind in ("boll", "boll_lb"):
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            df["boll_upper"] = (sma + 2 * std).round(4)
            df["boll_mid"] = sma.round(4)
            df["boll_lower"] = (sma - 2 * std).round(4)
            result = df[["date", "close", "boll_upper", "boll_mid", "boll_lower"]].dropna().to_string(index=False)

        elif ind == "atr":
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df["atr"] = tr.rolling(14).mean().round(4)
            result = df[["date", "close", "atr"]].dropna().to_string(index=False)

        else:
            result = (
                f"Indicator '{indicator}' not implemented for crypto. "
                f"Available: rsi, macd, sma, ema, boll, atr"
            )

        return f"{indicator.upper()} for {pair} (last {look_back_days} days up to {curr_date}):\n{result}"

    except Exception as e:
        return f"Error computing {indicator} for {symbol}: {str(e)}"


def get_crypto_news(ticker: str, start_date: str, end_date: str) -> str:
    """
    Return crypto news placeholder.
    Real-time news requires CryptoPanic or similar API.

    Args:
        ticker: Crypto symbol, e.g. BTC or BTC/USDT
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Placeholder string directing to news sources
    """
    symbol = ticker.split("/")[0].upper()
    return (
        f"Crypto news for {symbol} ({start_date} to {end_date}):\n"
        f"Real-time crypto news requires CryptoPanic or similar API. "
        f"Refer to CoinDesk, CoinTelegraph, and The Block for {symbol} coverage."
    )


def get_crypto_fundamentals(ticker: str, curr_date: str) -> str:
    """
    Return crypto fundamentals placeholder.
    On-chain metrics require CoinGecko or Glassnode API.

    Args:
        ticker: Crypto symbol, e.g. BTC or BTC/USDT
        curr_date: Current date in yyyy-mm-dd format

    Returns:
        Placeholder string with guidance on key metrics
    """
    symbol = ticker.split("/")[0].upper()
    return (
        f"Fundamental data for {symbol} as of {curr_date}:\n"
        f"On-chain metrics require CoinGecko or Glassnode API. "
        f"Key metrics: market cap rank, circulating supply, "
        f"24h trading volume, developer activity."
    )