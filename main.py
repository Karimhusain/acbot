import time
import requests
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from telegram import Bot
from datetime import datetime
import asyncio

# === CONFIG ===
API_TELEGRAM_BOT = '7614084480:AAEvOO2OdfBgaVLt_dPhwPbMLRW7sKAY0Nc'
CHAT_ID = '5986744500'
SYMBOL = 'BTCUSDT'
INTERVALS = ['1h', '4h', '1d']
LIMIT = 100
SLEEP_TIME = 900  # 15 menit

bot = Bot(token=API_TELEGRAM_BOT)

def fetch_klines(symbol, interval, limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    res = requests.get(url)
    data = res.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df

def detect_rsi_divergence(df, rsi):
    for i in range(-5, -1):
        if df['close'].iloc[i] < df['close'].iloc[i-1] and rsi.iloc[i] > rsi.iloc[i-1]:
            return "📈 Bullish RSI Divergence"
        if df['close'].iloc[i] > df['close'].iloc[i-1] and rsi.iloc[i] < rsi.iloc[i-1]:
            return "📉 Bearish RSI Divergence"
    return None

def detect_stoch_divergence(df, stoch_k):
    for i in range(-5, -1):
        if df['close'].iloc[i] < df['close'].iloc[i-1] and stoch_k.iloc[i] > stoch_k.iloc[i-1]:
            return "📈 Bullish Stochastic Divergence"
        if df['close'].iloc[i] > df['close'].iloc[i-1] and stoch_k.iloc[i] < stoch_k.iloc[i-1]:
            return "📉 Bearish Stochastic Divergence"
    return None

def analyze_df(df):
    ema13 = EMAIndicator(df['close'], window=13).ema_indicator()
    ema21 = EMAIndicator(df['close'], window=21).ema_indicator()
    rsi = RSIIndicator(df['close'], window=14).rsi()
    stoch = StochasticOscillator(df['high'], df['low'], df['close'], window=5, smooth_window=3)
    stoch_k = stoch.stoch()
    stoch_d = stoch.stoch_signal()

    trend = "Bullish" if ema13.iloc[-1] > ema21.iloc[-1] else "Bearish"
    recent_high = df['high'].iloc[-20:].max()
    recent_low = df['low'].iloc[-20:].min()
    last_close = df['close'].iloc[-1]
    order_block_price = recent_low if trend == "Bullish" else recent_high

    rsi_div = detect_rsi_divergence(df, rsi)
    stoch_div = detect_stoch_divergence(df, stoch_k)

    stoch_status = "Neutral"
    if stoch_k.iloc[-1] > 80:
        stoch_status = "Overbought"
    elif stoch_k.iloc[-1] < 20:
        stoch_status = "Oversold"

    score = 0
    score += 2 if trend == "Bullish" else -2
    score += 1 if rsi.iloc[-1] > 50 else -1
    score += 2 if stoch_status == "Oversold" and trend == "Bullish" else 0
    score -= 2 if stoch_status == "Overbought" and trend == "Bearish" else 0
    score += 2 if rsi_div == "📈 Bullish RSI Divergence" else -2 if rsi_div == "📉 Bearish RSI Divergence" else 0
    score += 2 if stoch_div == "📈 Bullish Stochastic Divergence" else -2 if stoch_div == "📉 Bearish Stochastic Divergence" else 0

    signal = "BUY" if score >= 4 else "SELL" if score <= -4 else "WAIT"

    return {
        'ema13': ema13.iloc[-1],
        'ema21': ema21.iloc[-1],
        'rsi': rsi.iloc[-1],
        'trend': trend,
        'order_block_price': order_block_price,
        'rsi_divergence': rsi_div,
        'stoch_k': stoch_k.iloc[-1],
        'stoch_d': stoch_d.iloc[-1],
        'stoch_status': stoch_status,
        'stoch_divergence': stoch_div,
        'last_close': last_close,
        'score': score,
        'signal': signal
    }

def plot_chart_with_annotations(df, analysis, filename='chart.png'):
    ema13 = EMAIndicator(df['close'], window=13).ema_indicator()
    ema21 = EMAIndicator(df['close'], window=21).ema_indicator()
    df_plot = df[['open', 'high', 'low', 'close', 'volume']].copy()
    df_plot.index.name = 'Date'

    ap0 = mpf.make_addplot(ema13, color='blue')
    ap1 = mpf.make_addplot(ema21, color='orange')

    fig, axlist = mpf.plot(df_plot, type='candle', volume=True, style='yahoo',
                           addplot=[ap0, ap1], returnfig=True)
    ax = axlist[0]

    ob_price = analysis.get('order_block_price')
    if ob_price:
        ob_idx = (df['close'] - ob_price).abs().idxmin()
        if ob_idx:
            ax.annotate('Order Block', xy=(ob_idx, ob_price),
                        xytext=(ob_idx, ob_price * 1.02),
                        arrowprops=dict(facecolor='green', shrink=0.05),
                        fontsize=9, color='green')

    stoch_status = analysis.get('stoch_status')
    if stoch_status and stoch_status != "Neutral":
        ax.annotate(f'Stoch: {stoch_status}', xy=(df.index[-1], df['close'].iloc[-1]),
                    xytext=(df.index[-1], df['close'].iloc[-1] * 1.02),
                    arrowprops=dict(facecolor='purple', shrink=0.05),
                    fontsize=9, color='purple')

    fig.savefig(filename)
    plt.close(fig)

def build_message(all_analysis):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    msg = f"📡 BTC/USDT MULTI-TF SIGNAL ({now})\n"
    summary_score = 0
    buy_tf = 0
    sell_tf = 0

    for tf, analysis in all_analysis.items():
        msg += f"\n⏱️ Timeframe: {tf}\n"
        msg += f"🔹 Trend: {analysis['trend']}\n"
        msg += f"🔹 EMA13: {analysis['ema13']:.2f}\n"
        msg += f"🔹 EMA21: {analysis['ema21']:.2f}\n"
        msg += f"🔹 RSI: {analysis['rsi']:.2f}\n"
        msg += f"🔹 Order Block: {analysis['order_block_price']:.2f}\n"
        msg += f"🔹 RSI Divergence: {analysis['rsi_divergence'] or 'None'}\n"
        msg += f"🔹 Stochastic K: {analysis['stoch_k']:.2f}\n"
        msg += f"🔹 Stochastic D: {analysis['stoch_d']:.2f}\n"
        msg += f"🔹 Stoch Status: {analysis['stoch_status']}\n"
        msg += f"🔹 Stoch Divergence: {analysis['stoch_divergence'] or 'None'}\n"
        msg += f"🔹 Signal Score: {analysis['score']}\n"
        msg += f"🔹 Signal: {analysis['signal']}\n"

        summary_score += analysis['score']
        if analysis['signal'] == 'BUY':
            buy_tf += 1
        elif analysis['signal'] == 'SELL':
            sell_tf += 1

    if summary_score >= 8 and buy_tf >= 2:
        summary = "📈 POTENSI BUY KONSISTEN (Multi-Timeframe Support)"
    elif summary_score <= -8 and sell_tf >= 2:
        summary = "📉 POTENSI SELL KONSISTEN (Multi-Timeframe Support)"
    else:
        summary = "⚠️ SIGNAL MIXED / WAITING - KONSULTASI DENGAN KONDISI PASAR"

    msg += f"\n📊 Ringkasan Sinyal Akhir: {summary}\n"
    msg += "\n#BTC #MultiTimeframe #Signal #Trading"

    return msg

async def send_telegram_photo(bot, chat_id, photo_path, caption):
    with open(photo_path, 'rb') as photo:
        await bot.send_photo(chat_id=chat_id, photo=photo, caption=caption)

async def main():
    while True:
        try:
            all_analysis = {}
            for tf in INTERVALS:
                df = fetch_klines(SYMBOL, tf, LIMIT)
                analysis = analyze_df(df)
                all_analysis[tf] = analysis
                if tf == '1h':
                    plot_chart_with_annotations(df, analysis, filename='chart.png')

            msg = build_message(all_analysis)
            await send_telegram_photo(bot, CHAT_ID, 'chart.png', msg)
            print("[INFO] Sinyal terkirim.")
        except Exception as e:
            print(f"[ERROR] {e}")
        await asyncio.sleep(SLEEP_TIME)

if __name__ == "__main__":
    asyncio.run(main())
