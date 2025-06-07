import asyncio
import requests
import pandas as pd
import mplfinance as mpf
import matplotlib
matplotlib.use("Agg")  # penting untuk VPS/headless
import matplotlib.pyplot as plt
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from telegram import InputFile
from telegram.ext import Application
from telegram.constants import ParseMode
from datetime import datetime

# === CONFIG ===
API_TELEGRAM_BOT = '7614084480:AAEvOO2OdfBgaVLt_dPhwPbMLRW7sKAY0Nc'
CHAT_ID = '5986744500'
SYMBOL = 'BTCUSDT'
INTERVALS = ['1h', '4h', '1d']
LIMIT = 100
SLEEP_TIME = 900  # 15 menit

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
    rsi = RSIIndicator(df['close'], window=9).rsi()
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

def plot_chart_with_annotations(df, analysis, filename='chart.png', timeframe='1H', symbol='BTCUSDT'):
    ema13 = EMAIndicator(df['close'], window=13).ema_indicator()
    ema21 = EMAIndicator(df['close'], window=21).ema_indicator()
    df_plot = df[['open', 'high', 'low', 'close', 'volume']].copy()
    df_plot.index.name = 'Date'

    # Custom marketcolors: up candle oranye, down merah
    mc = mpf.make_marketcolors(up='orange', down='red', wick='black', edge='black', volume='orange')

    # Custom style dari binance, tapi ganti marketcolors
    s = mpf.make_mpf_style(base_mpf_style='binance', marketcolors=mc)

    ap0 = mpf.make_addplot(ema13, color='cyan', width=1.5)
    ap1 = mpf.make_addplot(ema21, color='orange', width=1.5)

    fig, axlist = mpf.plot(df_plot, type='candle', volume=True, style=s,
                           addplot=[ap0, ap1], returnfig=True, figsize=(12,8))
    fig.set_dpi(150)
    ax = axlist[0]  # candlestick axes
    ax_vol = axlist[2]  # volume axes (default posisi di bawah)

    # Garis horizontal Order Block
    ob_price = analysis.get('order_block_price')
    if ob_price:
        ax.axhline(ob_price, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Order Block')

    # Tambahkan legend manual (EMA13, EMA21, Order Block)
    ax.legend(['EMA13 (cyan)', 'EMA21 (orange)', 'Order Block (green dashed)'])

    # Tambahkan tulisan di kiri atas chart
    ax.text(0.01, 0.95, f"{symbol} Timeframe {timeframe}",
            transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left', color='white',
            bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))

    # Tambahkan tulisan di bawah chart (pas di bawah candlestick)
    last_price = analysis.get('last_close')
    fig.text(0.01, 0.02, f"Harga saat ini: {last_price:.2f}",
             fontsize=11, fontweight='bold', color='black', ha='left')

    # --- Posisi volume axis di kanan (opsional dan sedikit tricky) ---
    # Default volume axis di bawah candlestick dan di kiri
    # Untuk pindah ke kanan, perlu ubah posisi axis, contohnya seperti ini:

    # Pindahkan volume axis ke kanan bawah (sebatas mungkin)
    ax_vol_pos = ax_vol.get_position()
    new_pos = [0.85, ax_vol_pos.y0, ax_vol_pos.width, ax_vol_pos.height]
    ax_vol.set_position(new_pos)
    ax_vol.yaxis.tick_right()
    ax_vol.yaxis.set_label_position("right")

    # Tambahkan label volume di kanan bawah
    ax_vol.set_ylabel('Volume', rotation=270, labelpad=15, fontsize=10)

    # Save dan tutup
    fig.savefig(filename, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def build_message(all_analysis):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    msg = f"📡 BTC/USDT MULTI-TF SIGNAL ({now})\n"
    summary_score = 0
    buy_tf = 0
    sell_tf = 0

    for tf, analysis in all_analysis.items():
        rsi_div = analysis['rsi_divergence'] or 'None'
        stoch_div = analysis['stoch_divergence'] or 'None'

        msg += (f"\n⏱️ Timeframe: {tf}\n"
                f"🔹 Trend: {analysis['trend']}\n"
                f"🔹 EMA13: {analysis['ema13']:.2f}\n"
                f"🔹 EMA21: {analysis['ema21']:.2f}\n"
                f"🔹 RSI: {analysis['rsi']:.2f}\n"
                f"🔹 Order Block: {analysis['order_block_price']:.2f}\n"
                f"🔹 RSI Divergence: {rsi_div}\n"
                f"🔹 Stoch K: {analysis['stoch_k']:.2f}\n"
                f"🔹 Stoch D: {analysis['stoch_d']:.2f}\n"
                f"🔹 Stoch Status: {analysis['stoch_status']}\n"
                f"🔹 Stoch Divergence: {stoch_div}\n"
                f"🔹 Score: {analysis['score']}\n"
                f"🔹 Signal: {analysis['signal']}\n")

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

    msg += f"\n📊 Ringkasan: {summary}\n"
    msg += "\n#BTC #MultiTimeframe #Signal #Trading"

    if len(msg) > 1000:
        msg = msg[:1000] + "\n\n[Pesan terpotong karena terlalu panjang]"

    return msg

async def main_loop(application):
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
            with open('chart.png', 'rb') as photo:
                await application.bot.send_photo(chat_id=CHAT_ID, photo=InputFile(photo), caption=msg, parse_mode=ParseMode.MARKDOWN)

            print("[INFO] Sinyal terkirim.")
        except Exception as e:
            print(f"[ERROR] {e}")
        await asyncio.sleep(SLEEP_TIME)

import asyncio

if __name__ == "__main__":
    app = Application.builder().token(API_TELEGRAM_BOT).build()
    asyncio.run(main_loop(app))
