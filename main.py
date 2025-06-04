import time
import requests
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from telegram import Bot
from datetime import datetime

# === CONFIG ===
API_TELEGRAM_BOT = '7614084480:AAEvOO2OdfBgaVLt_dPhwPbMLRW7sKAY0Nc'
CHAT_ID = '5986744500'
SYMBOL = 'BTCUSDT'
INTERVAL = '1h'   # time frame utama untuk chart dan kirim gambar
LIMIT = 100       # ambil 100 candle ke belakang
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

def analyze_df(df):
    # Hitung EMA13 dan EMA21
    ema13 = EMAIndicator(df['close'], window=13).ema_indicator()
    ema21 = EMAIndicator(df['close'], window=21).ema_indicator()

    # Hitung RSI
    rsi = RSIIndicator(df['close'], window=14).rsi()

    # Deteksi trend sederhana
    trend = "Bullish" if ema13.iloc[-1] > ema21.iloc[-1] else "Bearish"

    # Deteksi order block sederhana = candle high/low yang paling rendah/tinggi dalam 20 candle terakhir
    recent_high = df['high'].iloc[-20:].max()
    recent_low = df['low'].iloc[-20:].min()
    last_close = df['close'].iloc[-1]

    order_block_price = None
    if trend == "Bullish":
        order_block_price = recent_low
    else:
        order_block_price = recent_high

    # Deteksi Bullish Divergence RSI (sederhana): harga turun, RSI naik (bandingkan 2 candle terakhir)
    divergence = None
    if len(rsi) >= 3:
        if (df['close'].iloc[-2] < df['close'].iloc[-3]) and (rsi.iloc[-2] > rsi.iloc[-3]):
            divergence = "📈 Bullish RSI Divergence"
        elif (df['close'].iloc[-2] > df['close'].iloc[-3]) and (rsi.iloc[-2] < rsi.iloc[-3]):
            divergence = "📉 Bearish RSI Divergence"

    return {
        'ema13': ema13.iloc[-1],
        'ema21': ema21.iloc[-1],
        'rsi': rsi.iloc[-1],
        'trend': trend,
        'order_block_price': order_block_price,
        'divergence': divergence,
        'last_close': last_close,
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
    ob_idx = (df['close'] - ob_price).abs().idxmin() if ob_price else None

    if ob_idx:
        ax.annotate('Order Block', xy=(ob_idx, ob_price),
                    xytext=(ob_idx, ob_price * 1.02),
                    arrowprops=dict(facecolor='green', shrink=0.05),
                    fontsize=9, color='green')

    divergence = analysis.get('divergence')
    if divergence:
        last_candle_date = df.index[-1]
        last_close = df['close'].iloc[-1]
        ax.annotate(divergence, xy=(last_candle_date, last_close),
                    xytext=(last_candle_date, last_close * 1.05),
                    arrowprops=dict(facecolor='red', shrink=0.05),
                    fontsize=9, color='red')

    fig.savefig(filename)
    plt.close(fig)

def build_message(analysis):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    msg = (f"📡 BTC/USDT SIGNAL ({now})\n"
           f"💰 Price: ${analysis['last_close']:.2f}\n\n"
           f"🔹 Trend: {analysis['trend']}\n"
           f"🔹 EMA13: {analysis['ema13']:.2f}\n"
           f"🔹 EMA21: {analysis['ema21']:.2f}\n"
           f"🔹 RSI: {analysis['rsi']:.2f}\n"
           f"🔹 Order Block: {analysis['order_block_price']:.2f}\n"
           f"🔹 Divergence: {analysis['divergence'] or 'None'}\n\n")

    # Saran singkat berdasarkan kondisi
    if analysis['trend'] == "Bullish" and analysis['rsi'] < 30:
        msg += "📊 Signal: ✅ STRONG BUY ZONE\n"
    elif analysis['trend'] == "Bearish" and analysis['rsi'] > 70:
        msg += "📊 Signal: ⚠️ STRONG SELL ZONE\n"
    else:
        msg += "📊 Signal: ⚠️ Wait & See\n"

    msg += "\n#BTC #Signal #Trading"
    return msg

def main():
    last_message = ''
    while True:
        try:
            df = fetch_klines(SYMBOL, INTERVAL, LIMIT)
            analysis = analyze_df(df)
            plot_chart_with_annotations(df, analysis, filename='chart.png')
            msg = build_message(analysis)
            if msg != last_message:
                bot.send_photo(chat_id=CHAT_ID, photo=open('chart.png', 'rb'), caption=msg)
                last_message = msg
                print("Sent update to Telegram.")
            else:
                print("No change, skipping Telegram message.")
        except Exception as e:
            print(f"[ERROR] {e}")
        time.sleep(SLEEP_TIME)

if __name__ == "__main__":
    main()
