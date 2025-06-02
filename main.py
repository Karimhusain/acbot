import asyncio
import json
import pandas as pd
from datetime import datetime, timezone
from ta.trend import EMAIndicator
from ta.momentum import StochasticOscillator
from telegram import Bot, ParseMode
import websockets

# === KONFIGURASI ===
# PENTING: Ganti nilai-nilai placeholder ini dengan milik Anda!
TELEGRAM_TOKEN = "7614084480:AAEvOO2OdfBgaVLt_dPhwPbMLRW7sKAY0Nc"  # Ganti dengan token bot Telegram Anda
TELEGRAM_CHAT_ID = 5986744500       # Ganti dengan ID chat Telegram Anda, contoh: 123456789

# Daftar Timeframe yang akan dipantau
TIMEFRAMES = ["1m", "1h", "4h", "1w"]
BASE_BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@kline_" 

# Simpan candles untuk setiap timeframe
# {'1m': DataFrame, '1h': DataFrame, '4h': DataFrame, '1w': DataFrame}
all_candles = {tf: pd.DataFrame() for tf in TIMEFRAMES}

# Parameter Indikator (bisa disesuaikan per timeframe jika mau, tapi ini global)
EMA_FAST = 13
EMA_SLOW = 21
STOCH_K = 5
STOCH_D = 3

# Timestamp terakhir sinyal dikirim per timeframe
last_sent = {tf: None for tf in TIMEFRAMES}
# Ambang batas skor untuk mengirim sinyal (misal, 0 agar selalu terkirim)
SIGNAL_CONFIDENCE_THRESHOLD = 0 # Ubah jika ingin sinyal yang lebih kuat saja

# === FUNGSI UTILITAS ===

# Fungsi untuk memperbarui data candlestick dari stream kline
def update_candles(data, timeframe):
    global all_candles
    k = data["k"]
    # Hanya proses candle yang sudah close (terakhir)
    if not k["x"]:  
        return

    open_time = pd.to_datetime(k["t"], unit="ms", utc=True)
    open_p = float(k["o"])
    high_p = float(k["h"])
    low_p = float(k["l"])
    close_p = float(k["c"])
    volume = float(k["v"])

    new_row = pd.DataFrame([{
        "open_time": open_time,
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close_p,
        "volume": volume,
    }])

    current_df = all_candles[timeframe]
    current_df = pd.concat([current_df, new_row], ignore_index=True)
    # Batasi jumlah baris untuk efisiensi
    if len(current_df) > 500: 
        current_df = current_df.iloc[-500:].reset_index(drop=True)
    all_candles[timeframe] = current_df # Update DataFrame di dictionary

# Hitung indikator teknikal di DataFrame
def calculate_indicators(df):
    df_copy = df.copy()
    
    # Pastikan ada cukup data sebelum menghitung EMA
    if len(df_copy) >= EMA_SLOW:
        df_copy["EMA13"] = EMAIndicator(df_copy["close"], EMA_FAST).ema_indicator()
        df_copy["EMA21"] = EMAIndicator(df_copy["close"], EMA_SLOW).ema_indicator()
    else: 
        df_copy["EMA13"] = pd.NA
        df_copy["EMA21"] = pd.NA

    # Pastikan ada cukup data sebelum menghitung Stochastic
    if len(df_copy) >= STOCH_K + STOCH_D:
        stoch = StochasticOscillator(df_copy["high"], df_copy["low"], df_copy["close"], window=STOCH_K, smooth_window=STOCH_D)
        df_copy["stoch_k"] = stoch.stoch()
        df_copy["stoch_d"] = stoch.stoch_signal()
    else: 
        df_copy["stoch_k"] = pd.NA
        df_copy["stoch_d"] = pd.NA
        
    return df_copy


# Fungsi untuk mengecek kondisi teknikal
def check_bullish_cross(df):
    if len(df) < 2 or df["EMA13"].iloc[-1] is pd.NA or df["EMA21"].iloc[-1] is pd.NA or \
       df["EMA13"].iloc[-2] is pd.NA or df["EMA21"].iloc[-2] is pd.NA:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    return (prev["EMA13"] < prev["EMA21"]) and (curr["EMA13"] > curr["EMA21"])


def check_price_above_emas(df):
    if len(df) < 1 or df["EMA13"].iloc[-1] is pd.NA or df["EMA21"].iloc[-1] is pd.NA:
        return False
    last_close = df.iloc[-1]["close"]
    ema13 = df.iloc[-1]["EMA13"]
    ema21 = df.iloc[-1]["EMA21"]
    return last_close > ema13 and last_close > ema21


def check_stoch_oversold_cross(df):
    if len(df) < 2 or df["stoch_k"].iloc[-1] is pd.NA or df["stoch_d"].iloc[-1] is pd.NA or \
       df["stoch_k"].iloc[-2] is pd.NA or df["stoch_d"].iloc[-2] is pd.NA:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    cross_up = (prev["stoch_k"] < prev["stoch_d"]) and (curr["stoch_k"] > curr["stoch_d"])
    oversold = curr["stoch_k"] < 20 and curr["stoch_d"] < 20
    return cross_up and oversold


def detect_divergence(df, lookback=20, oscillator_col="stoch_k"):
    if len(df) < lookback:
        return False

    temp_df = df.iloc[-lookback:].copy()

    if temp_df['low'].isnull().any() or temp_df[oscillator_col].isnull().any():
        return False

    lows = temp_df['low']
    osc = temp_df[oscillator_col]

    low_points_price_mask = (lows.shift(1) > lows) & (lows.shift(-1) > lows)
    low_points_osc_mask = (osc.shift(1) > osc) & (osc.shift(-1) > osc)

    low_points_price_idx = lows[low_points_price_mask].index.tolist()
    low_points_osc_idx = osc[low_points_osc_mask].index.tolist()

    if len(low_points_price_idx) < 2 or len(low_points_osc_idx) < 2:
        return False

    idx_price1 = low_points_price_idx[-2]
    idx_price2 = low_points_price_idx[-1]

    idx_osc1 = low_points_osc_idx[-2]
    idx_osc2 = low_points_osc_idx[-1]

    tolerance = lookback * 0.2
    
    is_osc_low1_near_price_low1 = abs(idx_osc1 - idx_price1) <= tolerance
    is_osc_low2_near_price_low2 = abs(idx_osc2 - idx_price2) <= tolerance

    if not (is_osc_low1_near_price_low1 and is_osc_low2_near_price_low2):
        return False

    price_lower_low = df.loc[idx_price2, 'low'] < df.loc[idx_price1, 'low']
    osc_higher_low = df.loc[idx_osc2, oscillator_col] > df.loc[idx_osc1, oscillator_col]

    return price_lower_low and osc_higher_low


def check_volume(df):
    if len(df) < 20: 
        return False
    df_copy = df.copy()
    if df_copy["volume"].isnull().all():
        return False
    df_copy["vol_ma20"] = df_copy["volume"].rolling(window=20).mean()
    
    if df_copy["vol_ma20"].iloc[-1] is pd.NA:
        return False
    return df_copy.iloc[-1]["volume"] > df_copy.iloc[-1]["vol_ma20"]


def compute_confidence(valid_conditions):
    if not valid_conditions:
        return 0.0
    score = sum(valid_conditions)
    return round(score / len(valid_conditions) * 10, 1)


# Fungsi untuk escape karakter khusus MarkdownV2
def escape_markdown_v2(text):
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return ''.join(['\\' + char if char in escape_chars else char for char in text])


# === PERUBAHAN DI SINI: format_signal_output ===
def format_signal_output(signal_data):
    technicals = []
    for cond, valid in signal_data["technical_conditions"].items():
        emoji = "✅" if valid else "❌"
        # Escape nama kondisi juga jika mengandung karakter khusus
        technicals.append(f"{emoji} {escape_markdown_v2(cond)}")

    technical_str = "\n".join(technicals)

    # Escape semua data yang dimasukkan ke dalam pesan
    symbol = escape_markdown_v2(signal_data['symbol'])
    timeframe = escape_markdown_v2(signal_data['timeframe'])
    entry_type = escape_markdown_v2(signal_data['entry_type'])
    
    entry_price = escape_markdown_v2(f"{signal_data['entry_price']:.2f}")
    target_low = escape_markdown_v2(f"{signal_data['target_range'][0]:.2f}")
    target_high = escape_markdown_v2(f"{signal_data['target_range'][1]:.2f}")
    stop_loss = escape_markdown_v2(f"{signal_data['stop_loss']:.2f}")
    
    confidence_score_val = f"{signal_data['confidence']} / 10"
    confidence_score_escaped = escape_markdown_v2(confidence_score_val)
    
    time_str = escape_markdown_v2(signal_data["time"].strftime("%d %b %Y - %H:%M UTC"))

    msg = (
        f"🚀 *SIGNAL BUY DETECTED*\n"
        f"*Symbol:* {symbol}\n"
        f"*Timeframe:* {timeframe}\n"
        f"*Entry Type:* {entry_type}\n"
        f"*Entry Price:* `${entry_price}`\n"
        f"*Target:* `${target_low}` \\- `${target_high}`\n" # Escape hyphen jika diapit oleh backtick
        f"*Stop Loss:* `${stop_loss}`\n\n"
        f"*Confidence:* `{confidence_score_escaped}`\n\n"
        f"*Conditions:*\n{technical_str}\n\n"
        f"_Generated at:_ `{time_str}`"
    )
    return msg

# === PERUBAHAN DI SINI: send_signal_to_telegram ===
async def send_signal_to_telegram(msg: str):
    bot = Bot(token=TELEGRAM_TOKEN)
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

# === PERUBAHAN DI SINI: handle_websocket ===
async def handle_websocket(timeframe):
    url = BASE_BINANCE_WS_URL + timeframe
    while True: # Loop untuk reconnect otomatis
        try:
            async with websockets.connect(url) as ws:
                print(f"Connected to Binance WebSocket for {timeframe}...")
                async for raw in ws:
                    data = json.loads(raw)
                    # Pastikan ini adalah event kline dan data 'k' ada
                    if data.get('e') == 'kline' and 'k' in data:
                        # Kita hanya peduli dengan candle yang sudah closed
                        if not data['k']['x']:
                            continue # Skip candle yang masih berjalan

                        update_candles(data, timeframe)

                        df = all_candles[timeframe]
                        # Minimal data yang cukup untuk menghitung semua indikator
                        required_candles = max(EMA_SLOW, STOCH_K + STOCH_D, 20) 
                        if df.empty or len(df) < required_candles:
                            # print(f"[{timeframe}] Not enough data ({len(df)}/{required_candles} candles). Skipping analysis.")
                            continue

                        df_analyzed = calculate_indicators(df)
                        
                        # Pastikan tidak ada NaN di kolom terakhir untuk perhitungan kondisi
                        if df_analyzed.iloc[-1].isnull().any():
                            # print(f"[{timeframe}] Warning: NaN values found in indicator columns for the latest candle. Skipping signal analysis.")
                            continue

                        now = datetime.now(timezone.utc)

                        conditions = {
                            "EMA13 Cross Above EMA21": check_bullish_cross(df_analyzed), # Gunakan df_analyzed
                            "Price Above EMA13 & EMA21": check_price_above_emas(df_analyzed), # Gunakan df_analyzed
                            "Stochastic Oversold Cross": check_stoch_oversold_cross(df_analyzed), # Gunakan df_analyzed
                            "Bullish Divergence": detect_divergence(df_analyzed), # Gunakan df_analyzed
                            "Volume > MA20": check_volume(df_analyzed) # Gunakan df_analyzed
                        }

                        confidence = compute_confidence(list(conditions.values()))
                        last = last_sent[timeframe]

                        # Kirim sinyal jika ambang batas confidence terpenuhi dan sudah melewati jeda waktu
                        if confidence >= SIGNAL_CONFIDENCE_THRESHOLD and (last is None or (now - last).total_seconds() > 1800):
                            last_sent[timeframe] = now
                            price = df_analyzed.iloc[-1]["close"] # Gunakan df_analyzed
                            signal = {
                                "symbol": "BTC/USDT",
                                "timeframe": timeframe,
                                "entry_type": "LONG",
                                "entry_price": price,
                                "target_range": (price * 1.01, price * 1.03), # Target 1-3%
                                "stop_loss": price * 0.99, # Stop Loss 1%
                                "confidence": confidence,
                                "technical_conditions": conditions,
                                "time": now
                            }
                            message = format_signal_output(signal)
                            print(f"[{timeframe}] Sending signal with score: {confidence}")
                            await send_signal_to_telegram(message)

        except websockets.exceptions.ConnectionClosedOK:
            print(f"WebSocket connection for {timeframe} closed normally. Reconnecting in 5 seconds...")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"WebSocket connection for {timeframe} closed with error: {e}. Reconnecting in 5 seconds...")
        except Exception as e:
            print(f"An unexpected error occurred for {timeframe}: {e}. Reconnecting in 5 seconds...")
        
        await asyncio.sleep(5) # Jeda sebelum mencoba reconnect

# === PERUBAHAN DI SINI: main function ===
async def main():
    # Loop event dibiarkan secara default oleh asyncio.run()
    # Bot object perlu di-instantiate di dalam send_signal_to_telegram jika tidak diteruskan
    # Alternatifnya, bisa dibuat global atau diteruskan ke handle_websocket
    # Untuk kesederhanaan, biarkan send_signal_to_telegram membuat instansinya sendiri untuk setiap pengiriman.
    # Namun, lebih efisien jika bot dibuat sekali dan diteruskan.
    # Karena send_signal_to_telegram tidak menerima bot, maka tidak ada perubahan di sini.

    # Buat daftar tugas untuk setiap koneksi WebSocket
    tasks = [handle_websocket(tf) for tf in TIMEFRAMES]

    # Jalankan semua tugas secara bersamaan
    print("Starting all WebSocket handlers...")
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    # Pastikan TELEGRAM_TOKEN dan TELEGRAM_CHAT_ID telah diisi dengan benar
    if TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN" or not isinstance(TELEGRAM_CHAT_ID, int) or TELEGRAM_CHAT_ID == 0:
        print("ERROR: Mohon isi TELEGRAM_TOKEN dan TELEGRAM_CHAT_ID di bagian KONFIGURASI.")
        print("TELEGRAM_CHAT_ID harus berupa integer non-nol (contoh: 123456789 atau -123456789).")
        print("Skrip tidak dapat berjalan tanpa konfigurasi yang benar.")
    else:
        print("Starting the multi-timeframe signal bot...")
        asyncio.run(main())

