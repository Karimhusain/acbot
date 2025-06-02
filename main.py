import asyncio
import json
import pandas as pd
from datetime import datetime, timezone
from ta.trend import EMAIndicator
from ta.momentum import StochasticOscillator
from telegram import Bot
from telegram.constants import ParseMode
import websockets

# === KONFIGURASI ===
# PENTING: Ganti nilai-nilai placeholder ini dengan milik Anda!
TELEGRAM_TOKEN = "7614084480:AAEvOO2OdfBgaVLt_dPhwPbMLRW7sKAY0Nc"  # Ganti dengan token bot Telegram Anda
TELEGRAM_CHAT_ID = 5986744500       # Ganti dengan ID chat Telegram Anda, contoh: 123456789

# Daftar Timeframe yang akan dipantau
TIMEFRAMES = ["1m", "1h", "4h", "1w"]
BASE_BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@kline_" 

# Simpan candles untuk setiap timeframe
all_candles = {tf: pd.DataFrame() for tf in TIMEFRAMES}
# Status untuk melacak apakah candle terakhir sudah diproses untuk setiap timeframe
# Ini akan membantu memastikan kita hanya menganalisis candle yang sudah close di semua TFs
last_candle_processed_time = {tf: None for tf in TIMEFRAMES}

# Parameter Indikator (bisa disesuaikan per timeframe jika mau, tapi ini global)
EMA_FAST = 13
EMA_SLOW = 21
STOCH_K = 5
STOCH_D = 3

# Timestamp terakhir sinyal GLOBAL dikirim
last_global_signal_sent = None
# Jeda minimum antar sinyal global (detik). Misal, 1 jam (3600 detik)
GLOBAL_SIGNAL_COOLDOWN = 3600 # 1 hour cooldown for combined signals

# Ambang batas skor untuk mengirim sinyal (misal, 0 agar selalu terkirim)
# Ini adalah ambang batas per timeframe agar memenuhi syarat untuk sinyal global
PER_TIMEFRAME_SIGNAL_THRESHOLD = 6.0 # Contoh: minimal skor 6.0 di setiap TF untuk sinyal global

# === FUNGSI UTILITAS ===

def update_candles(data, timeframe):
    global all_candles, last_candle_processed_time
    k = data["k"]
    
    # Hanya proses candle yang sudah close
    if k["x"]:  
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
        if len(current_df) > 500: 
            current_df = current_df.iloc[-500:].reset_index(drop=True)
        all_candles[timeframe] = current_df 
        last_candle_processed_time[timeframe] = open_time # Update waktu candle terakhir yang diproses

def calculate_indicators(df):
    df_copy = df.copy()
    
    if len(df_copy) >= EMA_SLOW:
        df_copy["EMA13"] = EMAIndicator(df_copy["close"], EMA_FAST).ema_indicator()
        df_copy["EMA21"] = EMAIndicator(df_copy["close"], EMA_SLOW).ema_indicator()
    else: 
        df_copy["EMA13"] = pd.NA
        df_copy["EMA21"] = pd.NA

    if len(df_copy) >= STOCH_K + STOCH_D:
        stoch = StochasticOscillator(df_copy["high"], df_copy["low"], df_copy["close"], window=STOCH_K, smooth_window=STOCH_D)
        df_copy["stoch_k"] = stoch.stoch()
        df_copy["stoch_d"] = stoch.stoch_signal()
    else: 
        df_copy["stoch_k"] = pd.NA
        df_copy["stoch_d"] = pd.NA
        
    return df_copy


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


# === format_global_signal_output - Perubahan besar di sini ===
def format_global_signal_output(global_signal_data):
    symbol = escape_markdown_v2(global_signal_data['symbol'])
    entry_type = escape_markdown_v2(global_signal_data['entry_type'])
    
    entry_price = escape_markdown_v2(f"{global_signal_data['entry_price']:.2f}")
    target_low = escape_markdown_v2(f"{global_signal_data['target_range'][0]:.2f}")
    target_high = escape_markdown_v2(f"{global_signal_data['target_range'][1]:.2f}")
    stop_loss = escape_markdown_v2(f"{global_signal_data['stop_loss']:.2f}")

    combined_tech_conditions = []
    # Loop melalui setiap timeframe dan kondisi teknikalnya
    for timeframe, tf_data in global_signal_data['timeframe_details'].items():
        timeframe_escaped = escape_markdown_v2(timeframe)
        combined_tech_conditions.append(f"\\*\\__{timeframe_escaped} Timeframe__\\*\n")
        
        for cond, valid in tf_data["technical_conditions"].items():
            emoji = "✅" if valid else "❌"
            combined_tech_conditions.append(f"  {emoji} {escape_markdown_v2(cond)}")
        
        confidence_val = f"{tf_data['confidence']} / 10"
        combined_tech_conditions.append(f"  Confidence: `{escape_markdown_v2(confidence_val)}`\n") # Baris kosong untuk pemisah
        
    technical_str = "\n".join(combined_tech_conditions)
    
    time_str = escape_markdown_v2(global_signal_data["time"].strftime("%d %b %Y - %H:%M UTC"))

    msg = (
        f"🚀 \\*\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\*\n"
        f"🌟 \\*[GLOBAL SIGNAL DETECTED]\\* {symbol} {entry_type}\n"
        f"\\*\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\*\n\n"
        f"📆 \\*Waktu\\*: {time_str}\n\n"
        f"💡 \\*Entry Suggestion\\*:\n"
        f"📍 \\*Entry\\*: {entry_price} USDT\n"
        f"🎯 \\*Target\\*: {target_low} \\- {target_high} USDT\n"
        f"🛑 \\*Stop Loss\\*: {stop_loss} USDT\n\n"
        f"📈 \\*Kondisi Teknikal per Timeframe\\*:\n{technical_str}\n\n"
        f"\\#BTC \\#LONG \\#GlobalSignal \\#AlgoBot"
    )
    return msg

async def send_signal_to_telegram(msg: str):
    bot = Bot(token=TELEGRAM_TOKEN)
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

# === handle_websocket - Sekarang hanya untuk update data ===
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
                        # update_candles sekarang hanya dipanggil untuk candle yang sudah close
                        update_candles(data, timeframe) 

        except websockets.exceptions.ConnectionClosedOK:
            print(f"WebSocket connection for {timeframe} closed normally. Reconnecting in 5 seconds...")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"WebSocket connection for {timeframe} closed with error: {e}. Reconnecting in 5 seconds...")
        except Exception as e:
            print(f"An unexpected error occurred for {timeframe}: {e}. Reconnecting in 5 seconds...")
        
        await asyncio.sleep(5) # Jeda sebelum mencoba reconnect

# === check_global_signal_and_send - Fungsi baru untuk menganalisis semua timeframe ===
async def check_global_signal_and_send():
    global last_global_signal_sent
    required_candles_for_analysis = max(EMA_SLOW, STOCH_K + STOCH_D, 20) 

    while True:
        now = datetime.now(timezone.utc)
        
        # Cek cooldown sinyal global
        if last_global_signal_sent and (now - last_global_signal_sent).total_seconds() < GLOBAL_SIGNAL_COOLDOWN:
            # print(f"Global signal cooldown active. Next check in {(GLOBAL_SIGNAL_COOLDOWN - (now - last_global_signal_sent).total_seconds()):.0f} seconds.")
            await asyncio.sleep(60) # Cek lagi dalam 1 menit
            continue

        # Cek apakah semua timeframe sudah punya candle yang cukup dan terbaru
        all_timeframes_ready = True
        detailed_signal_info = {}
        current_price = None # Akan diambil dari 1m atau TF terkecil

        for tf in TIMEFRAMES:
            df = all_candles[tf]
            
            # Pastikan ada cukup data untuk analisis
            if df.empty or len(df) < required_candles_for_analysis:
                # print(f"Global check: Not enough data for {tf} ({len(df)}/{required_candles_for_analysis} candles).")
                all_timeframes_ready = False
                break
            
            df_analyzed = calculate_indicators(df)
            
            # Pastikan candle terakhir tidak memiliki NaN setelah perhitungan indikator
            if df_analyzed.iloc[-1].isnull().any():
                # print(f"Global check: NaN values in {tf} indicators for latest candle. Skipping this global check.")
                all_timeframes_ready = False
                break

            # Jika ini adalah timeframe tercepat (1m), ambil harga untuk entry suggestion
            if tf == "1m":
                current_price = df_analyzed.iloc[-1]["close"]

            # Hitung kondisi dan confidence untuk timeframe ini
            conditions = {
                "EMA13 Cross Above EMA21": check_bullish_cross(df_analyzed),
                "Price Above EMA13 & EMA21": check_price_above_emas(df_analyzed),
                "Stochastic Oversold Cross": check_stoch_oversold_cross(df_analyzed),
                "Bullish Divergence": detect_divergence(df_analyzed),
                "Volume > MA20": check_volume(df_analyzed)
            }
            confidence = compute_confidence(list(conditions.values()))

            # Cek apakah confidence per timeframe memenuhi syarat
            if confidence < PER_TIMEFRAME_SIGNAL_THRESHOLD:
                # print(f"Global check: {tf} confidence ({confidence}) below threshold ({PER_TIMEFRAME_SIGNAL_THRESHOLD}).")
                all_timeframes_ready = False
                break
            
            detailed_signal_info[tf] = {
                "technical_conditions": conditions,
                "confidence": confidence
            }
            
        if all_timeframes_ready and current_price is not None:
            print("Global signal conditions MET across all timeframes!")
            
            # Buat pesan sinyal global
            global_signal = {
                "symbol": "BTC/USDT",
                "entry_type": "LONG",
                "entry_price": current_price,
                "target_range": (current_price * 1.01, current_price * 1.03), # Target 1-3%
                "stop_loss": current_price * 0.99, # Stop Loss 1%
                "time": now,
                "timeframe_details": detailed_signal_info # Ini akan berisi detail per timeframe
            }
            
            message = format_global_signal_output(global_signal)
            await send_signal_to_telegram(message)
            last_global_signal_sent = now # Update timestamp sinyal global
            
        else:
            print("Global signal conditions NOT met or not enough data yet.")
            
        # Jeda sebelum cek global berikutnya. Bisa disesuaikan.
        # Misalnya, setiap 1 menit atau 5 menit.
        await asyncio.sleep(60) # Cek setiap 1 menit

# === main function ===
async def main():
    # Buat daftar tugas untuk setiap koneksi WebSocket
    websocket_tasks = [handle_websocket(tf) for tf in TIMEFRAMES]
    
    # Tambahkan tugas untuk memeriksa sinyal global
    global_signal_check_task = check_global_signal_and_send()

    # Jalankan semua tugas secara bersamaan
    print("Starting all WebSocket handlers and global signal checker...")
    await asyncio.gather(*websocket_tasks, global_signal_check_task)


if __name__ == "__main__":
    if TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN" or not isinstance(TELEGRAM_CHAT_ID, int) or TELEGRAM_CHAT_ID == 0:
        print("ERROR: Mohon isi TELEGRAM_TOKEN dan TELEGRAM_CHAT_ID di bagian KONFIGURASI.")
        print("TELEGRAM_CHAT_ID harus berupa integer non-nol (contoh: 123456789 atau -123456789).")
        print("Skrip tidak dapat berjalan tanpa konfigurasi yang benar.")
    else:
        print("Starting the multi-timeframe GLOBAL signal bot...")
        asyncio.run(main())
