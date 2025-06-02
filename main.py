import asyncio
import json
import pandas as pd
from datetime import datetime, timezone
from ta.trend import EMAIndicator
from ta.momentum import StochasticOscillator
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes # <-- PERUBAHAN DI SINI
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
        # Pastikan tidak ada duplikat berdasarkan open_time
        if not current_df.empty and current_df['open_time'].iloc[-1] == open_time:
            # Update baris terakhir jika candle yang sama diperbarui (price berubah)
            current_df.iloc[-1] = new_row.iloc[0]
        else:
            # Tambah baris baru jika open_time berbeda (candle baru)
            current_df = pd.concat([current_df, new_row], ignore_index=True)
        
        if len(current_df) > 500: 
            current_df = current_df.iloc[-500:].reset_index(drop=True)
        all_candles[timeframe] = current_df 
        last_candle_processed_time[timeframe] = open_time # Update waktu candle terakhir yang diproses

# Hitung indikator teknikal di DataFrame
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


# === format_global_signal_output ===
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

# === send_message_to_telegram - Fungsi umum untuk mengirim pesan ===
# Diubah agar bisa dipanggil dari command handler juga
async def send_message_to_telegram(chat_id: int, text: str):
    bot = Bot(token=TELEGRAM_TOKEN)
    try:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

# === handle_websocket - Hanya untuk update data ===
async def handle_websocket(timeframe):
    url = BASE_BINANCE_WS_URL + timeframe
    while True: 
        try:
            async with websockets.connect(url) as ws:
                print(f"Connected to Binance WebSocket for {timeframe}...")
                async for raw in ws:
                    data = json.loads(raw)
                    if data.get('e') == 'kline' and 'k' in data:
                        update_candles(data, timeframe) 

        except websockets.exceptions.ConnectionClosedOK:
            print(f"WebSocket connection for {timeframe} closed normally. Reconnecting in 5 seconds...")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"WebSocket connection for {timeframe} closed with error: {e}. Reconnecting in 5 seconds...")
        except Exception as e:
            print(f"An unexpected error occurred for {timeframe}: {e}. Reconnecting in 5 seconds...")
        
        await asyncio.sleep(5) 

# === check_global_signal_and_send ===
async def check_global_signal_and_send():
    global last_global_signal_sent
    required_candles_for_analysis = max(EMA_SLOW, STOCH_K + STOCH_D, 20) 

    while True:
        now = datetime.now(timezone.utc)
        
        if last_global_signal_sent and (now - last_global_signal_sent).total_seconds() < GLOBAL_SIGNAL_COOLDOWN:
            await asyncio.sleep(60) 
            continue

        all_timeframes_ready = True
        detailed_signal_info = {}
        current_price = None 

        for tf in TIMEFRAMES:
            df = all_candles[tf]
            
            if df.empty or len(df) < required_candles_for_analysis:
                all_timeframes_ready = False
                break
            
            df_analyzed = calculate_indicators(df)
            
            if df_analyzed.iloc[-1].isnull().any():
                all_timeframes_ready = False
                break

            if tf == "1m":
                current_price = df_analyzed.iloc[-1]["close"]

            conditions = {
                "EMA13 Cross Above EMA21": check_bullish_cross(df_analyzed),
                "Price Above EMA13 & EMA21": check_price_above_emas(df_analyzed),
                "Stochastic Oversold Cross": check_stoch_oversold_cross(df_analyzed),
                "Bullish Divergence": detect_divergence(df_analyzed),
                "Volume > MA20": check_volume(df_analyzed)
            }
            confidence = compute_confidence(list(conditions.values()))

            if confidence < PER_TIMEFRAME_SIGNAL_THRESHOLD:
                all_timeframes_ready = False
                break
            
            detailed_signal_info[tf] = {
                "technical_conditions": conditions,
                "confidence": confidence
            }
            
        if all_timeframes_ready and current_price is not None:
            print("Global signal conditions MET across all timeframes!")
            
            global_signal = {
                "symbol": "BTC/USDT",
                "entry_type": "LONG",
                "entry_price": current_price,
                "target_range": (current_price * 1.01, current_price * 1.03), 
                "stop_loss": current_price * 0.99, 
                "time": now,
                "timeframe_details": detailed_signal_info 
            }
            
            message = format_global_signal_output(global_signal)
            await send_message_to_telegram(TELEGRAM_CHAT_ID, message) # Menggunakan send_message_to_telegram
            last_global_signal_sent = now 
            
        else:
            print("Global signal conditions NOT met or not enough data yet.")
            
        await asyncio.sleep(60) 

# === Fungsi untuk Perintah On-Demand ===

# Fungsi pembantu untuk memformat output status per timeframe
def format_timeframe_status(timeframe: str, df: pd.DataFrame, current_time: datetime):
    if df.empty or len(df) < max(EMA_SLOW, STOCH_K + STOCH_D, 20):
        return (
            f"\\*\\_\\_Status {escape_markdown_v2(timeframe.upper())} (BTC/USDT)__\\*\n"
            f"_Belum ada data yang cukup untuk analisis\._\n"
            f"_Terakhir Diperbarui:_ `{escape_markdown_v2(current_time.strftime('%d %b %Y - %H:%M:%S UTC'))}`"
        )
    
    df_analyzed = calculate_indicators(df)
    last_row = df_analyzed.iloc[-1]

    # Ambil harga real-time (candle mungkin masih open)
    current_price = last_row["close"]
    
    # Periksa kondisi
    conditions = {
        "EMA13 Cross Above EMA21": check_bullish_cross(df_analyzed),
        "Price Above EMA13 & EMA21": check_price_above_emas(df_analyzed),
        "Stochastic Oversold Cross": check_stoch_oversold_cross(df_analyzed),
        "Bullish Divergence": detect_divergence(df_analyzed),
        "Volume > MA20": check_volume(df_analyzed)
    }
    confidence = compute_confidence(list(conditions.values()))

    tech_status = []
    for cond, valid in conditions.items():
        emoji = "✅" if valid else "❌"
        tech_status.append(f"  {emoji} {escape_markdown_v2(cond)}")
    tech_str = "\n".join(tech_status)

    last_update_time = escape_markdown_v2(current_time.strftime("%d %b %Y - %H:%M:%S UTC"))
    
    msg = (
        f"\\*\\_\\_Status {escape_markdown_v2(timeframe.upper())} (BTC/USDT)__\\*\n"
        f"Current Price: `${escape_markdown_v2(f'{current_price:.2f}')}`\n"
        f"EMA13: `{escape_markdown_v2(f'{last_row.get(\'EMA13\', pd.NA):.2f}' if pd.notna(last_row.get('EMA13')) else 'N/A')}`\n"
        f"EMA21: `{escape_markdown_v2(f'{last_row.get(\'EMA21\', pd.NA):.2f}' if pd.notna(last_row.get('EMA21')) else 'N/A')}`\n"
        f"Stoch K: `{escape_markdown_v2(f'{last_row.get(\'stoch_k\', pd.NA):.2f}' if pd.notna(last_row.get('stoch_k')) else 'N/A')}`\n"
        f"Stoch D: `{escape_markdown_v2(f'{last_row.get(\'stoch_d\', pd.NA):.2f}' if pd.notna(last_row.get('stoch_d')) else 'N/A')}`\n"
        f"Volume: `{escape_markdown_v2(f'{last_row.get(\'volume\', pd.NA):.2f}' if pd.notna(last_row.get('volume')) else 'N/A')}`\n"
        f"Confidence Score: `{escape_markdown_v2(f'{confidence} / 10')}`\n\n"
        f"Conditions:\n{tech_str}\n"
        f"_Terakhir Diperbarui:_ `{last_update_time}`"
    )
    return msg


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Menanggapi perintah /status <timeframe> atau /statusbtc."""
    user_chat_id = update.effective_chat.id

    # Izinkan hanya dari TELEGRAM_CHAT_ID yang sudah dikonfigurasi
    if user_chat_id != TELEGRAM_CHAT_ID:
        await send_message_to_telegram(user_chat_id, "Maaf, Anda tidak diizinkan menggunakan bot ini.")
        return

    args = context.args
    target_timeframe = None

    if len(args) == 1 and args[0].lower() in [tf.lower() for tf in TIMEFRAMES]:
        target_timeframe = args[0].lower()
    elif len(args) == 0:
        # Jika tidak ada argumen, default ke /statusbtc yang menunjukkan semua TF
        pass
    else:
        await send_message_to_telegram(user_chat_id, 
                                     f"Perintah tidak valid\. Gunakan `/status <timeframe>` \(contoh: `/status 1m`\) atau `/statusbtc` untuk semua timeframe\.")
        return

    current_time = datetime.now(timezone.utc)
    response_messages = []

    if target_timeframe:
        # Kirim status untuk timeframe spesifik
        if target_timeframe in all_candles:
            message_content = format_timeframe_status(target_timeframe, all_candles[target_timeframe], current_time)
            response_messages.append(message_content)
        else:
            response_messages.append(f"Data untuk timeframe `{escape_markdown_v2(target_timeframe)}` tidak tersedia\.")
    else:
        # Kirim status untuk semua timeframe (seperti /statusbtc)
        for tf in TIMEFRAMES:
            if tf in all_candles:
                message_content = format_timeframe_status(tf, all_candles[tf], current_time)
                response_messages.append(message_content)
            else:
                response_messages.append(f"Data untuk timeframe `{escape_markdown_v2(tf)}` tidak tersedia\.")
    
    # Gabungkan semua pesan menjadi satu jika terlalu banyak, atau kirim terpisah jika pendek
    if len("\n\n---\n\n".join(response_messages)) < 4096: # Batas pesan Telegram
        await send_message_to_telegram(user_chat_id, "\n\n---\n\n".join(response_messages))
    else:
        for msg_part in response_messages:
            await send_message_to_telegram(user_chat_id, msg_part)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Menanggapi perintah /start."""
    user = update.effective_user
    await send_message_to_telegram(update.effective_chat.id, 
                                 f"Halo, {escape_markdown_v2(user.full_name)}! Saya adalah bot sinyal BTC/USDT Anda\. "
                                 f"Gunakan `/status <timeframe>` \(misalnya `/status 1h`\) untuk melihat status terkini, "
                                 f"atau `/statusbtc` untuk melihat semua timeframe\. "
                                 f"Saya juga akan mengirim sinyal global saat semua kondisi terpenuhi\.")


# === main function ===
async def main():
    # Inisialisasi Telegram Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Tambahkan command handler
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("statusbtc", status_command)) # Alias untuk /status tanpa argumen

    # Buat daftar tugas untuk setiap koneksi WebSocket
    websocket_tasks = [handle_websocket(tf) for tf in TIMEFRAMES]
    
    # Tambahkan tugas untuk memeriksa sinyal global
    global_signal_check_task = check_global_signal_and_send()

    # Jalankan semua tugas secara bersamaan
    print("Starting all WebSocket handlers, global signal checker, and Telegram bot polling...")
    # application.run_polling() harus dijalankan di loop yang sama atau di thread terpisah.
    # Cara terbaik adalah menggabungkannya ke dalam asyncio.gather.
    await asyncio.gather(
        *websocket_tasks,
        global_signal_check_task,
        application.run_polling() # <-- Ini akan menjalankan polling bot Telegram
    )


if __name__ == "__main__":
    if TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN" or not isinstance(TELEGRAM_CHAT_ID, int) or TELEGRAM_CHAT_ID == 0:
        print("ERROR: Mohon isi TELEGRAM_TOKEN dan TELEGRAM_CHAT_ID di bagian KONFIGURASI.")
        print("TELEGRAM_CHAT_ID harus berupa integer non-nol (contoh: 123456789 atau -123456789).")
        print("Skrip tidak dapat berjalan tanpa konfigurasi yang benar.")
    else:
        print("Starting the multi-timeframe GLOBAL signal bot with on-demand status...")
        asyncio.run(main())

