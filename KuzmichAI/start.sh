#!/bin/bash

# ==============================================================================
# Unitree G1 Voice Engine - Startup & Supervisor Script
# ==============================================================================

cd "$(dirname "$(readlink -f "$0")")"

# --- Настройки Цветов ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# БЛОК КОНФИГУРАЦИИ
# ==============================================================================

# 1. Основные режимы работы
export VOICE_ENGINE_ENABLE_STREAMING="false"  # true - потоковая генерация, false - ждать ответ целиком
export VOICE_ENGINE_ENABLE_PLAYBACK="true"   # true - робот озвучивает ответ, false - только пишет в консоль
export VOICE_ENGINE_AUDIO="local"
export VOICE_ENGINE_ENABLE_CONTEXT="true"  # true - робот помнит последние фразы, false - каждый вопрос как новый
export VOICE_ENGINE_MAX_HISTORY_TURNS=5    # количество связок(вопрос-ответ) которые помнит робот
export VOICE_ENGINE_ENABLE_GESTURES="true"

# 2. Настройки STT (Распознавание)
export VOICE_ENGINE_STT_BACKEND="gigastt"      # "gigastt", "whisper" или "nemo"
export VOICE_ENGINE_WHISPER_MODEL="large-v3-turbo"
export VOICE_ENGINE_NEMO_MODEL="nvidia/parakeet-tdt-0.6b-v3"      # nvidia/parakeet-tdt-0.6b-v3 или nvidia/canary-1b
export VOICE_ENGINE_STT_LANGUAGE="ru"          # gigastt специализируется на русском
export VOICE_ENGINE_GIGASTT_MODEL_DIR="${PWD}/models/gigastt"
export VOICE_ENGINE_GIGASTT_MODEL_VARIANT="rnnt"
export VOICE_ENGINE_GIGASTT_PUNCTUATION="off"
export VOICE_ENGINE_GIGASTT_ITN="auto"

# 2.5. Настройки LLM (текстовый мозг)
export VOICE_ENGINE_LLM_MODEL="/home/dayn/KuzmichAI_v2_saintofcoding/models/Qwen3.5-4B-Instruct-Q4_K_M.gguf"
export VOICE_ENGINE_LLM_NO_THINK="true"
export VOICE_ENGINE_LLM_THINKING="false"
export VOICE_ENGINE_LLM_HIDE_THINKING="true"
export VOICE_ENGINE_LLM_RAW_COMPLETION="false"
export VOICE_ENGINE_LLM_CTX="4096"
export VOICE_ENGINE_LLM_MAX_TOKENS="384"
export VOICE_ENGINE_LLM_TEMPERATURE="0.2"
export VOICE_ENGINE_TIMEZONE="Europe/Moscow"

# 3. Настройки TTS (Синтез)
export VOICE_ENGINE_TTS="silero"
export VOICE_ENGINE_TTS_FALLBACK_PIPER="1"   # 1 - использовать Piper как запасной голос
export VOICE_ENGINE_TTS_TEMPO="1.0"         # Скорость речи
export VOICE_ENGINE_TTS_GAIN_DB="0"          # Без дополнительного усиления, чтобы не было перегруза
export VOICE_ENGINE_TTS_WARMUP_LANGS="ru"    # Для Silero прогреваем русский голос
export VOICE_ENGINE_XTTS_SPLIT_SENTENCES="true"
export VOICE_ENGINE_SILERO_MODEL="v5_5_ru"
export VOICE_ENGINE_SILERO_SPEAKER="aidar"
export VOICE_ENGINE_SILERO_SAMPLE_RATE="48000"
export VOICE_ENGINE_TTS_OUTPUT_SAMPLE_RATE="24000"  # 24000 качественнее; поставь 16000, если G1 потребует старый формат
export VOICE_ENGINE_TTS_NORM_DB="-1.5"
export VOICE_ENGINE_SILERO_DEVICE="auto"
export VOICE_ENGINE_SILERO_PUT_ACCENT="true"
export VOICE_ENGINE_SILERO_PUT_YO="true"
export VOICE_ENGINE_RU_STRESS_OVERRIDES="true"  # ручные правила ударений для омографов: связи и похожие слова
export VOICE_ENGINE_RESYNTHESIZE_CACHE_AUDIO="true"
export VOICE_ENGINE_SAVE_AUDIO_CACHE="false"  # false: не хранить новые .wav в cache_audio/kb_audio
export VOICE_ENGINE_SEMANTIC_CACHE_THRESHOLD="0.94"  # выше порог: предзаписанный ответ только при очень близком вопросе
export VOICE_ENGINE_KB_CACHE_THRESHOLD="0.90"  # база знаний: не брать слабые похожие совпадения
export VOICE_ENGINE_TTS_CHUNKING="true"
export VOICE_ENGINE_TTS_MAX_CHARS="180"
export VOICE_ENGINE_TTS_MIN_CHARS="45"

# 4. Настройки Сети и G1
export VOICE_ENGINE_DDS_INTERFACE="eth0"
export VOICE_ENGINE_MIC_LOCAL_IP="192.168.123.164"
export VOICE_ENGINE_G1_VOLUME="100"
export VOICE_ENGINE_ROBOT_IP="192.168.1.103"
export VOICE_ENGINE_MOTION_API_TIMEOUT="3.0"

# 5. Системные настройки и Python
export CUDA_VISIBLE_DEVICES="0"
export JACK_NO_AUDIO_RESERVATION="1"
export AUDIODEV="null"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export TOKENIZERS_PARALLELISM="false"
export NO_PROXY="127.0.0.1,localhost,192.168.1.103,192.168.1.120,192.168.123.161,192.168.123.164"
export no_proxy="$NO_PROXY"
PYTHON_BIN="python3.11"                       # Версия Python для корректной работы TTS

# 6. Настройки шумоподавления и триггерных слов
export VOICE_ENGINE_NOISE_THRESHOLD="0.6"   # Порог RMS-фильтра (чем выше, тем сильнее давит фо>
export VOICE_ENGINE_WAKEUP_WORD="привет"     # Слово активации
export VOICE_ENGINE_STOP_WORD="стоп"         # Слово ухода в сон
export VOICE_ENGINE_ANALYSIS_WORD="анализ"   # Команда визуального анализа сцены
export VOICE_ENGINE_USE_TRIGGERS="true"      # true - реагировать на слова-триггеры, false - игн>
export VOICE_ENGINE_SERVER_URL="http://192.168.10.179:8002" # URL perception-сервера для анализа>
# Флаг перебивания: true - робот замолкает, если пользователь заговорил во время ответа;
# false - робот игнорирует попытки перебить себя и договаривает фразу до конца.
export VOICE_ENGINE_ENABLE_INTERRUPT="false"

# Сборка сложных команд из переменных
export VOICE_ENGINE_PLAYBACK_COMMAND="env -u LD_LIBRARY_PATH -u CYCLONEDDS_URI /home/unitree/g1_audio_play --iface ${VOICE_ENGINE_DDS_INTERFACE} --volume ${VOICE_ENGINE_G1_VOLUME} --file"
export VOICE_ENGINE_PLAYBACK_TIMEOUT_PAD="8.0"
export VOICE_ENGINE_PLAYBACK_TIMEOUT_MIN="10.0"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/lib/cuda/lib64:${LD_LIBRARY_PATH:-}"
# export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT="https://hf-mirror.com"

# ==============================================================================
# КОНЕЦ БЛОКА КОНФИГУРАЦИИ
# ==============================================================================

echo -e "${BLUE}====================================================${NC}"
echo -e "${BLUE}  Unitree G1 Voice Engine - Auto-Start Supervisor   ${NC}"
echo -e "${BLUE}====================================================${NC}"

# --- 1. Права sudo ---
echo -e "${GREEN}[+] Проверка прав для аппаратного ускорения Jetson...${NC}"
sudo -v || { echo -e "${RED}[-] Ошибка: не удалось получить права sudo.${NC}"; exit 1; }

# --- 2. Логирование ---
LOG_FILE="g1_voice_engine.log"
echo -e "${GREEN}[+] Настройка логирования в ${LOG_FILE}...${NC}"
exec > >(tee -a "$LOG_FILE") 2>&1

# --- 3. Graceful Shutdown ---
PYTHON_PID=""
SHUTTING_DOWN=false

cleanup() {
    echo -e "\n${YELLOW}[!] Получен сигнал SIGINT/SIGTERM. Shutting down...${NC}"
    SHUTTING_DOWN=true
    
    if [ -n "$PYTHON_PID" ]; then
        echo -e "${YELLOW}[!] Остановка процесса Python (PID: $PYTHON_PID)...${NC}"
        kill -TERM "$PYTHON_PID" 2>/dev/null
        wait "$PYTHON_PID" 2>/dev/null
    fi
    echo -e "${GREEN}[+] Движок успешно остановлен. До свидания!${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# --- 4. Оптимизация платформы ---
echo -e "${GREEN}[+] Перевод оборудования в режим MAXN...${NC}"
sudo nvpmodel -m 0 2>/dev/null || true
sudo jetson_clocks 2>/dev/null || true

# --- 5. Управление виртуальным окружением ---
echo -e "${GREEN}[+] Снятие лимитов на открытые файлы (ulimit -n 65535)...${NC}"
ulimit -n 65535

VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}[!] Виртуальное окружение не найдено. Создаю через ${PYTHON_BIN}...${NC}"
    $PYTHON_BIN -m venv "$VENV_DIR"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}[-] Ошибка при создании venv. Убедитесь, что установлен пакет ${PYTHON_BIN}-venv.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}[+] Активация виртуального окружения...${NC}"
    source "$VENV_DIR/bin/activate"
    
    echo -e "${GREEN}[+] Установка зависимостей из requirements.txt...${NC}"
    if [ -f "requirements.txt" ]; then
        pip install --upgrade pip
        pip install -r requirements.txt
    else
        echo -e "${RED}[-] Файл requirements.txt не найден! Пропускаю установку библиотек.${NC}"
    fi
else
    echo -e "${GREEN}[+] Активация существующего виртуального окружения...${NC}"
    source "$VENV_DIR/bin/activate"
fi

# --- 6. Основной цикл запуска ---
echo -e "${GREEN}[+] Запуск основного конвейера (main.py)...${NC}"

while true; do
    if [ "$SHUTTING_DOWN" = true ]; then
        break
    fi

    echo -e "${BLUE}[*] Запуск Python процесса...${NC}"
    
    "$VENV_DIR/bin/python" main.py &
    PYTHON_PID=$!
    
    wait $PYTHON_PID
    EXIT_CODE=$?

    if [ "$SHUTTING_DOWN" = true ]; then
        break
    fi

    if [ $EXIT_CODE -ne 0 ]; then
        echo -e "${RED}[-] Процесс упал с кодом ошибки: $EXIT_CODE.${NC}"
    else
        echo -e "${YELLOW}[!] Процесс завершился штатно, но ожидалась непрерывная работа.${NC}"
    fi

    echo -e "${YELLOW}[!] Авторестарт через 5 секунд... (Нажмите Ctrl+C для отмены)${NC}"
    sleep 5
done
