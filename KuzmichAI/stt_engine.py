"""
stt_engine.py

Модуль распознавания речи (Speech-to-Text) для голосового робота.
Поддерживает три бэкенда (выбираются через env VOICE_ENGINE_STT_BACKEND):
  1. "gigastt" (по умолчанию в start.sh) - локальный русский GigaSTT/GigaAM v3.
  2. "whisper" - faster-whisper с жестким тюнингом от галлюцинаций.
  3. "nemo" - NVIDIA NeMo (модели Parakeet/Canary) для экстремально шумных условий.

Совместим с Python 3.8+.
"""

from __future__ import annotations

import logging
import ctypes
import os
import time
import tempfile
import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# CTranslate2 Hack для локального инференса
# ---------------------------------------------------------------------------
_CT2_LIB_DIR = Path(os.environ.get("CTRANSLATE2_LIB_DIR", "/home/unitree/AgroBot/build_deps/ctranslate2_cuda_install/lib"))
_CT2_LIB = _CT2_LIB_DIR / "libctranslate2.so.4"
if _CT2_LIB.exists():
    try:
        ctypes.CDLL(str(_CT2_LIB), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass

logger = logging.getLogger(__name__)

class STTEngineError(RuntimeError):
    """Базовое исключение для ошибок STTEngine."""

# ---------------------------------------------------------------------------
# Бэкенд 1: faster-whisper (Оригинальный, но с защитой от галлюцинаций)
# ---------------------------------------------------------------------------
class WhisperBackend:
    def __init__(self, model_size: str, device: str, compute_type: str, download_root: Optional[str]):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError("Пакет 'faster-whisper' не установлен.") from exc
            
        logger.info(f"[Whisper] Загрузка модели '{model_size}' (device={device}, compute_type={compute_type})...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root,
        )

    def transcribe(self, audio_array: np.ndarray, language: Optional[str] = None) -> Tuple[str, str]:
        segments, info = self.model.transcribe(
            audio_array,
            language=language,
            beam_size=5,                       
            temperature=0.0,                   
            no_speech_threshold=0.5,           
            condition_on_previous_text=False,  
            vad_filter=False,                  
        )
        text = "".join(segment.text for segment in segments).strip()
        detected_lang = getattr(info, "language", None)
        lang_prob = getattr(info, "language_probability", 0.0)

        if detected_lang != "ru" and lang_prob < 0.6:
            logger.info("Низкая вероятность языка (%.2f). Принудительно ставлю 'ru'.", lang_prob)
            detected_lang = "ru"

        return text, detected_lang

# ---------------------------------------------------------------------------
# Бэкенд 2: NVIDIA NeMo (Parakeet / Canary)
# ---------------------------------------------------------------------------
class NeMoBackend:
    def __init__(self, model_size: str, device: str):
        try:
            import nemo.collections.asr as nemo_asr
            import soundfile as sf
            self.sf = sf
        except ImportError as exc:
            raise ImportError(
                "Пакеты NeMo не установлены. Установите: "
                "pip install nemo_toolkit['asr'] soundfile"
            ) from exc

        # main.py по умолчанию может передавать абсолютный путь к локальному whisper_small.
        # Если это так, мы игнорируем этот путь и используем Parakeet.
        if "whisper" in model_size.lower() or os.path.exists(model_size):
            nemo_model_name = os.environ.get("VOICE_ENGINE_NEMO_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
        else:
            nemo_model_name = model_size if "/" in model_size else "nvidia/parakeet-tdt-0.6b-v3"
        
        logger.info(f"[NeMo] Загрузка ASR-модели '{nemo_model_name}' (device={device})...")
        # NeMo сама умеет маппить на устройство
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=nemo_model_name, map_location=device)
        self.model.eval()

        # Для Canary-моделей нужно явно указывать задачу
        if "canary" in nemo_model_name.lower():
            self.is_canary = True
            # Настройка под транскрипцию (без перевода)
            self.model.task = "asr"
            self.model.source_lang = "ru" # Можно менять динамически в transcribe
            self.model.dest_lang = "ru"
        else:
            self.is_canary = False

    def transcribe(self, audio_array: np.ndarray, language: Optional[str] = None) -> Tuple[str, str]:
        # NVIDIA NeMo ASRModel.transcribe() ожидает список путей к аудиофайлам.
        # Поскольку у нас в памяти float32-массив, мы сбрасываем его во временный WAV.
        # На Linux tempfile использует tmpfs (ОЗУ), поэтому диск не трогается, IO-задержка равна ~0.
        
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            self.sf.write(tmp_path, audio_array, 16000)
            
            # Если это Canary и мы знаем язык запроса
            if self.is_canary and language:
                # В NeMo языковые коды отличаются от Whisper, но базовые ru/en совпадают
                self.model.source_lang = language
                self.model.dest_lang = language

            # В разных версиях NeMo параметр называется по-разному.
            # Пробуем все возможные варианты, чтобы избежать TypeError:
            try:
                texts = self.model.transcribe(audio=[tmp_path], batch_size=1)
            except TypeError:
                try:
                    texts = self.model.transcribe(paths2audio_files=[tmp_path], batch_size=1)
                except TypeError:
                    try:
                        texts = self.model.transcribe(audio_paths=[tmp_path], batch_size=1)
                    except TypeError:
                        # Самый жесткий фоллбек: передаем как позиционный аргумент
                        texts = self.model.transcribe([tmp_path])
            
            if not texts or not texts[0]:
                recognized_text = ""
            else:
                res = texts[0]
                if isinstance(res, list):
                    res = res[0]
                
                if hasattr(res, 'text'):
                    recognized_text = res.text
                elif isinstance(res, dict):
                    recognized_text = res.get('text', '')
                elif isinstance(res, str):
                    recognized_text = res
                else:
                    recognized_text = str(res)
                    
        finally:
            # Обязательно подчищаем tmpfs
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        recognized_text = recognized_text.strip()
        
        # NVIDIA NeMo обычно не возвращает вероятность языка как Whisper,
        # поэтому мы отдаем тот язык, который ожидался (или None для фоллбека).
        return recognized_text, language



# ---------------------------------------------------------------------------
# Бэкенд 3: GigaSTT (GigaAM v3, русский on-device STT)
# ---------------------------------------------------------------------------
class GigaSTTBackend:
    def __init__(self):
        try:
            import gigastt_uniffi as gigastt
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Пакет 'gigastt' не установлен в этом venv. "
                "Установите: ./venv/bin/python -m pip install gigastt"
            ) from exc
        except Exception as exc:
            raise ImportError(f"Не удалось импортировать gigastt: {exc}") from exc

        model_dir = os.environ.get(
            "VOICE_ENGINE_GIGASTT_MODEL_DIR",
            str(Path(__file__).resolve().parent / "models" / "gigastt"),
        )
        self.model_dir = Path(os.path.expanduser(model_dir)).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)

        variant = os.environ.get("VOICE_ENGINE_GIGASTT_MODEL_VARIANT", "rnnt").strip()
        if variant:
            os.environ.setdefault("GIGASTT_MODEL_VARIANT", variant)
        os.environ.setdefault(
            "GIGASTT_PUNCTUATION",
            os.environ.get("VOICE_ENGINE_GIGASTT_PUNCTUATION", "off"),
        )
        os.environ.setdefault("GIGASTT_ITN", os.environ.get("VOICE_ENGINE_GIGASTT_ITN", "auto"))

        logger.info("[GigaSTT] Загрузка модели из %s (variant=%s)...", self.model_dir, variant or "auto")
        try:
            self.engine = gigastt.Engine(str(self.model_dir))
        except Exception as exc:
            raise STTEngineError(
                "GigaSTT не смог загрузить модель. Если модель еще не скачана, выполните:\n"
                f"mkdir -p \"{self.model_dir}\" && cd \"{self.model_dir}\" && "
                "wget -c https://github.com/ekhodzitsky/gigastt/releases/download/models-v3-2026-06-22/v3_rnnt_encoder_int8.onnx && "
                "wget -c https://github.com/ekhodzitsky/gigastt/releases/download/models-v3-2026-06-22/v3_rnnt_decoder.onnx && "
                "wget -c https://github.com/ekhodzitsky/gigastt/releases/download/models-v3-2026-06-22/v3_rnnt_joint.onnx && "
                "wget -c https://github.com/ekhodzitsky/gigastt/releases/download/models-v3-2026-06-22/v3_vocab.txt"
            ) from exc

    def _write_temp_wav(self, audio_array: np.ndarray) -> str:
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        samples = np.clip(audio_array, -1.0, 1.0)
        pcm16 = (samples * 32767.0).astype(np.int16)
        with wave.open(tmp_path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(pcm16.tobytes())
        return tmp_path

    def transcribe(self, audio_array: np.ndarray, language: Optional[str] = None) -> Tuple[str, str]:
        tmp_path = self._write_temp_wav(audio_array)
        try:
            result = self.engine.transcribe_file(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if hasattr(result, "text"):
            text = result.text
        elif isinstance(result, dict):
            text = result.get("text", "")
        else:
            text = str(result)

        return text.strip(), "ru"


# ---------------------------------------------------------------------------
# Главный фасад (Диспетчер)
# ---------------------------------------------------------------------------
class STTEngine:
    """
    Обертка над STT. Загружает нужный движок на основе VOICE_ENGINE_STT_BACKEND.
    """

    _WHISPER_TO_XTTS_LANG_MAP = {
        "en": "en", "ru": "ru", "es": "es", "fr": "fr", "de": "de",
        "zh": "zh-cn", "ar": "ar", "pt": "pt", "it": "it", "pl": "pl",
        "tr": "tr", "nl": "nl", "cs": "cs", "ja": "ja", "ko": "ko", "hu": "hu",
    }
    _FALLBACK_LANG = "ru"

    @classmethod
    def supported_tts_languages(cls) -> Tuple[str, ...]:
        backend = os.environ.get("VOICE_ENGINE_STT_BACKEND", "whisper").strip().lower()
        if backend == "gigastt":
            return ("ru",)
        return tuple(sorted(set(cls._WHISPER_TO_XTTS_LANG_MAP.values())))

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Optional[str] = None,
    ) -> None:
        self.device = device
        self._is_warmed_up = False
        
        # Определяем, какой движок использовать
        self.backend_type = os.environ.get("VOICE_ENGINE_STT_BACKEND", "whisper").strip().lower()
        
        load_start = time.perf_counter()
        if self.backend_type == "gigastt":
            self.backend = GigaSTTBackend()
        elif self.backend_type == "nemo":
            self.backend = NeMoBackend(model_size=model_size, device=device)
        else:
            self.backend_type = "whisper"
            self.backend = WhisperBackend(
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                download_root=download_root
            )

        load_time = time.perf_counter() - load_start
        logger.info("Движок '%s' загружен за %.2f сек.", self.backend_type, load_time)

    def warmup(self, sample_rate: int = 16000, duration_sec: float = 1.0) -> None:
        """Прогревает модель массивом тишины для инициализации CUDA-графов."""
        logger.debug("Запуск warmup() STTEngine...")
        silence = np.zeros(int(sample_rate * duration_sec), dtype=np.float32)

        warmup_start = time.perf_counter()
        try:
            self.backend.transcribe(silence)
        except Exception as exc:
            logger.exception("Ошибка во время warmup().")
            raise STTEngineError(f"Ошибка прогрева модели STT: {exc}") from exc

        warmup_time = time.perf_counter() - warmup_start
        self._is_warmed_up = True
        logger.info("Warmup STTEngine завершён за %.3f сек.", warmup_time)

    def _map_language_to_xtts(self, detected_lang: Optional[str]) -> str:
        if not detected_lang:
            return self._FALLBACK_LANG
        xtts_lang = self._WHISPER_TO_XTTS_LANG_MAP.get(detected_lang.lower())
        return xtts_lang if xtts_lang else self._FALLBACK_LANG

    def transcribe(
        self,
        audio_array: np.ndarray,
        language: Optional[str] = None,
        beam_size: int = 5, # Оставлен для совместимости сигнатуры, игнорируется внутри для анти-галлюцинаций
    ) -> Tuple[str, str]:
        
        if not self._is_warmed_up:
            logger.warning("transcribe() вызван до warmup().")

        if audio_array is None or audio_array.size == 0:
            return "", self._FALLBACK_LANG

        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        start_time = time.perf_counter()
        try:
            recognized_text, detected_lang = self.backend.transcribe(audio_array, language=language)
        except Exception as exc:
            logger.exception("Ошибка во время transcribe().")
            raise STTEngineError(f"Ошибка транскрибации: {exc}") from exc

        latency = time.perf_counter() - start_time
        xtts_lang_code = self._map_language_to_xtts(detected_lang)

        logger.debug(
            "STT latency=%.3f сек | backend=%s | audio_len=%.2f сек | lang=%s -> xtts_lang=%s | text_len=%d",
            latency,
            self.backend_type,
            audio_array.shape[0] / 16000.0,
            detected_lang,
            xtts_lang_code,
            len(recognized_text),
        )

        return recognized_text, xtts_lang_code


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Для теста GigaSTT установи: export VOICE_ENGINE_STT_BACKEND=gigastt
    engine = STTEngine()
    engine.warmup()
    dummy_audio = np.zeros(16000 * 2, dtype=np.float32)
    text, lang = engine.transcribe(dummy_audio)
    logger.info("Результат: text=%r, lang=%s", text, lang)
