"""
semantic_cache.py

Модуль семантического кэша для мгновенных ответов на повторяющиеся вопросы
(Zero-latency responses).

Использует:
- sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) для эмбеддингов
- faiss (CPU, IndexFlatIP) для поиска по косинусному сходству

ВАЖНО: для IndexFlatIP косинусное сходство корректно считается только если
все векторы (и в индексе, и запросные) L2-нормализованы. Поэтому нормализация
выполняется на каждом шаге: при добавлении и при поиске.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SemanticCache:
    """
    Семантический кэш "вопрос -> готовый ответ (текст + аудио)".

    Хранилище состоит из двух частей:
      1. FAISS-индекс (IndexFlatIP) с L2-нормализованными эмбеддингами.
      2. Маппинг vector_id -> {'query_text', 'response_text', 'audio_path'}
         (pickle или json на выбор).

    Пример:
        cache = SemanticCache(
            index_path="cache/index.faiss",
            mapping_path="cache/mapping.pkl",
        )
        cache.add("Как дела?", "У меня всё хорошо!", "audio/ok.wav")

        hit = cache.search("как у тебя дела")
        if hit:
            print(hit["response_text"], hit["audio_path"])
    """

    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    DEFAULT_THRESHOLD = 0.85
    DEFAULT_DUPLICATE_THRESHOLD = 0.995
    KEYWORD_STOPWORDS = {
        "расскажи", "покажи", "скажи", "кто", "что", "это", "про", "обо",
        "об", "как", "какой", "какая", "какие", "сейчас", "текущий",
        "текущая", "мне", "нам", "его", "ее", "её", "там", "тут",
        "такое", "такая", "такой", "такие", "тако", "оно", "она", "они",
        "включает", "включают", "входит", "входят", "состоит", "назови", "перечисли",
        "россия", "россии", "российский", "рф",
        "страна", "стране", "страны", "страну", "страной", "стран",
        "государство", "государства", "государстве", "государством", "государ",
        "наш", "наша", "наше", "нашей", "нашу", "нашим", "наших",
        "министерство", "министерства", "министерстве", "министерству",
        "министерский",
        "сельское", "сельского", "сельском", "сельским", "сельск",
        "хозяйство", "хозяйства", "хозяйстве", "хозяйством", "хозяй",
    }
    TOPIC_TOKENS = {
        "животновод", "растениевод", "птицевод", "свиновод", "молочн",
        "мясн", "крс", "скот", "ферм", "агропромышлен", "аграрн",
        "отрасл", "направлен", "апк",
    }
    REQUIRED_FALLBACK_TOKENS = TOPIC_TOKENS | {
        "министр", "минсельхоз", "оксана", "лут", "сирот", "сыроварн",
        "россельхознадзор", "росрыболов", "субсид", "кредит", "господдерж",
        "агростартап", "кооперац", "надо", "надой", "коров", "сыр",
    }

    @staticmethod
    def _normalize_match_text(text: str) -> str:
        normalized = (text or "").lower().replace("ё", "е")
        normalized = re.sub(r"[^a-zа-я0-9]+", " ", normalized, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", normalized).strip()

    def _semantic_candidate_score(self, query_text: str, entry: Dict, similarity: float) -> float:
        query_norm = self._normalize_match_text(query_text)
        title = str(entry.get("query_text") or "")
        title_norm = self._normalize_match_text(title)
        response_text = str(entry.get("response_text") or "")
        response_norm = self._normalize_match_text(response_text)
        query_tokens = self._keyword_tokens(query_text)
        title_tokens = self._keyword_tokens(title)
        response_tokens = self._keyword_tokens(response_text)

        title_overlap = query_tokens & title_tokens
        response_overlap = query_tokens & response_tokens
        score = similarity * 10.0 + len(title_overlap) * 3.0 + len(response_overlap) * 0.25

        if query_norm and title_norm:
            if query_norm == title_norm:
                score += 100.0
            elif query_norm in title_norm or title_norm in query_norm:
                score += 8.0
        if query_tokens and title_overlap == query_tokens:
            score += 8.0
        if title_norm.startswith(("кто ", "что ", "какие ", "какой ", "какая ", "сколько ", "когда ")):
            score += 1.0

        important_tokens = {"министр", "оксана", "лут", "животновод", "животноводств", "растениевод", "растениеводств", "отрасл", "агропромышлен", "комплекс", "апк"}
        for token in important_tokens & query_tokens:
            if token in title_tokens:
                score += 5.0
            elif token in response_tokens:
                score += 1.0
            else:
                score -= 6.0

        if "министр" in query_norm:
            if "занятость" in title_norm:
                score -= 20.0
            if "оксана" in title_norm or "лут" in title_norm or "действующий министр" in title_norm:
                score += 18.0
        if "сельское хозяйство" in query_norm and "мировое сельское хозяйство" in title_norm and "миров" not in query_norm:
            score -= 15.0
        if "агропромышленный комплекс" in query_norm and "интеграция" in title_norm:
            score -= 15.0
        if any(word in query_norm for word in ("что такое", "что входит", "какие отрасли", "из каких отрасл")):
            if any(word in title_norm for word in ("что такое", "что входит", "какие отрасли", "отрасли")):
                score += 5.0
        if "только справочник агронома" in response_norm or "коротит от ваших вопросов" in response_norm:
            score -= 100.0
        return score

    def __init__(
        self,
        index_path: Union[str, Path] = "semantic_cache_index.faiss",
        mapping_path: Union[str, Path] = "semantic_cache_mapping.pkl",
        model_name: str = MODEL_NAME,
        similarity_threshold: float = DEFAULT_THRESHOLD,
        duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
        device: Optional[str] = None,
        keyword_fallback_mode: str = "conservative",
    ) -> None:
        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)
        self.similarity_threshold = similarity_threshold
        self.duplicate_threshold = duplicate_threshold
        self.keyword_fallback_mode = (keyword_fallback_mode or "conservative").strip().lower()

        self._lock = threading.Lock()

        model_ref = self._resolve_model_ref(model_name)
        
        if not hasattr(SemanticCache, "_shared_model"):
            logger.info("Загрузка модели эмбеддингов: %s", model_ref)
            SemanticCache._shared_model = SentenceTransformer(model_ref, device=device)
        else:
            logger.info("Использую уже загруженную модель эмбеддингов из кэша")
            
        self.model = SemanticCache._shared_model
        
        if hasattr(self.model, "get_embedding_dimension"):
            self.dim = self.model.get_embedding_dimension()
        else:
            self.dim = self.model.get_sentence_embedding_dimension()

        self.index: faiss.Index = self._load_or_create_index()
        self.mapping: Dict[int, Dict] = self._load_or_create_mapping()

        self._next_id = (max(self.mapping.keys()) + 1) if self.mapping else 0

    @staticmethod
    def _resolve_model_ref(model_name: str) -> str:
        env_model = os.environ.get("VOICE_ENGINE_EMBEDDER_MODEL", "").strip()
        if env_model:
            return env_model
        for candidate in (
            Path(__file__).resolve().parent / "models" / "sentence_transformer",
            Path("/home/unitree/AgroBot-G1-Unified/models/sentence_transformer"),
            Path("/home/unitree/agrobot/AgroHub/models/sentence_transformer"),
        ):
            if (candidate / "modules.json").exists():
                return str(candidate)
        return model_name

    # ------------------------------------------------------------------ #
    # Загрузка / создание индекса и маппинга
    # ------------------------------------------------------------------ #

    def _load_or_create_index(self) -> faiss.Index:
        if self.index_path.exists():
            logger.info("Загружаю FAISS индекс из %s", self.index_path)
            index = faiss.read_index(str(self.index_path))
            if index.d != self.dim:
                raise ValueError(
                    f"Размерность индекса ({index.d}) не совпадает с "
                    f"размерностью модели ({self.dim}). "
                    f"Проверьте, что индекс создан той же моделью."
                )
            return index

        logger.info("Индекс не найден, создаю новый IndexFlatIP(dim=%d)", self.dim)
        flat = faiss.IndexFlatIP(self.dim)
        # Оборачиваем в IndexIDMap2, чтобы можно было явно задавать id
        # и удалять записи по id (обычный IndexFlatIP этого не умеет).
        return faiss.IndexIDMap2(flat)

    def _load_or_create_mapping(self) -> Dict[int, Dict]:
        if self.mapping_path.exists():
            logger.info("Загружаю маппинг из %s", self.mapping_path)
            suffix = self.mapping_path.suffix.lower()
            if suffix == ".json":
                with open(self.mapping_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # JSON хранит ключи как строки -> конвертируем обратно в int
                return {int(k): v for k, v in raw.items()}
            else:
                with open(self.mapping_path, "rb") as f:
                    return pickle.load(f)

        logger.info("Маппинг не найден, создаю пустой")
        return {}

    def _save(self) -> None:
        """Сохраняет индекс и маппинг на диск."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.mapping_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self.index_path))

        suffix = self.mapping_path.suffix.lower()
        if suffix == ".json":
            with open(self.mapping_path, "w", encoding="utf-8") as f:
                json.dump(
                    {str(k): v for k, v in self.mapping.items()},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        else:
            with open(self.mapping_path, "wb") as f:
                pickle.dump(self.mapping, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.debug("Кэш сохранён: %s, %s", self.index_path, self.mapping_path)

    # ------------------------------------------------------------------ #
    # Вспомогательное: эмбеддинг + L2-нормализация
    # ------------------------------------------------------------------ #

    def _embed(self, text: str) -> np.ndarray:
        """
        Считает эмбеддинг и L2-нормализует его.

        L2-нормализация ОБЯЗАТЕЛЬНА: IndexFlatIP считает просто скалярное
        произведение (inner product). Оно эквивалентно косинусному сходству
        только если ||a|| = ||b|| = 1. Без нормализации значения будут
        зависеть от длины вектора и порог 0.92 потеряет смысл.
        """
        vec = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=False,  # нормализуем сами, явно
        ).astype("float32")

        vec = vec.reshape(1, -1)
        faiss.normalize_L2(vec)  # in-place L2-нормализация (норма = 1)
        return vec

    # ------------------------------------------------------------------ #
    # Публичный API
    # ------------------------------------------------------------------ #

    def warmup(self) -> None:
        """Прогревает sentence-transformer, чтобы первый search не тормозил."""
        start = time.perf_counter()
        self._embed("привет")
        logger.info("Warmup SemanticCache завершён за %.3f сек.", time.perf_counter() - start)

    def search(self, query_text: str, language: Optional[str] = None) -> Optional[Dict]:
        """
        Ищет семантически близкий вопрос в кэше.

        Возвращает:
            {'response_text': str, 'audio_path': str, 'similarity': float,
             'matched_query': str, 'language': str}
            если найдено совпадение с cosine similarity > threshold,
            иначе None.
        """
        if self.index.ntotal == 0:
            return None

        if not query_text or not query_text.strip():
            return None

        requested_language = (language or "").strip().lower() or None
        query_norm = self._normalize_match_text(query_text)
        if query_norm:
            for exact_id, exact_entry in self.mapping.items():
                entry_language = str(exact_entry.get("language") or "").strip().lower()
                if requested_language:
                    if entry_language:
                        if entry_language != requested_language:
                            continue
                    elif requested_language not in {"ru", "en"}:
                        continue
                if query_norm == self._normalize_match_text(str(exact_entry.get("query_text") or "")):
                    logger.info(
                        "exact title hit: query=%r id=%d matched=%r",
                        query_text,
                        int(exact_id),
                        exact_entry.get("query_text"),
                    )
                    return {
                        "response_text": exact_entry["response_text"],
                        "audio_path": exact_entry["audio_path"],
                        "similarity": 1.0,
                        "matched_query": exact_entry.get("query_text"),
                        "language": entry_language or None,
                        "match_type": "exact_title",
                    }

        query_vec = self._embed(query_text)

        search_k = min(int(os.environ.get("VOICE_ENGINE_CACHE_SEARCH_K", "24")), self.index.ntotal)
        with self._lock:
            similarities, ids = self.index.search(query_vec, k=search_k)

        candidates = []
        for raw_sim, raw_id in zip(similarities[0], ids[0]):
            best_sim = float(raw_sim)
            best_id = int(raw_id)

            # faiss возвращает -1, если совпадений не найдено вообще
            if best_id == -1 or best_sim <= self.similarity_threshold:
                continue

            entry = self.mapping.get(best_id)
            if entry is None:
                # Индекс и маппинг рассинхронизировались — защищаемся от KeyError
                logger.warning(
                    "id %d есть в FAISS, но отсутствует в mapping", best_id
                )
                continue

            entry_language = str(entry.get("language") or "").strip().lower()
            if requested_language:
                if entry_language:
                    if entry_language != requested_language:
                        continue
                elif requested_language not in {"ru", "en"}:
                    continue

            rerank_score = self._semantic_candidate_score(query_text, entry, best_sim)
            candidates.append((rerank_score, best_sim, best_id, entry, entry_language))

        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            rerank_score, best_sim, best_id, entry, entry_language = candidates[0]
            logger.info(
                "semantic hit: query=%r id=%d sim=%.4f rerank=%.2f matched=%r",
                query_text,
                best_id,
                best_sim,
                rerank_score,
                entry.get("query_text"),
            )
            return {
                "response_text": entry["response_text"],
                "audio_path": entry["audio_path"],
                "similarity": best_sim,
                "matched_query": entry.get("query_text"),
                "language": entry_language or None,
            }

        if self.keyword_fallback_mode != "off":
            keyword_hit = self._keyword_fallback(query_text, requested_language)
            if keyword_hit:
                return keyword_hit

        return None

    @classmethod
    def _keyword_tokens(cls, text: str) -> set:
        words = re.findall(r"[a-zа-яё0-9]+", (text or "").lower().replace("ё", "е"))
        tokens = set()
        for word in words:
            if word in cls.KEYWORD_STOPWORDS or len(word) < 3:
                continue
            stem = cls._rough_ru_stem(word)
            if stem in cls.KEYWORD_STOPWORDS:
                continue
            tokens.add(stem)
        return tokens

    @staticmethod
    def _rough_ru_stem(word: str) -> str:
        for suffix in (
            "овичем", "евичем", "овича", "евича", "овичу", "евичу",
            "евной", "овной", "евну", "овну", "евна", "овна",
            "ством", "ствах", "ства", "ство",
            "ого", "ему", "ому", "ыми", "ими", "ами", "ями",
            "ая", "яя", "ую", "юю", "ый", "ий", "ой",
            "ом", "ем", "ым", "им", "ах", "ях",
            "а", "у", "е", "ы", "и",
        ):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                return word[: -len(suffix)]
        return word

    def _keyword_fallback(self, query_text: str, requested_language: Optional[str]) -> Optional[Dict]:
        query_tokens = self._keyword_tokens(query_text)
        if not query_tokens:
            return None

        best_entry = None
        best_score = 0
        best_overlap = set()
        for entry_id, entry in self.mapping.items():
            entry_language = str(entry.get("language") or "").strip().lower()
            if requested_language and entry_language and entry_language != requested_language:
                continue
            haystack = "{} {}".format(entry.get("query_text", ""), entry.get("response_text", ""))
            entry_tokens = self._keyword_tokens(haystack)
            overlap = query_tokens & entry_tokens
            meaningful_overlap = overlap & self.REQUIRED_FALLBACK_TOKENS
            query_is_domain_specific = bool(query_tokens & self.REQUIRED_FALLBACK_TOKENS)
            if not meaningful_overlap or not query_is_domain_specific:
                continue
            min_overlap = 1 if (
                self.keyword_fallback_mode == "topic"
                and (query_tokens & self.TOPIC_TOKENS)
                and (overlap & self.TOPIC_TOKENS)
            ) else 2
            if len(overlap) < min_overlap:
                continue
            score = len(overlap) * 100 + len(meaningful_overlap) * 150 - len(entry_tokens)
            if score > best_score:
                best_score = score
                best_entry = entry
                best_overlap = overlap

        if not best_entry:
            return None

        logger.info(
            "keyword fallback: query=%r matched=%r overlap=%s",
            query_text,
            best_entry.get("query_text"),
            sorted(best_overlap),
        )
        return {
            "response_text": best_entry["response_text"],
            "audio_path": best_entry["audio_path"],
            "similarity": 1.0,
            "matched_query": best_entry.get("query_text"),
            "language": str(best_entry.get("language") or "").strip().lower() or None,
        }

    def add(
        self,
        query_text: str,
        response_text: str,
        audio_path: str,
        language: Optional[str] = None,
        force: bool = False,
    ) -> Optional[int]:
        """
        Добавляет новую пару "вопрос -> ответ" в кэш.

        Перед добавлением проверяет, нет ли уже почти идентичного вектора
        в индексе (по умолчанию порог дубля = 0.995 — строже, чем порог
        поиска в search(), т.к. это разные семантические пороги:
        search() ищет "достаточно похожий вопрос", а дубль-проверка —
        "практически тот же самый вопрос").

        Args:
            force: если True, добавляет запись, даже если найден дубль.

        Returns:
            vector_id новой записи, либо id существующего дубля (если
            запись не была добавлена), либо None при ошибке.
        """
        if not query_text or not query_text.strip():
            raise ValueError("query_text не может быть пустым")
        if not response_text or not response_text.strip():
            raise ValueError("response_text не может быть пустым")

        vec = self._embed(query_text)

        normalized_language = (language or "").strip().lower() or None

        with self._lock:
            if not force and self.index.ntotal > 0:
                search_k = min(8, self.index.ntotal)
                similarities, ids = self.index.search(vec, k=search_k)
                for raw_sim, raw_id in zip(similarities[0], ids[0]):
                    sim = float(raw_sim)
                    existing_id = int(raw_id)
                    if existing_id == -1 or sim < self.duplicate_threshold:
                        continue
                    existing_language = str(self.mapping.get(existing_id, {}).get("language") or "").strip().lower()
                    if normalized_language and existing_language and existing_language != normalized_language:
                        continue
                    logger.info(
                        "Дубль обнаружен (id=%d, sim=%.4f >= %.4f), "
                        "добавление пропущено. Существующий вопрос: %r",
                        existing_id,
                        sim,
                        self.duplicate_threshold,
                        self.mapping.get(existing_id, {}).get("query_text"),
                    )
                    return existing_id

            new_id = self._next_id
            self._next_id += 1

            ids_array = np.array([new_id], dtype="int64")
            self.index.add_with_ids(vec, ids_array)

            self.mapping[new_id] = {
                "query_text": query_text,
                "response_text": response_text,
                "audio_path": audio_path,
                "language": normalized_language,
            }

            self._save()

        logger.info("Добавлена новая запись id=%d: %r", new_id, query_text)
        return new_id

    def put(
        self,
        query_text: str,
        response_text: str,
        audio_path: str,
        language: Optional[str] = None,
        force: bool = False,
    ) -> Optional[int]:
        """
        Совместимый алиас для add().
        """
        return self.add(query_text, response_text, audio_path, language=language, force=force)

    def remove(self, vector_id: int) -> bool:
        """Удаляет запись из кэша по id"""
        with self._lock:
            if vector_id not in self.mapping:
                return False

            selector = faiss.IDSelectorArray(np.array([vector_id], dtype="int64"))
            self.index.remove_ids(selector)
            del self.mapping[vector_id]
            self._save()

        logger.info("Запись id=%d удалена", vector_id)
        return True

    def __len__(self) -> int:
        return self.index.ntotal

    def stats(self) -> Dict:
        return {
            "total_entries": self.index.ntotal,
            "dim": self.dim,
            "index_path": str(self.index_path),
            "mapping_path": str(self.mapping_path),
            "similarity_threshold": self.similarity_threshold,
            "duplicate_threshold": self.duplicate_threshold,
        }


# ---------------------------------------------------------------------- #
# Пример использования
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    cache = SemanticCache(
        index_path="cache/semantic_index.faiss",
        mapping_path="cache/semantic_mapping.pkl",
    )

    cache.add(
        query_text="Как оформить возврат товара?",
        response_text="Чтобы оформить возврат, перейдите в раздел «Мои заказы»...",
        audio_path="audio/return_policy.wav",
    )

    # Дубль — не должен добавиться повторно
    cache.add(
        query_text="Как оформить возврат товара?",
        response_text="Другой ответ, но вопрос тот же",
        audio_path="audio/return_policy_v2.wav",
    )

    # Похожий, но не идентичный вопрос — должен найтись через search()
    result = cache.search("как вернуть товар обратно")
    if result:
        print(f"[HIT] sim={result['similarity']:.3f} -> {result['response_text']}")
        print(f"audio: {result['audio_path']}")
    else:
        print("[MISS] релевантного ответа не найдено")

    print(cache.stats())
