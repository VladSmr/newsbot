"""
LLM-фильтр новостей.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from ..collector import NewsItem
from .client import DeepSeekClient, LLMError, BalanceInfo
from .prompts import (
    SYSTEM_PROMPT, BATCH_SYSTEM_PROMPT,
    build_feedback_block, build_batch_user_message, sanitize_item_field,
    DEDUP_SYSTEM_PROMPT, build_dedup_user_message,
)

logger = logging.getLogger("newsbot.llm.filter")


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------


@dataclass
class FilterResult:
    """
    Результат оценки одной новости LLM-фильтром.

    Attributes
    ----------
    ok : bool
        True — фильтр отработал валидно (score валиден, можно принимать решение).
        False — ошибка LLM или невалидный ответ → fallback: слать непрофильтрованным.
    score : Optional[int]
        Оценка 0–10. None если ok=False.
    summary : str
        Выжимка фактов. Пустая если ok=False.
    reason : str
        Краткое обоснование. Пустая если ok=False.
    urgent : bool
        Флаг срочности. True — событие мирового/национального масштаба
        (война, мобилизация, катастрофа с массовыми жертвами, ядерный инцидент).
        По умолчанию False (безопасный дефолт).
    """

    ok: bool
    score: Optional[int] = None
    summary: str = ""
    reason: str = ""
    urgent: bool = False


# ----------------------------------------------------------------------
# Public helpers
# ----------------------------------------------------------------------


def route(
    score: int,
    push_threshold: int,
    digest_threshold: int,
    urgent: bool = False,
    push_urgent_only: bool = False,
) -> str:
    """
    Решает, куда отправить новость по числовой оценке и флагу срочности.

    Parameters
    ----------
    score : int
        Оценка LLM, 0–10.
    push_threshold : int
        Порог немедленного пуша.
    digest_threshold : int
        Порог дайджеста (push_threshold > score >= digest_threshold → digest;
        score < digest_threshold → skip).
    urgent : bool
        Флаг срочности от LLM. При сомнении LLM → false (безопасный дефолт).
    push_urgent_only : bool
        Если True — пуш только при urgent=True AND score>=push_threshold.
        Если False — прежнее поведение: score>=push_threshold → push.

    Returns
    -------
    str
        "push" | "digest" | "skip"

    Логика при push_urgent_only=True:
        urgent AND score>=push_threshold → "push"
        score>=digest_threshold           → "digest"
        иначе                            → "skip"

    Логика при push_urgent_only=False (обратная совместимость):
        score>=push_threshold            → "push"
        score>=digest_threshold          → "digest"
        иначе                            → "skip"
    """
    if push_urgent_only:
        # пуш ТОЛЬКО если urgent + достаточно высокий score (вторичная защита
        # от перестарания модели: даже срочное с score<push_threshold → digest)
        if urgent and score >= push_threshold:
            return "push"
        elif score >= digest_threshold:
            return "digest"
        else:
            return "skip"
    else:
        # прежнее поведение: score — единственный гейт
        if score >= push_threshold:
            return "push"
        elif score >= digest_threshold:
            return "digest"
        else:
            return "skip"


# ----------------------------------------------------------------------
# LLMFilter
# ----------------------------------------------------------------------


class LLMFilter:
    """
    Оценивает NewsItem через LLM (DeepSeek) и возвращает FilterResult.

    Parameters
    ----------
    config_llm : dict
        Секция config["llm"] — provider, model, api_key, base_url,
        temperature, timeout_sec, max_retries, push_threshold, digest_threshold,
        system_prompt.
    client : DeepSeekClient, optional
        Клиент. Если None — создаётся из config_llm.
    """

    def __init__(
        self,
        config_llm: Dict[str, Any],
        client: Optional[DeepSeekClient] = None,
    ):
        self.enabled = self._parse_bool(config_llm.get("enabled", True))
        self.model = config_llm.get("model", "")
        self.push_threshold = int(config_llm.get("push_threshold", 7))
        self.digest_threshold = int(config_llm.get("digest_threshold", 5))

        # M1-фикс: при отсутствии ключа push_urgent_only — явный warning.
        # Дефолт True (urgent-only). Без warning'а legacy-конфиг без ключа
        # молча переключался бы в urgent-only (filter=True, route def=False).
        if "push_urgent_only" not in config_llm:
            logger.warning(
                "llm.push_urgent_only отсутствует в конфиге — "
                "используется дефолт True (urgent-only). "
                "Явно задайте push_urgent_only: true/false в config.yaml."
            )
        self.push_urgent_only = self._parse_bool(config_llm.get("push_urgent_only", True))
        self.system_prompt = config_llm.get("system_prompt") or SYSTEM_PROMPT

        if client is not None:
            self._client = client
        else:
            self._client = DeepSeekClient(
                api_key=config_llm.get("api_key", ""),
                base_url=config_llm.get("base_url", "https://api.deepseek.com"),
                model=config_llm.get("model", ""),
                timeout=float(config_llm.get("timeout_sec", 60)),
                max_retries=int(config_llm.get("max_retries", 3)),
                temperature=float(config_llm.get("temperature", 0.0)),
                max_tokens=int(config_llm.get("max_tokens", 4096)),
            )

    @staticmethod
    def _parse_bool(value) -> bool:
        """
        Явный парсер bool-значений из конфига.
        - bool: как есть
        - str: "true","1","yes","on" (case-insensitive) → True; всё остальное → False
        - прочее: False
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return False

    def evaluate(
        self,
        item: NewsItem,
        feedback_context: Optional[tuple] = None,
    ) -> FilterResult:
        """
        Оценивает одну новость.

        Parameters
        ----------
        item : NewsItem
            Новость для оценки.
        feedback_context : tuple, optional
            Кортеж (dislikes, likes), где каждый элемент — список dict
            из get_recent_feedback(). Если передан — блок обратной связи
            подставляется в user-сообщение перед новостью.
            При ошибке получения фидбэка — best-effort, вызывающий код
            уже отловил исключение и передал пустые списки.

        При ошибке LLM — возвращает FilterResult(ok=False), что означает
        «fallback: слать непрофильтрованным». Исключение не бросается.
        """
        if not self.enabled:
            return FilterResult(ok=True, score=10, summary=item.summary, reason="llm disabled")

        user_message = self._build_user_message(item, feedback_context=feedback_context)

        try:
            raw = self._client.chat_json(
                system=self.system_prompt,
                user=user_message,
            )
            return self._parse_response(raw)

        except LLMError as e:
            logger.warning(
                "LLM-ошибка при оценке '%s': %s — fallback: шлём непрофильтрованным",
                item.title[:60],
                e,
            )
            return FilterResult(ok=False)

    def evaluate_batch(
        self,
        items: List["NewsItem"],
        feedback_context: Optional[tuple] = None,
    ) -> List[FilterResult]:
        """
        Оценивает батч новостей одним LLM-вызовом.

        Все items нумеруются индексами 1..N в user-сообщении; LLM
        возвращает JSON ``{"results": [{"id": 1, ...}, ...]}``.
        Маппинг по ``id`` → FilterResult для каждого входного item.
        Per-item fallback: невалидный/пропущенный/битый элемент → ``ok=False``
        для этого item (остальные обрабатываются normally).

        Полный сбой LLM (``LLMError``) → все items ``ok=False``.

        При ``llm.batch_size=1`` ведёт себя идентично ``evaluate()`` по одному item.

        Parameters
        ----------
        items : List[NewsItem]
            Список новостей для оценки (порядок сохраняется в выходном списке).
        feedback_context : tuple, optional
            (dislikes, likes) — списки dict из ``get_recent_feedback()``.
            Блок обратной связи вставляется один раз на батч.

        Returns
        -------
        List[FilterResult]
            ``len(results) == len(items)``; результаты в том же порядке,
            что и входные items. Каждый результат соответствует item с
            тем же индексом 1-based в ответе LLM.
        """
        if not items:
            return []

        if not self.enabled:
            return [
                FilterResult(
                    ok=True,
                    score=10,
                    summary=item.summary,
                    reason="llm disabled",
                )
                for item in items
            ]

        if len(items) == 1:
            result = self.evaluate(items[0], feedback_context=feedback_context)
            return [result]

        # собираем user-сообщение: few-shot один раз на батч + нумерация items
        user_message = build_batch_user_message(items, feedback_context=feedback_context)

        try:
            raw = self._client.chat_json(
                system=BATCH_SYSTEM_PROMPT,
                user=user_message,
            )
            return self._parse_batch_response(raw, len(items))

        except LLMError as e:
            logger.warning(
                "LLM-ошибка при батч-оценке %d items: %s — "
                "fallback для всех: шлём непрофильтрованными",
                len(items),
                e,
            )
            return [FilterResult(ok=False) for _ in items]

    def _parse_batch_response(self, raw: Dict[str, Any], expected_count: int) -> List[FilterResult]:
        """
        Парсит батч-ответ LLM ``{"results": [...]}`` → List[FilterResult].

        Per-item валидация (как ``_parse_response``):
          - ``id`` — целое 1..N, если пропущен/битый → этот item ok=False.
          - ``score`` — int 0..10 (tolerant: целочисленный float, str-"int").
          - ``summary`` — непустой.
          - ``urgent`` — bool (tolerant: "true"/"false"/1/0).
          - Все поля валидны → ``FilterResult(ok=True, ...)``.

        При невалидном ``score``/``summary`` → ``ok=False`` для ЭТОГО item.
        Полный сбой (нет results, не dict) → все items ok=False.

        Parameters
        ----------
        raw : dict
            Распарсенный JSON от LLM.
        expected_count : int
            Количество входных items (длина батча).

        Returns
        -------
        List[FilterResult]
            Длина == expected_count. Результаты соответствуют входным items
            по индексу (id=1 → index 0, id=2 → index 1, ...).
        """
        # полный сбой структуры ответа
        if not isinstance(raw, dict):
            logger.warning(
                "LLM вернул не-объект в батч-ответе: %s → fallback для всех",
                type(raw).__name__,
            )
            return [FilterResult(ok=False) for _ in range(expected_count)]

        raw_results: Any = raw.get("results")
        if not isinstance(raw_results, list):
            logger.warning(
                "LLM вернул results не-списком: %s → fallback для всех",
                type(raw_results).__name__,
            )
            return [FilterResult(ok=False) for _ in range(expected_count)]

        # стартовый массив: все — ok=False (fallback по умолчанию)
        results: List[FilterResult] = [FilterResult(ok=False) for _ in range(expected_count)]

        # множество уже обработанных id (для детекции дубликатов)
        seen_ids: Set[int] = set()

        for item_raw in raw_results:
            if not isinstance(item_raw, dict):
                # битый элемент массива — пропускаем, ok=False уже стоит
                continue

            raw_id = item_raw.get("id")
            # парсим id: принимаем int; tolerant — целочисленный float/str
            parsed_id: Optional[int] = None
            if isinstance(raw_id, int) and not isinstance(raw_id, bool):
                if 1 <= raw_id <= expected_count:
                    parsed_id = raw_id
            elif isinstance(raw_id, float):
                import math
                if math.isfinite(raw_id) and float(int(raw_id)) == raw_id:
                    candidate = int(raw_id)
                    if 1 <= candidate <= expected_count:
                        parsed_id = candidate
            elif isinstance(raw_id, str):
                try:
                    candidate = int(raw_id.strip())
                    if 1 <= candidate <= expected_count:
                        parsed_id = candidate
                except (ValueError, TypeError):
                    pass

            if parsed_id is None:
                logger.warning(
                    "LLM вернул невалидный или out-of-range id %r → пропускаем",
                    raw_id,
                )
                continue  # ok=False уже стоит

            if parsed_id in seen_ids:
                logger.warning("LLM вернул дубликат id=%d → пропускаем", parsed_id)
                continue  # уже обработан (первый выигрывает), ok=False

            seen_ids.add(parsed_id)
            index = parsed_id - 1  # 1-based id → 0-based index

            # ── score (per-item валидация, tolerant) ─────────────────
            raw_score = item_raw.get("score")
            score: Optional[int] = None

            if isinstance(raw_score, bool):
                pass
            elif isinstance(raw_score, int) and not isinstance(raw_score, bool):
                score = raw_score
            elif isinstance(raw_score, float):
                import math
                if (math.isfinite(raw_score)
                        and float(int(raw_score)) == raw_score
                        and 0 <= raw_score <= 10):
                    score = int(raw_score)
            elif isinstance(raw_score, str):
                try:
                    candidate = int(raw_score.strip())
                    if 0 <= candidate <= 10:
                        score = candidate
                except (ValueError, TypeError):
                    pass

            if score is None or not (0 <= score <= 10):
                logger.warning(
                    "Батч item id=%d: score %r невалиден (не int 0..10) → ok=False",
                    parsed_id,
                    raw_score,
                )
                results[index] = FilterResult(ok=False)
                continue

            # ── summary (per-item валидация) ─────────────────────────
            summary = str(item_raw.get("summary", "")).strip()
            if not summary:
                logger.warning(
                    "Батч item id=%d: пустой summary → ok=False",
                    parsed_id,
                )
                results[index] = FilterResult(ok=False)
                continue

            reason = str(item_raw.get("reason", "")).strip()

            # ── urgent (per-item валидация, tolerant) ─────────────────
            raw_urgent = item_raw.get("urgent")
            urgent: bool = False
            if isinstance(raw_urgent, bool):
                urgent = raw_urgent
            elif isinstance(raw_urgent, int) and not isinstance(raw_urgent, bool):
                urgent = bool(raw_urgent)
            elif isinstance(raw_urgent, str):
                s = raw_urgent.strip().lower()
                if s in ("true", "1", "yes", "on"):
                    urgent = True
                elif s in ("false", "0", "no", "off"):
                    urgent = False
                # остальное → False

            results[index] = FilterResult(
                ok=True,
                score=score,
                summary=summary,
                reason=reason,
                urgent=urgent,
            )

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_user_message(
        self,
        item: NewsItem,
        feedback_context: Optional[tuple] = None,
    ) -> str:
        """
        Формирует user-сообщение для LLM из полей NewsItem.

        Parameters
        ----------
        item : NewsItem
        feedback_context : tuple, optional
            (dislikes, likes) — списки dict из get_recent_feedback().
            Если не None и не пуст — перед новостью вставляется блок
            обратной связи, ЯВНО промаркированный как данные.
        """
        parts: List[str] = []

        # Блок обратной связи — перед новостью (few-shot, Фаза 2c)
        if feedback_context is not None:
            dislikes, likes = feedback_context
            feedback_block = build_feedback_block(
                dislikes=dislikes,
                likes=likes,
                max_chars_per_example=200,
            )
            if feedback_block:
                parts.append(feedback_block)
                parts.append("")  # пустая строка как разделитель

        parts.append(f"# Заголовок\n{sanitize_item_field(item.title)}")

        if item.summary:
            parts.append(f"# Содержание\n{sanitize_item_field(item.summary)}")

        if item.source:
            parts.append(f"# Источник: {sanitize_item_field(item.source)}")

        return "\n\n".join(parts)

    # ── balance ─────────────────────────────────────────────────────────────

    def get_balance(self) -> BalanceInfo:
        """Делегирует клиенту. Возвращает BalanceInfo; исключений не бросает."""
        return self._client.get_balance()

    @property
    def client(self) -> DeepSeekClient:
        """Доступ к HTTP-клиенту для pipeline / close."""
        return self._client

    # ── internal ────────────────────────────────────────────────────────────

    def _parse_response(self, raw: Dict[str, Any]) -> FilterResult:
        """
        Валидирует JSON-ответ LLM и возвращает FilterResult.

        T2-9 (трейд-офф):
          Принимаем: int; float если целочисленный (7.0); str если парсится в целое ("7").
          Отклоняем: bool (подкласс int — проверяем ДО int); NaN/inf/значения вне 0..10;
          нечисловые строки.
          Комментарий: модель иногда возвращает 7.0 (float) или "7" (str) —
          это рабочие значения, не мусор. Принимаем их вместо fallback'а.
        """
        raw_score = raw.get("score")
        score: Optional[int] = None

        # ── T2-9: порядок ВАЖЕН — bool перед int (bool подкласс в Python) ──
        if isinstance(raw_score, bool):
            # bool отклоняем (True→1, False→0 — это не число, а флаг)
            pass
        elif isinstance(raw_score, int) and not isinstance(raw_score, bool):
            # int (но не bool, т.к. bool уже отсечён)
            score = raw_score
        elif isinstance(raw_score, float):
            # float: принимаем если целочисленный и в диапазоне 0..10
            import math
            if (math.isfinite(raw_score)
                    and float(int(raw_score)) == raw_score
                    and 0 <= raw_score <= 10):
                score = int(raw_score)
            # NaN/inf/дробные — не принимаем
        elif isinstance(raw_score, str):
            s = raw_score.strip()
            try:
                # принимаем str если парсится в целое в диапазоне
                candidate = int(s)
                if 0 <= candidate <= 10:
                    score = candidate
            except (ValueError, TypeError):
                pass  # нечисловая строка → не принимаем

        if score is None or not (0 <= score <= 10):
            logger.warning(
                "LLM вернул score %r (не int 0..10) → fallback: шлём непрофильтрованным",
                raw_score,
            )
            return FilterResult(ok=False)

        summary = str(raw.get("summary", "")).strip()
        if not summary:
            logger.warning(
                "LLM вернул пустой summary → fallback: шлём непрофильтрованным"
            )
            return FilterResult(ok=False)

        reason = str(raw.get("reason", "")).strip()

        # ── T2-11: парсинг флага urgent ───────────────────────────────
        # Принимаем: bool, "true"/"false" (case-insensitive), 1/0 (int).
        # При отсутствии или невалидном значении → False (безопасный дефолт).
        raw_urgent = raw.get("urgent")
        urgent: bool = False
        if isinstance(raw_urgent, bool):
            urgent = raw_urgent
        elif isinstance(raw_urgent, int) and not isinstance(raw_urgent, bool):
            urgent = bool(raw_urgent)
        elif isinstance(raw_urgent, str):
            s = raw_urgent.strip().lower()
            if s in ("true", "1", "yes", "on"):
                urgent = True
            elif s in ("false", "0", "no", "off"):
                urgent = False
            # всё остальное (включая пустую строку, None и т.п.) → False

        return FilterResult(
            ok=True,
            score=score,
            summary=summary,
            reason=reason,
            urgent=urgent,
        )

    # ── detect_duplicates ─────────────────────────────────────────────────────────

    # Порог чанкования: не более 40 кандидатов на один LLM-вызов
    _DEDUP_CHUNK_SIZE = 40

    def detect_duplicates(
        self,
        items: List["NewsItem"],
        recent: List[dict],
    ) -> Set[int]:
        """
        Группирует семантические дубликаты среди кандидатов и relative to recent.

        Отдельный лёгкий LLM-вызов (только заголовки) ДО скоринга.
        Критерий: «одно и то же событие, разные слова/язык/числа».
        Примеры: RU-пара (дроны/БПЛА), EN-пара (Pakistan school).

        Полный сбой (``LLMError``) → пустое множество (безопасно, ничего не пропущено).
        Битые/несуществующие индексы в ответе → игнорируются.
        Если кандидатов < 2 → вызов не делается, возвращается пустое множество.

        Parameters
        ----------
        items : List[NewsItem]
            Кандидаты этого цикла (новые items, прошедшие лексический дедуп).
        recent : List[dict]
            Недавние новости из БД ``(title, source)`` — для кросс-циклового дедупа.
            Каждый dict с ключами ``title`` и ``source``.

        Returns
        -------
        Set[int]
            0-based индексы items, которые являются дубликатами
            БОЛЕЕ РАННЕЙ новости (из основного списка или из recent).
            Пустое множество — ошибка или кандидатов < 2.
        """
        if len(items) < 2:
            return set()

        all_dup_indices: Set[int] = set()

        # Чанкование: не более _DEDUP_CHUNK_SIZE кандидатов на вызов
        for chunk_start in range(0, len(items), self._DEDUP_CHUNK_SIZE):
            chunk = items[chunk_start: chunk_start + self._DEDUP_CHUNK_SIZE]

            user_message = build_dedup_user_message(chunk, recent)

            try:
                raw = self._client.chat_json(
                    system=DEDUP_SYSTEM_PROMPT,
                    user=user_message,
                )
            except LLMError as e:
                logger.warning(
                    "LLM-ошибка при дедупликации чанка [%d:%d] из %d items: %s "
                    "→ прерываем, сохраняем уже найденные дубли (%d шт.)",
                    chunk_start,
                    chunk_start + len(chunk),
                    len(items),
                    e,
                    len(all_dup_indices),
                )
                # m1-фикс: НЕ сбрасываем all_dup_indices — сохраняем находки
                # предыдущих чанков. Токены уже списаны, информация не потеряна.
                break

            # Парсим ответ: {"duplicates": [1, 3, ...]}
            dup_indices = self._parse_dedup_response(
                raw, len(chunk), chunk_start,
            )
            all_dup_indices.update(dup_indices)

        logger.debug(
            "LLM-дедуп: %d/%d кандидатов помечены дубликатами",
            len(all_dup_indices), len(items),
        )
        return all_dup_indices

    def _parse_dedup_response(
        self,
        raw: Dict[str, Any],
        chunk_len: int,
        chunk_offset: int,
    ) -> Set[int]:
        """
        Парсит ответ LLM ``{"duplicates": [1, 2, ...]}`` → Set[int] 0-based.

        Битые/несуществующие/вне-диапазона индексы — игнорируются.
        Невалидная структура → пустое множество.

        Minor #4-фикс — защита от «самоссылки» / лидера чанка:
          (a) Guard лидера: первый item чанка (chunk_offset+1 в 1-based) НЕ может
              быть помечен как дубль — у него нет «более раннего» в чанке.
              LLM может вернуть индекс первого item (ошибка модели или спам),
              иначе первый item чанка никогда не выжил бы для других как «ранний».
          (b) Guard пустого чанка: если модель пометила ВСЕ items чанка как дубли
              (например, весь чанк = дубликаты recent-новостей) — оставляем хотя бы
              первый item чанка (он станет «лидером» в next-цикле). Иначе все items
              получат skipped/deferred и информация о событии потеряется.

        Риск: осознанный компромисс. Если событие полностью описано одним
        item из чанка и он помечен как дубль — событие потеряется для пуша.
        Это приемлемо: дедуп рассчитан на пары (A/B), а не на полное
        совпадение всего чанка. Защита (b) гарантирует: минимум 1 item выживает.

        Parameters
        ----------
        raw : dict
            Распарсенный JSON от LLM.
        chunk_len : int
            Количество items в текущем чанке.
        chunk_offset : int
            Стартовый offset оригинального списка (chunk_start).

        Returns
        -------
        Set[int]
            0-based индексы дублей В ПРЕДЕЛАХ ОРИГИНАЛЬНОГО СПИСКА.
        """
        if not isinstance(raw, dict):
            logger.warning("LLM-дедуп вернул не-объект: %s", type(raw).__name__)
            return set()

        raw_list = raw.get("duplicates")
        if not isinstance(raw_list, list):
            logger.warning(
                "LLM-дедуп: 'duplicates' не список (%s) → пропускаем",
                type(raw_list).__name__,
            )
            return set()

        result: Set[int] = set()

        for val in raw_list:
            idx: Optional[int] = None
            # int — напрямую (bool отклоняется как подкласс)
            if isinstance(val, bool):
                continue
            if isinstance(val, int):
                idx = val
            elif isinstance(val, str):
                # строковые "1", "2" — парсим
                try:
                    idx = int(val.strip())
                except (ValueError, TypeError):
                    continue

            if idx is None:
                continue  # float, None, dict и прочее — пропускаем

            # проверяем границы чанка
            if not (1 <= idx <= chunk_len):
                # несуществующий индекс — игнорируем
                continue

            # Guard (a): лидер чанка (idx=1 1-based) не пропускается как дубль,
            # потому что у него нет «более раннего» в пределах этого чанка.
            # LLM может вернуть 1 по ошибке или спаму.
            #
            # Осознанный компромисс: лидер чанка (в т.ч. кросс-цикловый дубль-
            # лидер) не может быть помечен дублем. Это защищает от ложного скипа
            # ценой пропуска кросс-циклового дубля-лидера. Риск принят: выигрыш
            # от защиты (A/B-пары не теряют «ранний» item) превышает редкий
            # пропуск дубля-лидера.
            if idx == 1:
                logger.debug(
                    "LLM-дедуп: модель пометила лидера чанка (1-based idx=1) как "
                    "дубль — игнорируем (защита от самоссылки)",
                )
                continue

            result.add(chunk_offset + (idx - 1))

        return result