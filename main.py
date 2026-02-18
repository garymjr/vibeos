from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
import tomllib
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from telegram import Message, MessageEntity, Update
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import Application, ApplicationBuilder, ContextTypes, MessageHandler, filters

LOGGER = logging.getLogger("assistant.bot")
DEFAULT_CONFIG_PATH = Path("bot.config.toml")

_PI_RPC_CLIENT_CLASS: Any | None = None
_PI_RPC_ERROR_CLASS: type[Exception] = Exception

DeltaHandler = Callable[[str], Awaitable[None]]


def _as_table(value: object, *, name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeError(f"[{name}] must be a TOML table")
    return value


def _get_required_string(table: dict[str, object], key: str, *, section: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"[{section}] {key} must be a non-empty string")
    return value.strip()


def _get_optional_string(table: dict[str, object], key: str, *, section: str) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError(f"[{section}] {key} must be a string when set")
    stripped = value.strip()
    return stripped or None


def _get_required_positive_int(table: dict[str, object], key: str, *, section: str) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(f"[{section}] {key} must be a positive integer")
    return value


def _get_positive_int(table: dict[str, object], key: str, *, section: str, default: int) -> int:
    value = table.get(key)
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(f"[{section}] {key} must be a positive integer")
    return value


def _get_non_negative_int(table: dict[str, object], key: str, *, section: str, default: int) -> int:
    value = table.get(key)
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"[{section}] {key} must be a non-negative integer")
    return value


def _get_positive_int_list(table: dict[str, object], key: str, *, section: str) -> list[int]:
    value = table.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise RuntimeError(f"[{section}] {key} must be a list of positive integers when set")

    values: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise RuntimeError(f"[{section}] {key}[{index}] must be a positive integer")
        values.append(item)

    return list(dict.fromkeys(values))


@dataclass(slots=True)
class BotConfig:
    telegram_bot_token: str
    telegram_bot_owner_user_id: int
    telegram_trusted_user_ids: tuple[int, ...]
    pi_executable: str
    pi_provider: str | None
    pi_model: str | None
    pi_data_dir: Path | None
    pi_session_root: Path
    pi_session_ttl_seconds: int
    pi_call_timeout_seconds: int
    pi_session_sweeper_interval_seconds: int
    bot_queue_maxsize: int
    bot_worker_concurrency: int
    bot_heartbeat_interval_seconds: int
    bot_metrics_interval_seconds: int
    bot_stream_edit_interval_ms: int
    bot_latency_window_size: int
    log_level: str


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> BotConfig:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise RuntimeError(
            f"Config file not found at {resolved_path}. "
            "Create bot.config.toml from bot.config.toml.example."
        )

    with resolved_path.open("rb") as handle:
        config_data = tomllib.load(handle)

    if not isinstance(config_data, dict):
        raise RuntimeError("Config root must be a TOML table")

    telegram_config = _as_table(config_data.get("telegram"), name="telegram")
    if not telegram_config:
        raise RuntimeError("Missing required [telegram] section in bot.config.toml")

    pi_config = _as_table(config_data.get("pi"), name="pi")
    bot_config = _as_table(config_data.get("bot"), name="bot")

    token = _get_required_string(telegram_config, "bot_token", section="telegram")
    owner_user_id = _get_required_positive_int(telegram_config, "bot_owner_user_id", section="telegram")
    trusted_user_ids = _get_positive_int_list(telegram_config, "trusted_user_ids", section="telegram")

    pi_executable = _get_optional_string(pi_config, "executable", section="pi") or "pi"
    pi_provider = _get_optional_string(pi_config, "provider", section="pi")
    pi_model = _get_optional_string(pi_config, "model", section="pi")
    pi_data_dir_raw = _get_optional_string(pi_config, "data_dir", section="pi")
    pi_session_root = _get_optional_string(pi_config, "session_root", section="pi") or ".pi_sessions"
    pi_session_ttl_seconds = _get_non_negative_int(
        pi_config,
        "session_ttl_seconds",
        section="pi",
        default=0,
    )
    pi_call_timeout_seconds = _get_non_negative_int(
        pi_config,
        "call_timeout_seconds",
        section="pi",
        default=180,
    )
    pi_session_sweeper_interval_seconds = _get_non_negative_int(
        pi_config,
        "session_sweeper_interval_seconds",
        section="pi",
        default=60,
    )

    bot_queue_maxsize = _get_positive_int(bot_config, "queue_maxsize", section="bot", default=1000)
    bot_worker_concurrency = _get_positive_int(bot_config, "worker_concurrency", section="bot", default=4)
    bot_heartbeat_interval_seconds = _get_non_negative_int(
        bot_config,
        "heartbeat_interval_seconds",
        section="bot",
        default=0,
    )
    bot_metrics_interval_seconds = _get_non_negative_int(
        bot_config,
        "metrics_interval_seconds",
        section="bot",
        default=60,
    )
    bot_stream_edit_interval_ms = _get_positive_int(
        bot_config,
        "stream_edit_interval_ms",
        section="bot",
        default=700,
    )
    bot_latency_window_size = _get_positive_int(
        bot_config,
        "latency_window_size",
        section="bot",
        default=200,
    )
    log_level = (_get_optional_string(bot_config, "log_level", section="bot") or "INFO").upper()

    pi_data_dir = Path(pi_data_dir_raw).expanduser().resolve() if pi_data_dir_raw else None
    if pi_data_dir is not None:
        pi_data_dir.mkdir(parents=True, exist_ok=True)

    return BotConfig(
        telegram_bot_token=token,
        telegram_bot_owner_user_id=owner_user_id,
        telegram_trusted_user_ids=tuple(trusted_user_ids),
        pi_executable=pi_executable,
        pi_provider=pi_provider,
        pi_model=pi_model,
        pi_data_dir=pi_data_dir,
        pi_session_root=Path(pi_session_root).expanduser().resolve(),
        pi_session_ttl_seconds=pi_session_ttl_seconds,
        pi_call_timeout_seconds=pi_call_timeout_seconds,
        pi_session_sweeper_interval_seconds=pi_session_sweeper_interval_seconds,
        bot_queue_maxsize=bot_queue_maxsize,
        bot_worker_concurrency=bot_worker_concurrency,
        bot_heartbeat_interval_seconds=bot_heartbeat_interval_seconds,
        bot_metrics_interval_seconds=bot_metrics_interval_seconds,
        bot_stream_edit_interval_ms=bot_stream_edit_interval_ms,
        bot_latency_window_size=bot_latency_window_size,
        log_level=log_level,
    )


def _load_pi_sdk() -> tuple[Any, type[Exception]]:
    global _PI_RPC_CLIENT_CLASS, _PI_RPC_ERROR_CLASS
    if _PI_RPC_CLIENT_CLASS is not None:
        return _PI_RPC_CLIENT_CLASS, _PI_RPC_ERROR_CLASS

    try:
        from pi_sdk import PiRPCClient, PiRPCError
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import pi_sdk. "
            "Install pi_sdk in the active Python environment."
        ) from exc

    _PI_RPC_CLIENT_CLASS = PiRPCClient
    _PI_RPC_ERROR_CLASS = PiRPCError
    return _PI_RPC_CLIENT_CLASS, _PI_RPC_ERROR_CLASS


def _configure_pi_environment(config: BotConfig) -> None:
    if config.pi_data_dir is None:
        return

    os.environ["PI_CODING_AGENT_DIR"] = str(config.pi_data_dir)
    LOGGER.info("Configured PI_CODING_AGENT_DIR=%s", config.pi_data_dir)


@dataclass(slots=True)
class QueuedMessage:
    message: Message
    content: str
    enqueued_at_monotonic: float


@dataclass(slots=True)
class PiClientState:
    client: Any
    created_at_monotonic: float
    last_used_monotonic: float
    session_dir: Path


class PersonalAssistantBot:
    def __init__(self, config: BotConfig) -> None:
        self.message_queue: asyncio.Queue[QueuedMessage] = asyncio.Queue(maxsize=config.bot_queue_maxsize)
        self._queue_worker_tasks: list[asyncio.Task[None]] = []
        self._next_worker_id = 1
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._session_sweeper_task: asyncio.Task[None] | None = None
        self._metrics_task: asyncio.Task[None] | None = None

        self._pi_clients: dict[str, PiClientState] = {}
        self._conversation_locks: dict[str, asyncio.Lock] = {}
        self._queued_message_enqueued_at: dict[tuple[int, int], float] = {}
        self._conversation_latency_seconds: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=config.bot_latency_window_size)
        )
        self._timeout_count = 0

        self._bot_owner_user_id = config.telegram_bot_owner_user_id
        self._trusted_user_ids = set(config.telegram_trusted_user_ids)
        self._application: Application | None = None
        self._bot_user_id: int | None = None
        self._bot_username: str | None = None
        self._pi_executable = config.pi_executable
        self._pi_provider = config.pi_provider
        self._pi_model = config.pi_model
        self._pi_data_dir = config.pi_data_dir
        self._pi_session_root = config.pi_session_root
        self._pi_session_ttl_seconds = config.pi_session_ttl_seconds
        self._pi_call_timeout_seconds = config.pi_call_timeout_seconds
        self._pi_session_sweeper_interval_seconds = config.pi_session_sweeper_interval_seconds
        self._session_cleanup_ttl_seconds = config.pi_session_ttl_seconds if config.pi_session_ttl_seconds > 0 else 1800

        self._worker_concurrency = config.bot_worker_concurrency
        self._heartbeat_interval_seconds = config.bot_heartbeat_interval_seconds
        self._metrics_interval_seconds = config.bot_metrics_interval_seconds
        self._stream_edit_interval_seconds = config.bot_stream_edit_interval_ms / 1000

        self._pi_session_root.mkdir(parents=True, exist_ok=True)

    async def on_ready(self, application: Application) -> None:
        self._application = application
        bot_user = await application.bot.get_me()
        self._bot_user_id = bot_user.id
        self._bot_username = bot_user.username.lower() if bot_user.username else None
        LOGGER.info("Telegram bot connected as %s (id=%s)", bot_user.full_name, bot_user.id)
        self._ensure_queue_workers()

        if (
            self._heartbeat_interval_seconds > 0
            and (self._heartbeat_task is None or self._heartbeat_task.done())
        ):
            self._heartbeat_task = asyncio.create_task(self._heartbeat_worker(), name="heartbeat-worker")
            LOGGER.info("Heartbeat worker started with interval=%ss", self._heartbeat_interval_seconds)

        if (
            self._pi_session_sweeper_interval_seconds > 0
            and (self._session_sweeper_task is None or self._session_sweeper_task.done())
        ):
            self._session_sweeper_task = asyncio.create_task(
                self._session_sweeper_worker(),
                name="pi-session-sweeper-worker",
            )
            LOGGER.info(
                "Session sweeper started with interval=%ss",
                self._pi_session_sweeper_interval_seconds,
            )

        if self._metrics_interval_seconds > 0 and (self._metrics_task is None or self._metrics_task.done()):
            self._metrics_task = asyncio.create_task(self._metrics_worker(), name="metrics-worker")
            LOGGER.info("Metrics worker started with interval=%ss", self._metrics_interval_seconds)

    async def on_message(self, update: Update, _context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        sender = update.effective_user
        if message is None or sender is None:
            return
        if sender.is_bot:
            return
        if not self._is_authorized_user(sender.id):
            return
        if message.chat.type != "private" and not self._is_bot_mentioned(message):
            return

        text = (message.text or message.caption or "").strip()
        if not text:
            return

        enqueued_at = time.monotonic()
        queued = QueuedMessage(message=message, content=text, enqueued_at_monotonic=enqueued_at)
        await self.message_queue.put(queued)
        self._queued_message_enqueued_at[self._message_key(message)] = enqueued_at

        LOGGER.info(
            "Queued message %s from user %s in chat %s (queue=%s)",
            message.message_id,
            sender.id,
            message.chat_id,
            self.message_queue.qsize(),
        )

    async def on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update is None:
            LOGGER.exception("Telegram polling error", exc_info=context.error)
            return
        LOGGER.exception("Telegram update handler failed for update=%s", update, exc_info=context.error)

    def _ensure_queue_workers(self) -> None:
        alive_workers: list[asyncio.Task[None]] = []
        for task in self._queue_worker_tasks:
            if task.done():
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None:
                    LOGGER.error("Queue worker exited with an error", exc_info=exc)
                continue
            alive_workers.append(task)

        self._queue_worker_tasks = alive_workers

        while len(self._queue_worker_tasks) < self._worker_concurrency:
            worker_id = self._next_worker_id
            self._next_worker_id += 1
            task = asyncio.create_task(
                self._queue_worker(worker_id),
                name=f"telegram-message-queue-worker-{worker_id}",
            )
            self._queue_worker_tasks.append(task)

        LOGGER.info("Queue worker pool ready (workers=%s)", len(self._queue_worker_tasks))

    def _is_authorized_user(self, user_id: int) -> bool:
        return user_id == self._bot_owner_user_id or user_id in self._trusted_user_ids

    def _is_bot_mentioned(self, message: Message) -> bool:
        bot_username = self._bot_username
        if bot_username is not None:
            mention = f"@{bot_username}"
            mentions = self._extract_mentions(message.text, message.entities)
            mentions.update(self._extract_mentions(message.caption, message.caption_entities))
            if mention in mentions:
                return True

        if self._bot_user_id is not None:
            replied = message.reply_to_message
            if replied is not None and replied.from_user is not None and replied.from_user.id == self._bot_user_id:
                return True

        return False

    @staticmethod
    def _extract_mentions(text: str | None, entities: tuple[MessageEntity, ...] | None) -> set[str]:
        if not text or not entities:
            return set()

        mentions: set[str] = set()
        for entity in entities:
            if entity.type != MessageEntity.MENTION:
                continue
            start = entity.offset
            end = start + entity.length
            if start < 0 or end > len(text):
                continue
            token = text[start:end].strip().lower()
            if token.startswith("@"):
                mentions.add(token)
        return mentions

    async def shutdown(self, _application: Application) -> None:
        await self._cancel_task(self._heartbeat_task, "heartbeat worker")
        self._heartbeat_task = None

        await self._cancel_task(self._session_sweeper_task, "session sweeper")
        self._session_sweeper_task = None

        await self._cancel_task(self._metrics_task, "metrics worker")
        self._metrics_task = None

        for task in self._queue_worker_tasks:
            task.cancel()
        if self._queue_worker_tasks:
            await asyncio.gather(*self._queue_worker_tasks, return_exceptions=True)
        self._queue_worker_tasks = []

        await asyncio.to_thread(self._close_pi_clients)

    async def _cancel_task(self, task: asyncio.Task[None] | None, name: str) -> None:
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            LOGGER.info("%s stopped", name)
        except Exception:  # noqa: BLE001
            LOGGER.exception("%s failed during shutdown", name)

    def _conversation_lock(self, conversation_key: str) -> asyncio.Lock:
        lock = self._conversation_locks.get(conversation_key)
        if lock is None:
            lock = asyncio.Lock()
            self._conversation_locks[conversation_key] = lock
        return lock

    async def _queue_worker(self, worker_id: int) -> None:
        LOGGER.info("Queue worker %s started", worker_id)
        try:
            while True:
                queued = await self.message_queue.get()
                self._queued_message_enqueued_at.pop(self._message_key(queued.message), None)
                try:
                    await self._process_queued_message(queued)
                except Exception:  # noqa: BLE001
                    LOGGER.exception("Unexpected error while handling queued message %s", queued.message.message_id)
                    try:
                        await self._send_text(
                            queued.message.chat_id,
                            "I hit an unexpected error while processing that message. Please try again.",
                        )
                    except TelegramError:
                        LOGGER.exception(
                            "Failed to send unexpected-error message for queued message %s",
                            queued.message.message_id,
                        )
                finally:
                    self.message_queue.task_done()
        except asyncio.CancelledError:
            LOGGER.info("Queue worker %s stopped", worker_id)
            raise

    async def _process_queued_message(self, queued: QueuedMessage) -> None:
        message = queued.message
        conversation_key = self._conversation_key(message)
        prompt = self._prompt_from_message(queued)
        sender = message.from_user
        sender_id = sender.id if sender is not None else 0
        force_session = message.chat.type == "private" and sender_id == self._bot_owner_user_id
        _, pi_error_type = _load_pi_sdk()

        queue_wait_seconds = max(0.0, time.monotonic() - queued.enqueued_at_monotonic)
        lock = self._conversation_lock(conversation_key)

        async with lock:
            started_at = time.monotonic()
            outcome = "success"
            response_chars = 0
            LOGGER.info(
                "Run started message=%s conversation=%s queue_wait_ms=%.0f",
                message.message_id,
                conversation_key,
                queue_wait_seconds * 1000,
            )

            status_message: Message | None = None
            status_updates_enabled = True
            streamed_text = ""
            last_edit_at = 0.0

            try:
                status_message = await self._send_text(message.chat_id, "Thinking...")
            except TelegramError:
                LOGGER.exception("Failed to send initial progress message for %s", message.message_id)

            async def on_delta(delta: str) -> None:
                nonlocal streamed_text, last_edit_at, status_updates_enabled

                if not delta:
                    return

                streamed_text += delta
                if status_message is None or not status_updates_enabled:
                    return

                now = time.monotonic()
                if now - last_edit_at < self._stream_edit_interval_seconds:
                    return

                preview = self._stream_preview(streamed_text)
                try:
                    await status_message.edit_text(preview)
                    last_edit_at = now
                except TelegramError:
                    status_updates_enabled = False
                    LOGGER.exception("Failed to edit streaming response for %s", message.message_id)

            try:
                try:
                    await self._send_chat_action(message.chat_id, ChatAction.TYPING)
                    response_text = await self._run_pi_prompt_async(
                        conversation_key,
                        prompt,
                        force_session=force_session,
                        on_delta=on_delta,
                    )
                except asyncio.TimeoutError:
                    outcome = "timeout"
                    await self._send_or_edit_response(
                        message.chat_id,
                        status_message,
                        "That request timed out while waiting for pi. Please try again.",
                    )
                    return
                except pi_error_type as exc:
                    outcome = "pi_error"
                    LOGGER.exception("pi_sdk error for message %s: %s", message.message_id, exc)
                    await self._send_or_edit_response(message.chat_id, status_message, f"pi error: {exc}")
                    return
                except Exception:
                    outcome = "unexpected_error"
                    raise

                if not response_text.strip():
                    response_text = "I did not get a response from pi for that message."

                await self._publish_final_response(message.chat_id, status_message, response_text)
                response_chars = len(response_text)

                LOGGER.info("Run response delivered message=%s conversation=%s", message.message_id, conversation_key)
            finally:
                elapsed_seconds = time.monotonic() - started_at
                self._record_latency(conversation_key, elapsed_seconds)
                LOGGER.info(
                    "Run finished message=%s conversation=%s outcome=%s latency_ms=%.0f response_chars=%s",
                    message.message_id,
                    conversation_key,
                    outcome,
                    elapsed_seconds * 1000,
                    response_chars,
                )

    async def _send_or_edit_response(
        self,
        chat_id: int,
        status_message: Message | None,
        text: str,
    ) -> None:
        if status_message is not None:
            try:
                await status_message.edit_text(text)
                return
            except TelegramError:
                LOGGER.exception("Failed to edit status message with final response")

        await self._send_text(chat_id, text)

    async def _publish_final_response(
        self,
        chat_id: int,
        status_message: Message | None,
        response_text: str,
    ) -> None:
        chunks = self._split_for_telegram(response_text)
        first_chunk = chunks[0]

        if status_message is not None:
            try:
                await status_message.edit_text(first_chunk)
            except TelegramError:
                LOGGER.exception("Failed to edit status message with first response chunk")
                await self._send_text(chat_id, first_chunk)
        else:
            await self._send_text(chat_id, first_chunk)

        for chunk in chunks[1:]:
            await self._send_text(chat_id, chunk)

    @staticmethod
    def _stream_preview(text: str, *, limit: int = 4096) -> str:
        if not text:
            return "Thinking..."
        if len(text) <= limit:
            return text

        header = f"[streaming, {len(text)} chars]\n"
        tail_size = max(1, limit - len(header))
        return header + text[-tail_size:]

    def _require_application(self) -> Application:
        if self._application is None:
            raise RuntimeError("Telegram application is not initialized")
        return self._application

    async def _send_text(self, chat_id: int, text: str) -> Message:
        application = self._require_application()
        return await application.bot.send_message(chat_id=chat_id, text=text)

    async def _send_chat_action(self, chat_id: int, action: str) -> None:
        application = self._require_application()
        await application.bot.send_chat_action(chat_id=chat_id, action=action)

    @staticmethod
    def _message_key(message: Message) -> tuple[int, int]:
        return (message.chat_id, message.message_id)

    async def _heartbeat_worker(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval_seconds)
                try:
                    await self._run_heartbeat_once()
                except Exception:  # noqa: BLE001
                    LOGGER.exception("Unexpected error while running heartbeat")
        except asyncio.CancelledError:
            LOGGER.info("Heartbeat worker stopped")
            raise

    async def _session_sweeper_worker(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._pi_session_sweeper_interval_seconds)
                try:
                    await self._run_session_sweep_once()
                except Exception:  # noqa: BLE001
                    LOGGER.exception("Unexpected error while sweeping sessions")
        except asyncio.CancelledError:
            LOGGER.info("Session sweeper stopped")
            raise

    async def _metrics_worker(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._metrics_interval_seconds)
                self._log_health_metrics(source="periodic")
        except asyncio.CancelledError:
            LOGGER.info("Metrics worker stopped")
            raise

    async def _run_heartbeat_once(self) -> None:
        _, pi_error_type = _load_pi_sdk()
        heartbeat_key = f"heartbeat-owner-{self._bot_owner_user_id}"
        heartbeat_prompt = (
            "This is a heartbeat session. "
            "You are running in an ephemeral context for a Telegram bot heartbeat check. "
            "First, read ~/.vibeos/HEARTBEAT.md and follow its instructions exactly. "
            "Log your checks in ~/.vibeos/memory/heartbeat-state.json. "
            "Reply with a short heartbeat status update only if there's something urgent, "
            "otherwise reply HEARTBEAT_OK."
        )

        heartbeat_lock = self._conversation_lock(heartbeat_key)
        try:
            async with heartbeat_lock:
                heartbeat_response = await self._run_pi_prompt_async(
                    heartbeat_key,
                    heartbeat_prompt,
                    force_ephemeral=True,
                )
        except asyncio.TimeoutError:
            LOGGER.warning("Heartbeat prompt timed out")
            return
        except pi_error_type as exc:
            LOGGER.exception("pi_sdk error while running heartbeat prompt: %s", exc)
            return

        if not heartbeat_response.strip():
            heartbeat_response = "Heartbeat response was empty."

        owner_chat_id = await self._get_or_create_owner_dm_chat_id()
        owner_key = self._dm_conversation_key(self._bot_owner_user_id, owner_chat_id)
        owner_prompt = (
            "Heartbeat update from the bot's heartbeat session.\n"
            f"Heartbeat response: {heartbeat_response}"
        )

        owner_lock = self._conversation_lock(owner_key)
        try:
            async with owner_lock:
                owner_response = await self._run_pi_prompt_async(owner_key, owner_prompt, force_session=True)
        except asyncio.TimeoutError:
            LOGGER.warning("Heartbeat forward prompt timed out for owner conversation %s", owner_key)
            return
        except pi_error_type as exc:
            LOGGER.exception("pi_sdk error while forwarding heartbeat to owner DM session: %s", exc)
            return

        if not owner_response.strip():
            owner_response = "I processed the heartbeat update, but the owner session returned no response."

        for chunk in self._split_for_telegram(owner_response):
            await self._send_text(owner_chat_id, chunk)

        LOGGER.info("Forwarded heartbeat response into owner DM session %s", owner_key)
        self._log_health_metrics(source="heartbeat")

    async def _run_session_sweep_once(self) -> None:
        now = time.monotonic()
        expired_keys: list[str] = []

        for conversation_key, state in list(self._pi_clients.items()):
            lock = self._conversation_locks.get(conversation_key)
            if lock is not None and lock.locked():
                continue
            if now - state.last_used_monotonic >= self._session_cleanup_ttl_seconds:
                expired_keys.append(conversation_key)

        for conversation_key in expired_keys:
            await self._evict_pi_client(conversation_key, reason="sweeper-idle-expired")

        active_session_dirs = {
            state.session_dir.resolve()
            for state in self._pi_clients.values()
        }
        deleted_dirs = await asyncio.to_thread(
            self._prune_stale_session_directories,
            active_session_dirs,
            self._session_cleanup_ttl_seconds,
        )

        if expired_keys or deleted_dirs:
            LOGGER.info(
                "Session sweep completed expired_clients=%s deleted_dirs=%s active_sessions=%s",
                len(expired_keys),
                deleted_dirs,
                len(self._pi_clients),
            )

    def _prune_stale_session_directories(self, active_session_dirs: set[Path], max_age_seconds: int) -> int:
        if not self._pi_session_root.exists():
            return 0

        now = time.time()
        deleted_dirs = 0

        for conversation_dir in self._pi_session_root.iterdir():
            if not conversation_dir.is_dir():
                continue

            for session_dir in conversation_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                resolved_dir = session_dir.resolve()
                if resolved_dir in active_session_dirs:
                    continue

                try:
                    age_seconds = now - session_dir.stat().st_mtime
                except OSError:
                    continue

                if age_seconds < max_age_seconds:
                    continue

                try:
                    shutil.rmtree(session_dir)
                    deleted_dirs += 1
                except FileNotFoundError:
                    continue
                except Exception:  # noqa: BLE001
                    LOGGER.exception("Failed deleting stale session directory %s", session_dir)

            try:
                conversation_dir.rmdir()
            except OSError:
                continue

        return deleted_dirs

    async def _run_pi_prompt_async(
        self,
        conversation_key: str,
        prompt: str,
        *,
        force_ephemeral: bool = False,
        force_session: bool = False,
        on_delta: DeltaHandler | None = None,
    ) -> str:
        if force_ephemeral and force_session:
            raise RuntimeError("force_ephemeral and force_session cannot both be true")

        run_coro = self._run_pi_prompt_once(
            conversation_key,
            prompt,
            force_ephemeral=force_ephemeral,
            force_session=force_session,
            on_delta=on_delta,
        )

        if self._pi_call_timeout_seconds <= 0:
            return await run_coro

        try:
            return await asyncio.wait_for(run_coro, timeout=self._pi_call_timeout_seconds)
        except asyncio.TimeoutError:
            self._timeout_count += 1
            LOGGER.warning(
                "Pi call timed out for %s after %ss",
                conversation_key,
                self._pi_call_timeout_seconds,
            )
            await self._evict_pi_client(conversation_key, reason="timeout")
            raise

    async def _run_pi_prompt_once(
        self,
        conversation_key: str,
        prompt: str,
        *,
        force_ephemeral: bool,
        force_session: bool,
        on_delta: DeltaHandler | None,
    ) -> str:
        if force_ephemeral or (self._pi_session_ttl_seconds == 0 and not force_session):
            client, session_dir = await asyncio.to_thread(self._create_pi_client, conversation_key, force_no_session=True)
            try:
                response_text = await self._stream_text_from_client_async(
                    conversation_key,
                    client,
                    prompt,
                    on_delta=on_delta,
                )
                LOGGER.info("Returned ephemeral pi response for %s", conversation_key)
                return response_text.strip()
            finally:
                await asyncio.shield(
                    asyncio.to_thread(
                        self._close_pi_client,
                        conversation_key,
                        client,
                        session_dir,
                        True,
                    )
                )

        state = await self._get_pi_client(conversation_key, force_session=force_session)
        state.last_used_monotonic = time.monotonic()
        try:
            response_text = await self._stream_text_from_client_async(
                conversation_key,
                state.client,
                prompt,
                on_delta=on_delta,
            )
        finally:
            state.last_used_monotonic = time.monotonic()

        LOGGER.info("Returned pi response for %s", conversation_key)
        return response_text.strip()

    async def _stream_text_from_client_async(
        self,
        conversation_key: str,
        client: Any,
        prompt: str,
        *,
        on_delta: DeltaHandler | None,
    ) -> str:
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[tuple[str, object]] = asyncio.Queue()

        def emit(event: str, payload: object) -> None:
            loop.call_soon_threadsafe(events.put_nowait, (event, payload))

        def stream_worker() -> None:
            try:
                for delta in client.stream_text(prompt):
                    emit("delta", delta)
            except Exception as exc:  # noqa: BLE001
                emit("error", exc)
            else:
                emit("done", None)

        stream_task = asyncio.create_task(asyncio.to_thread(stream_worker), name=f"pi-stream-{conversation_key}")
        stream_task.add_done_callback(self._discard_task_exception)

        chunks: list[str] = []
        while True:
            event, payload = await events.get()
            if event == "delta":
                delta = payload if isinstance(payload, str) else str(payload)
                chunks.append(delta)
                if on_delta is not None and delta:
                    await on_delta(delta)
                continue

            if event == "error":
                if isinstance(payload, Exception):
                    raise payload
                raise RuntimeError("pi stream failed")

            if event == "done":
                break

        return "".join(chunks)

    @staticmethod
    def _discard_task_exception(task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            LOGGER.exception("PI stream task failed", exc_info=exc)

    async def _evict_pi_client(self, conversation_key: str, *, reason: str) -> None:
        state = self._pi_clients.pop(conversation_key, None)
        if state is None:
            return

        await asyncio.to_thread(
            self._close_pi_client,
            conversation_key,
            state.client,
            state.session_dir,
            True,
        )
        LOGGER.info("Evicted pi client for %s reason=%s", conversation_key, reason)

    async def _get_or_create_owner_dm_chat_id(self) -> int:
        application = self._require_application()
        await application.bot.get_chat(self._bot_owner_user_id)
        return self._bot_owner_user_id

    async def _get_pi_client(self, conversation_key: str, *, force_session: bool = False) -> PiClientState:
        if self._pi_session_ttl_seconds == 0 and not force_session:
            raise RuntimeError("Session TTL of 0 requires per-message client creation")

        now = time.monotonic()
        existing = self._pi_clients.get(conversation_key)
        if existing is not None:
            if force_session and self._pi_session_ttl_seconds == 0:
                existing.last_used_monotonic = now
                LOGGER.info("Resuming forced pi session for %s", conversation_key)
                return existing

            if self._pi_session_ttl_seconds > 0 and now - existing.last_used_monotonic < self._pi_session_ttl_seconds:
                existing.last_used_monotonic = now
                LOGGER.info("Resuming pi session for %s", conversation_key)
                return existing

            await self._evict_pi_client(conversation_key, reason="ttl-expired")

        client, session_dir = await asyncio.to_thread(self._create_pi_client, conversation_key)
        state = PiClientState(
            client=client,
            created_at_monotonic=now,
            last_used_monotonic=now,
            session_dir=session_dir,
        )
        self._pi_clients[conversation_key] = state
        return state

    def _create_pi_client(
        self,
        conversation_key: str,
        *,
        force_no_session: bool = False,
        pi_rpc_client_class: Any | None = None,
    ) -> tuple[Any, Path]:
        if pi_rpc_client_class is None:
            pi_rpc_client_class, _ = _load_pi_sdk()

        session_dir = self._pi_session_root / conversation_key / f"session-{time.time_ns()}"
        session_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Starting pi session for %s", conversation_key)
        kwargs = self._build_pi_client_kwargs(
            force_no_session=force_no_session,
            session_dir=session_dir,
        )

        try:
            client = pi_rpc_client_class(**kwargs)
        except TypeError as exc:
            if self._pi_data_dir is not None and "cwd" in str(exc):
                raise RuntimeError(
                    "Configured [pi] data_dir requires pi_sdk with PiRPCClient(cwd=...). "
                    "Update your pi_sdk installation."
                ) from exc
            raise

        client.start()
        return client, session_dir

    def _build_pi_client_kwargs(self, *, force_no_session: bool, session_dir: Path) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "executable": self._pi_executable,
            "provider": self._pi_provider,
            "model": self._pi_model,
            "no_session": force_no_session,
            "session_dir": session_dir,
        }
        if self._pi_data_dir is not None:
            kwargs["cwd"] = self._pi_data_dir
        return kwargs

    def _close_pi_clients(self) -> None:
        for key, state in list(self._pi_clients.items()):
            self._close_pi_client(key, state.client, state.session_dir, True)
        self._pi_clients.clear()

    @staticmethod
    def _close_pi_client(
        conversation_key: str,
        client: Any,
        session_dir: Path | None = None,
        delete_session_dir: bool = False,
    ) -> None:
        try:
            client.close()
            LOGGER.info("Finished pi session for %s", conversation_key)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed closing pi client for %s", conversation_key)

        if delete_session_dir and session_dir is not None:
            try:
                shutil.rmtree(session_dir)
            except FileNotFoundError:
                return
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed deleting session directory %s for %s", session_dir, conversation_key)

    def _record_latency(self, conversation_key: str, elapsed_seconds: float) -> None:
        self._conversation_latency_seconds[conversation_key].append(elapsed_seconds)

    def _queue_oldest_age_seconds(self) -> float:
        if not self._queued_message_enqueued_at:
            return 0.0
        oldest_enqueued = min(self._queued_message_enqueued_at.values())
        return max(0.0, time.monotonic() - oldest_enqueued)

    @staticmethod
    def _percentile(sorted_values: list[float], percentile: float) -> float:
        if not sorted_values:
            return 0.0
        index = int((len(sorted_values) - 1) * percentile)
        return sorted_values[index]

    def _latency_summary(self) -> dict[str, dict[str, float | int]]:
        summary: dict[str, dict[str, float | int]] = {}
        for conversation_key, samples in self._conversation_latency_seconds.items():
            if not samples:
                continue
            values = sorted(samples)
            summary[conversation_key] = {
                "count": len(values),
                "p50_ms": round(self._percentile(values, 0.50) * 1000, 2),
                "p95_ms": round(self._percentile(values, 0.95) * 1000, 2),
                "p99_ms": round(self._percentile(values, 0.99) * 1000, 2),
            }
        return summary

    def _log_health_metrics(self, *, source: str) -> None:
        active_runs = sum(1 for lock in self._conversation_locks.values() if lock.locked())
        LOGGER.info(
            "Health source=%s queue_depth=%s queue_oldest_age_s=%.2f active_runs=%s active_sessions=%s timeouts=%s latency=%s",
            source,
            self.message_queue.qsize(),
            self._queue_oldest_age_seconds(),
            active_runs,
            len(self._pi_clients),
            self._timeout_count,
            self._latency_summary(),
        )

    @staticmethod
    def _conversation_key(message: Message) -> str:
        if message.chat.type == "private":
            sender_id = message.from_user.id if message.from_user is not None else 0
            return PersonalAssistantBot._dm_conversation_key(sender_id, message.chat_id)
        return f"chat-{message.chat_id}"

    @staticmethod
    def _dm_conversation_key(user_id: int, channel_id: int) -> str:
        return f"dm-user-{user_id}-channel-{channel_id}"

    @staticmethod
    def _prompt_from_message(queued: QueuedMessage) -> str:
        message = queued.message
        if message.chat.type == "private":
            location = "direct message"
        else:
            location = f"chat_type={message.chat.type} chat={message.chat.title or message.chat_id}"
        sender = message.from_user
        sender_name = sender.full_name if sender is not None else "unknown"
        sender_id = sender.id if sender is not None else 0
        return (
            "You are a helpful personal assistant on Telegram.\n"
            f"Author: {sender_name} (id={sender_id})\n"
            f"Location: {location}\n"
            f"Message: {queued.content}"
        )

    @staticmethod
    def _split_for_telegram(text: str, *, limit: int = 4096) -> list[str]:
        if len(text) <= limit:
            return [text]

        chunks: list[str] = []
        current = ""
        for line in text.splitlines(keepends=True):
            if len(line) > limit:
                if current:
                    chunks.append(current)
                    current = ""
                for idx in range(0, len(line), limit):
                    chunks.append(line[idx : idx + limit])
                continue

            if len(current) + len(line) <= limit:
                current += line
            else:
                chunks.append(current)
                current = line

        if current:
            chunks.append(current)
        return chunks


def _ensure_event_loop() -> None:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
        return

    if loop.is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())


def main() -> None:
    config = load_config()
    level = getattr(logging, config.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    provider = config.pi_provider or "<default>"
    model = config.pi_model or "<default>"
    LOGGER.info("Starting bot with pi provider=%s model=%s", provider, model)
    _configure_pi_environment(config)

    _load_pi_sdk()

    bot = PersonalAssistantBot(config)
    application = (
        ApplicationBuilder()
        .token(config.telegram_bot_token)
        .post_init(bot.on_ready)
        .post_shutdown(bot.shutdown)
        .build()
    )
    application.add_handler(MessageHandler(filters.TEXT | filters.CAPTION, bot.on_message))
    application.add_error_handler(bot.on_error)
    _ensure_event_loop()
    application.run_polling()


if __name__ == "__main__":
    main()
