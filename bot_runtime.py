from __future__ import annotations

import asyncio
import logging
import time

from telegram import Message, Update
from telegram.constants import ChatAction
from telegram.error import TelegramError
from telegram.ext import Application, ContextTypes

from configuration import BotConfig
from message_utils import (
    QueuedMessage,
    conversation_key,
    dm_conversation_key,
    is_bot_mentioned,
    message_key,
    prompt_from_message,
    split_for_telegram,
    stream_preview,
)
from pi_runtime import PiRuntime, load_pi_sdk
from telemetry import LatencyTracker

LOGGER = logging.getLogger("assistant.bot")


class PersonalAssistantBot:
    def __init__(self, config: BotConfig) -> None:
        self.message_queue: asyncio.Queue[QueuedMessage] = asyncio.Queue(maxsize=config.bot_queue_maxsize)
        self._queue_worker_tasks: list[asyncio.Task[None]] = []
        self._next_worker_id = 1
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._session_sweeper_task: asyncio.Task[None] | None = None
        self._metrics_task: asyncio.Task[None] | None = None

        self._conversation_locks: dict[str, asyncio.Lock] = {}
        self._queued_message_enqueued_at: dict[tuple[int, int], float] = {}
        self._latency_tracker = LatencyTracker(config.bot_latency_window_size)

        self._bot_owner_user_id = config.telegram_bot_owner_user_id
        self._trusted_user_ids = set(config.telegram_trusted_user_ids)
        self._application: Application | None = None
        self._bot_user_id: int | None = None
        self._bot_username: str | None = None

        self._worker_concurrency = config.bot_worker_concurrency
        self._heartbeat_interval_seconds = config.bot_heartbeat_interval_seconds
        self._metrics_interval_seconds = config.bot_metrics_interval_seconds
        self._stream_edit_interval_seconds = config.bot_stream_edit_interval_ms / 1000
        self._pi_session_sweeper_interval_seconds = config.pi_session_sweeper_interval_seconds

        self._pi_runtime = PiRuntime(config)

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
        self._queued_message_enqueued_at[message_key(message)] = enqueued_at

        LOGGER.debug(
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
        return is_bot_mentioned(message, bot_username=self._bot_username, bot_user_id=self._bot_user_id)

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

        await self._pi_runtime.close_all_clients()

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

    def _conversation_lock(self, conversation_key_value: str) -> asyncio.Lock:
        lock = self._conversation_locks.get(conversation_key_value)
        if lock is None:
            lock = asyncio.Lock()
            self._conversation_locks[conversation_key_value] = lock
        return lock

    async def _queue_worker(self, worker_id: int) -> None:
        LOGGER.debug("Queue worker %s started", worker_id)
        try:
            while True:
                queued = await self.message_queue.get()
                self._queued_message_enqueued_at.pop(message_key(queued.message), None)
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
            LOGGER.debug("Queue worker %s stopped", worker_id)
            raise

    async def _process_queued_message(self, queued: QueuedMessage) -> None:
        message = queued.message
        conversation_key_value = conversation_key(message)
        prompt = prompt_from_message(queued)
        sender = message.from_user
        sender_id = sender.id if sender is not None else 0
        force_session = message.chat.type == "private" and sender_id == self._bot_owner_user_id
        _, pi_error_type = load_pi_sdk()

        queue_wait_seconds = max(0.0, time.monotonic() - queued.enqueued_at_monotonic)
        lock = self._conversation_lock(conversation_key_value)

        async with lock:
            started_at = time.monotonic()
            outcome = "success"
            response_chars = 0
            LOGGER.info(
                "Run started message=%s conversation=%s queue_wait_ms=%.0f",
                message.message_id,
                conversation_key_value,
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

            try:
                await self._react_to_message(message, "👀")
            except TelegramError:
                LOGGER.exception("Failed to react to message %s", message.message_id)

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

                preview = stream_preview(streamed_text)
                try:
                    await status_message.edit_text(preview)
                    last_edit_at = now
                except TelegramError:
                    status_updates_enabled = False
                    LOGGER.exception("Failed to edit streaming response for %s", message.message_id)

            try:
                try:
                    await self._send_chat_action(message.chat_id, ChatAction.TYPING)
                    response_text = await self._pi_runtime.run_prompt(
                        conversation_key_value,
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
                LOGGER.debug(
                    "Run response delivered message=%s conversation=%s",
                    message.message_id,
                    conversation_key_value,
                )
            finally:
                elapsed_seconds = time.monotonic() - started_at
                self._latency_tracker.record(conversation_key_value, elapsed_seconds)
                LOGGER.info(
                    "Run finished message=%s conversation=%s outcome=%s latency_ms=%.0f response_chars=%s",
                    message.message_id,
                    conversation_key_value,
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
        chunks = split_for_telegram(response_text)
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

    async def _react_to_message(self, message: Message, reaction: str) -> None:
        await message.set_reaction(reaction=reaction)

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
        _, pi_error_type = load_pi_sdk()
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
                heartbeat_response = await self._pi_runtime.run_prompt(
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
        owner_key = dm_conversation_key(self._bot_owner_user_id, owner_chat_id)
        owner_prompt = (
            "Heartbeat update from the bot's heartbeat session.\n"
            f"Heartbeat response: {heartbeat_response}"
        )

        owner_lock = self._conversation_lock(owner_key)
        try:
            async with owner_lock:
                owner_response = await self._pi_runtime.run_prompt(owner_key, owner_prompt, force_session=True)
        except asyncio.TimeoutError:
            LOGGER.warning("Heartbeat forward prompt timed out for owner conversation %s", owner_key)
            return
        except pi_error_type as exc:
            LOGGER.exception("pi_sdk error while forwarding heartbeat to owner DM session: %s", exc)
            return

        if not owner_response.strip():
            owner_response = "I processed the heartbeat update, but the owner session returned no response."

        for chunk in split_for_telegram(owner_response):
            await self._send_text(owner_chat_id, chunk)

        LOGGER.info("Forwarded heartbeat response into owner DM session %s", owner_key)
        self._log_health_metrics(source="heartbeat")

    async def _run_session_sweep_once(self) -> None:
        expired_clients, deleted_dirs = await self._pi_runtime.sweep_stale_sessions(
            is_conversation_locked=self._is_conversation_locked
        )

        if expired_clients or deleted_dirs:
            LOGGER.info(
                "Session sweep completed expired_clients=%s deleted_dirs=%s active_sessions=%s",
                expired_clients,
                deleted_dirs,
                self._pi_runtime.active_session_count,
            )

    async def _get_or_create_owner_dm_chat_id(self) -> int:
        application = self._require_application()
        await application.bot.get_chat(self._bot_owner_user_id)
        return self._bot_owner_user_id

    def _is_conversation_locked(self, conversation_key_value: str) -> bool:
        lock = self._conversation_locks.get(conversation_key_value)
        return lock is not None and lock.locked()

    def _queue_oldest_age_seconds(self) -> float:
        if not self._queued_message_enqueued_at:
            return 0.0
        oldest_enqueued = min(self._queued_message_enqueued_at.values())
        return max(0.0, time.monotonic() - oldest_enqueued)

    def _log_health_metrics(self, *, source: str) -> None:
        active_runs = sum(1 for lock in self._conversation_locks.values() if lock.locked())
        LOGGER.info(
            "Health source=%s queue_depth=%s queue_oldest_age_s=%.2f active_runs=%s active_sessions=%s timeouts=%s latency=%s",
            source,
            self.message_queue.qsize(),
            self._queue_oldest_age_seconds(),
            active_runs,
            self._pi_runtime.active_session_count,
            self._pi_runtime.timeout_count,
            self._latency_tracker.summary(),
        )
