from __future__ import annotations

import asyncio
import logging
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import discord

LOGGER = logging.getLogger("assistant.bot")
DEFAULT_CONFIG_PATH = Path("bot.config.toml")

_PI_RPC_CLIENT_CLASS: Any | None = None
_PI_RPC_ERROR_CLASS: type[Exception] = Exception


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


@dataclass(slots=True)
class BotConfig:
    discord_bot_token: str
    discord_bot_owner_user_id: int
    pi_sdk_path: str | None
    pi_executable: str
    pi_provider: str | None
    pi_model: str | None
    pi_data_dir: Path | None
    pi_session_root: Path
    pi_session_ttl_seconds: int
    bot_queue_maxsize: int
    bot_heartbeat_interval_seconds: int
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

    discord_config = _as_table(config_data.get("discord"), name="discord")
    if not discord_config:
        raise RuntimeError("Missing required [discord] section in bot.config.toml")

    pi_config = _as_table(config_data.get("pi"), name="pi")
    bot_config = _as_table(config_data.get("bot"), name="bot")

    token = _get_required_string(discord_config, "bot_token", section="discord")
    owner_user_id = _get_required_positive_int(discord_config, "bot_owner_user_id", section="discord")
    pi_sdk_path = _get_optional_string(pi_config, "sdk_path", section="pi")
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
    bot_queue_maxsize = _get_positive_int(bot_config, "queue_maxsize", section="bot", default=1000)
    bot_heartbeat_interval_seconds = _get_non_negative_int(
        bot_config,
        "heartbeat_interval_seconds",
        section="bot",
        default=0,
    )
    log_level = (_get_optional_string(bot_config, "log_level", section="bot") or "INFO").upper()

    pi_data_dir = Path(pi_data_dir_raw).expanduser().resolve() if pi_data_dir_raw else None
    if pi_data_dir is not None:
        pi_data_dir.mkdir(parents=True, exist_ok=True)

    return BotConfig(
        discord_bot_token=token,
        discord_bot_owner_user_id=owner_user_id,
        pi_sdk_path=pi_sdk_path,
        pi_executable=pi_executable,
        pi_provider=pi_provider,
        pi_model=pi_model,
        pi_data_dir=pi_data_dir,
        pi_session_root=Path(pi_session_root).expanduser().resolve(),
        pi_session_ttl_seconds=pi_session_ttl_seconds,
        bot_queue_maxsize=bot_queue_maxsize,
        bot_heartbeat_interval_seconds=bot_heartbeat_interval_seconds,
        log_level=log_level,
    )


def _add_local_pi_sdk_to_path(configured: str | None) -> None:
    candidates = [configured] if configured else []
    candidates.extend(["~/Developer/pi_sdk", "~/Developer/pi-py"])

    for candidate in candidates:
        if not candidate:
            continue
        expanded = Path(candidate).expanduser().resolve()
        if not expanded.exists():
            continue
        path_str = str(expanded)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        break


def _load_pi_sdk() -> tuple[Any, type[Exception]]:
    global _PI_RPC_CLIENT_CLASS, _PI_RPC_ERROR_CLASS
    if _PI_RPC_CLIENT_CLASS is not None:
        return _PI_RPC_CLIENT_CLASS, _PI_RPC_ERROR_CLASS

    try:
        from pi_sdk import PiRPCClient, PiRPCError
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to import pi_sdk. "
            "Check [pi].sdk_path or install pi_sdk in the active Python environment."
        ) from exc

    _PI_RPC_CLIENT_CLASS = PiRPCClient
    _PI_RPC_ERROR_CLASS = PiRPCError
    return _PI_RPC_CLIENT_CLASS, _PI_RPC_ERROR_CLASS


@dataclass(slots=True)
class QueuedMessage:
    message: discord.Message
    content: str


@dataclass(slots=True)
class PiClientState:
    client: Any
    created_at_monotonic: float


class PersonalAssistantBot(discord.Client):
    def __init__(self, config: BotConfig) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self.message_queue: asyncio.Queue[QueuedMessage] = asyncio.Queue(maxsize=config.bot_queue_maxsize)
        self._queue_worker_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._pi_clients: dict[str, PiClientState] = {}
        self._pi_lock = asyncio.Lock()

        self._bot_owner_user_id = config.discord_bot_owner_user_id
        self._pi_executable = config.pi_executable
        self._pi_provider = config.pi_provider
        self._pi_model = config.pi_model
        self._pi_data_dir = config.pi_data_dir
        self._pi_session_root = config.pi_session_root
        self._pi_session_ttl_seconds = config.pi_session_ttl_seconds
        self._heartbeat_interval_seconds = config.bot_heartbeat_interval_seconds

    async def on_ready(self) -> None:
        LOGGER.info("Discord bot connected as %s (id=%s)", self.user, self.user.id if self.user else "unknown")
        if self._queue_worker_task is None or self._queue_worker_task.done():
            self._queue_worker_task = asyncio.create_task(self._queue_worker(), name="discord-message-queue-worker")
        if (
            self._heartbeat_interval_seconds > 0
            and (self._heartbeat_task is None or self._heartbeat_task.done())
        ):
            self._heartbeat_task = asyncio.create_task(self._heartbeat_worker(), name="heartbeat-worker")
            LOGGER.info("Heartbeat worker started with interval=%ss", self._heartbeat_interval_seconds)

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or message.is_system():
            return

        text = message.content.strip()
        if not text:
            return

        queued = QueuedMessage(message=message, content=text)
        await self.message_queue.put(queued)
        LOGGER.info(
            "Queued message %s from user %s in channel %s (queue=%s)",
            message.id,
            message.author.id,
            message.channel.id,
            self.message_queue.qsize(),
        )

    async def close(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._queue_worker_task is not None:
            self._queue_worker_task.cancel()
            try:
                await self._queue_worker_task
            except asyncio.CancelledError:
                pass
            self._queue_worker_task = None

        await asyncio.to_thread(self._close_pi_clients)
        await super().close()

    async def _queue_worker(self) -> None:
        while True:
            queued = await self.message_queue.get()
            try:
                await self._process_queued_message(queued)
            except Exception:  # noqa: BLE001
                LOGGER.exception("Unexpected error while handling queued message %s", queued.message.id)
                try:
                    await queued.message.channel.send(
                        "I hit an unexpected error while processing that message. Please try again."
                    )
                except discord.DiscordException:
                    LOGGER.exception("Failed to send unexpected-error message for queued message %s", queued.message.id)
            finally:
                self.message_queue.task_done()

    async def _process_queued_message(self, queued: QueuedMessage) -> None:
        message = queued.message
        conversation_key = self._conversation_key(message)
        prompt = self._prompt_from_message(queued)
        force_session = message.guild is None and message.author.id == self._bot_owner_user_id
        _, pi_error_type = _load_pi_sdk()

        LOGGER.info("Processing queued message %s for %s", message.id, conversation_key)
        try:
            async with message.channel.typing():
                response_text = await self._run_pi_prompt_async(conversation_key, prompt, force_session=force_session)
        except pi_error_type as exc:
            LOGGER.exception("pi_sdk error for message %s: %s", message.id, exc)
            await message.channel.send(f"pi error: {exc}")
            return

        if not response_text.strip():
            response_text = "I did not get a response from pi for that message."

        for chunk in self._split_for_discord(response_text):
            await message.channel.send(chunk)

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

    async def _run_heartbeat_once(self) -> None:
        _, pi_error_type = _load_pi_sdk()
        heartbeat_key = f"heartbeat-owner-{self._bot_owner_user_id}"
        heartbeat_prompt = (
            "This is a heartbeat session. "
            "You are running in an ephemeral context for a Discord bot heartbeat check. "
            "Reply with a short heartbeat status update."
        )
        try:
            heartbeat_response = await self._run_pi_prompt_async(heartbeat_key, heartbeat_prompt, force_ephemeral=True)
        except pi_error_type as exc:
            LOGGER.exception("pi_sdk error while running heartbeat prompt: %s", exc)
            return

        if not heartbeat_response.strip():
            heartbeat_response = "Heartbeat response was empty."

        owner_channel = await self._get_or_create_owner_dm_channel()
        owner_key = self._dm_conversation_key(self._bot_owner_user_id, owner_channel.id)
        owner_prompt = (
            "Heartbeat update from the bot's heartbeat session.\n"
            f"Heartbeat response: {heartbeat_response}"
        )
        try:
            owner_response = await self._run_pi_prompt_async(owner_key, owner_prompt, force_session=True)
        except pi_error_type as exc:
            LOGGER.exception("pi_sdk error while forwarding heartbeat to owner DM session: %s", exc)
            return

        if not owner_response.strip():
            owner_response = "I processed the heartbeat update, but the owner session returned no response."

        for chunk in self._split_for_discord(owner_response):
            await owner_channel.send(chunk)

        LOGGER.info("Forwarded heartbeat response into owner DM session %s", owner_key)

    async def _run_pi_prompt_async(
        self,
        conversation_key: str,
        prompt: str,
        *,
        force_ephemeral: bool = False,
        force_session: bool = False,
    ) -> str:
        if force_ephemeral and force_session:
            raise RuntimeError("force_ephemeral and force_session cannot both be true")

        async with self._pi_lock:
            if force_ephemeral:
                return await asyncio.to_thread(self._run_ephemeral_pi_prompt, conversation_key, prompt)
            if force_session:
                return await asyncio.to_thread(self._run_pi_prompt_forced_session, conversation_key, prompt)
            return await asyncio.to_thread(self._run_pi_prompt, conversation_key, prompt)

    async def _get_or_create_owner_dm_channel(self) -> discord.DMChannel:
        owner = self.get_user(self._bot_owner_user_id)
        if owner is None:
            owner = await self.fetch_user(self._bot_owner_user_id)

        dm_channel = owner.dm_channel
        if dm_channel is None:
            dm_channel = await owner.create_dm()
        return dm_channel

    def _run_ephemeral_pi_prompt(self, conversation_key: str, prompt: str) -> str:
        client = self._create_pi_client(conversation_key, force_no_session=True)
        try:
            response_text = "".join(client.stream_text(prompt)).strip()
            LOGGER.info("Returned ephemeral pi response for %s", conversation_key)
            return response_text
        finally:
            self._close_pi_client(conversation_key, client)

    def _run_pi_prompt(self, conversation_key: str, prompt: str) -> str:
        if self._pi_session_ttl_seconds == 0:
            return self._run_ephemeral_pi_prompt(conversation_key, prompt)

        client = self._get_pi_client(conversation_key)
        response_text = "".join(client.stream_text(prompt)).strip()
        LOGGER.info("Returned pi response for %s", conversation_key)
        return response_text

    def _run_pi_prompt_forced_session(self, conversation_key: str, prompt: str) -> str:
        client = self._get_pi_client(conversation_key, force_session=True)
        response_text = "".join(client.stream_text(prompt)).strip()
        LOGGER.info("Returned pi response for %s", conversation_key)
        return response_text

    def _get_pi_client(self, conversation_key: str, *, force_session: bool = False) -> Any:
        if self._pi_session_ttl_seconds == 0 and not force_session:
            raise RuntimeError("Session TTL of 0 requires per-message client creation")

        now = time.monotonic()
        pi_rpc_client_class, _ = _load_pi_sdk()
        existing = self._pi_clients.get(conversation_key)
        if existing is not None:
            if force_session and self._pi_session_ttl_seconds == 0:
                LOGGER.info("Resuming forced pi session for %s", conversation_key)
                return existing.client
            if now - existing.created_at_monotonic < self._pi_session_ttl_seconds:
                LOGGER.info("Resuming pi session for %s", conversation_key)
                return existing.client
            self._close_pi_client(conversation_key, existing.client)
            del self._pi_clients[conversation_key]

        client = self._create_pi_client(conversation_key, pi_rpc_client_class=pi_rpc_client_class)
        self._pi_clients[conversation_key] = PiClientState(client=client, created_at_monotonic=now)
        return client

    def _create_pi_client(
        self,
        conversation_key: str,
        *,
        force_no_session: bool = False,
        pi_rpc_client_class: Any | None = None,
    ) -> Any:
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
                    "Update your local pi_sdk installation."
                ) from exc
            raise
        client.start()
        return client

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
            self._close_pi_client(key, state.client)
        self._pi_clients.clear()

    @staticmethod
    def _close_pi_client(conversation_key: str, client: Any) -> None:
        try:
            client.close()
            LOGGER.info("Finished pi session for %s", conversation_key)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed closing pi client for %s", conversation_key)

    @staticmethod
    def _conversation_key(message: discord.Message) -> str:
        if message.guild is None:
            return PersonalAssistantBot._dm_conversation_key(message.author.id, message.channel.id)
        return f"guild-{message.guild.id}-channel-{message.channel.id}"

    @staticmethod
    def _dm_conversation_key(user_id: int, channel_id: int) -> str:
        return f"dm-user-{user_id}-channel-{channel_id}"

    @staticmethod
    def _prompt_from_message(queued: QueuedMessage) -> str:
        message = queued.message
        if message.guild is None:
            location = "direct message"
        else:
            location = f"guild={message.guild.name} channel={message.channel}"
        return (
            "You are a helpful personal assistant on Discord.\n"
            f"Author: {message.author.display_name} (id={message.author.id})\n"
            f"Location: {location}\n"
            f"Message: {queued.content}"
        )

    @staticmethod
    def _split_for_discord(text: str, *, limit: int = 2000) -> list[str]:
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


def main() -> None:
    config = load_config()
    level = getattr(logging, config.log_level, logging.INFO)
    discord.utils.setup_logging(level=level, root=True)

    _add_local_pi_sdk_to_path(config.pi_sdk_path)
    _load_pi_sdk()

    bot = PersonalAssistantBot(config)
    bot.run(config.discord_bot_token, log_handler=None)


if __name__ == "__main__":
    main()
