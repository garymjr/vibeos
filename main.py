from __future__ import annotations

import asyncio
import logging
import sys
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


def _get_bool(table: dict[str, object], key: str, *, section: str, default: bool) -> bool:
    value = table.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise RuntimeError(f"[{section}] {key} must be a boolean")
    return value


def _get_positive_int(table: dict[str, object], key: str, *, section: str, default: int) -> int:
    value = table.get(key)
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(f"[{section}] {key} must be a positive integer")
    return value


@dataclass(slots=True)
class BotConfig:
    discord_bot_token: str
    pi_sdk_path: str | None
    pi_executable: str
    pi_provider: str | None
    pi_model: str | None
    pi_no_session: bool
    pi_session_root: Path
    bot_queue_maxsize: int
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
    pi_sdk_path = _get_optional_string(pi_config, "sdk_path", section="pi")
    pi_executable = _get_optional_string(pi_config, "executable", section="pi") or "pi"
    pi_provider = _get_optional_string(pi_config, "provider", section="pi")
    pi_model = _get_optional_string(pi_config, "model", section="pi")
    pi_no_session = _get_bool(pi_config, "no_session", section="pi", default=False)
    pi_session_root = _get_optional_string(pi_config, "session_root", section="pi") or ".pi_sessions"
    bot_queue_maxsize = _get_positive_int(bot_config, "queue_maxsize", section="bot", default=1000)
    log_level = (_get_optional_string(bot_config, "log_level", section="bot") or "INFO").upper()

    return BotConfig(
        discord_bot_token=token,
        pi_sdk_path=pi_sdk_path,
        pi_executable=pi_executable,
        pi_provider=pi_provider,
        pi_model=pi_model,
        pi_no_session=pi_no_session,
        pi_session_root=Path(pi_session_root).expanduser().resolve(),
        bot_queue_maxsize=bot_queue_maxsize,
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


class PersonalAssistantBot(discord.Client):
    def __init__(self, config: BotConfig) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self.message_queue: asyncio.Queue[QueuedMessage] = asyncio.Queue(maxsize=config.bot_queue_maxsize)
        self._queue_worker_task: asyncio.Task[None] | None = None
        self._pi_clients: dict[str, Any] = {}

        self._pi_executable = config.pi_executable
        self._pi_provider = config.pi_provider
        self._pi_model = config.pi_model
        self._pi_no_session = config.pi_no_session
        self._pi_session_root = config.pi_session_root

    async def on_ready(self) -> None:
        LOGGER.info("Discord bot connected as %s (id=%s)", self.user, self.user.id if self.user else "unknown")
        if self._queue_worker_task is None or self._queue_worker_task.done():
            self._queue_worker_task = asyncio.create_task(self._queue_worker(), name="discord-message-queue-worker")

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
        _, pi_error_type = _load_pi_sdk()

        LOGGER.info("Processing queued message %s for %s", message.id, conversation_key)
        try:
            async with message.channel.typing():
                response_text = await asyncio.to_thread(self._run_pi_prompt, conversation_key, prompt)
        except pi_error_type as exc:
            LOGGER.exception("pi_sdk error for message %s: %s", message.id, exc)
            await message.channel.send(f"pi error: {exc}")
            return

        if not response_text.strip():
            response_text = "I did not get a response from pi for that message."

        for chunk in self._split_for_discord(response_text):
            await message.channel.send(chunk)

    def _run_pi_prompt(self, conversation_key: str, prompt: str) -> str:
        client = self._get_pi_client(conversation_key)
        return "".join(client.stream_text(prompt)).strip()

    def _get_pi_client(self, conversation_key: str) -> Any:
        pi_rpc_client_class, _ = _load_pi_sdk()
        existing = self._pi_clients.get(conversation_key)
        if existing is not None:
            return existing

        session_dir = self._pi_session_root / conversation_key
        session_dir.mkdir(parents=True, exist_ok=True)
        client = pi_rpc_client_class(
            executable=self._pi_executable,
            provider=self._pi_provider,
            model=self._pi_model,
            no_session=self._pi_no_session,
            session_dir=session_dir,
        )
        client.start()
        self._pi_clients[conversation_key] = client
        return client

    def _close_pi_clients(self) -> None:
        for key, client in list(self._pi_clients.items()):
            try:
                client.close()
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed closing pi client for %s", key)
        self._pi_clients.clear()

    @staticmethod
    def _conversation_key(message: discord.Message) -> str:
        if message.guild is None:
            return f"dm-{message.channel.id}"
        return f"guild-{message.guild.id}-channel-{message.channel.id}"

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
