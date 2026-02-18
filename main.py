from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import discord

LOGGER = logging.getLogger("assistant.bot")


def _add_local_pi_sdk_to_path() -> None:
    configured = os.getenv("PI_SDK_PATH")
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


_add_local_pi_sdk_to_path()

from pi_sdk import PiRPCClient, PiRPCError  # noqa: E402


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class QueuedMessage:
    message: discord.Message
    content: str


class PersonalAssistantBot(discord.Client):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self.message_queue: asyncio.Queue[QueuedMessage] = asyncio.Queue(
            maxsize=int(os.getenv("BOT_QUEUE_MAXSIZE", "1000"))
        )
        self._queue_worker_task: asyncio.Task[None] | None = None
        self._pi_clients: dict[str, PiRPCClient] = {}

        self._pi_executable = os.getenv("PI_EXECUTABLE", "pi")
        self._pi_provider = os.getenv("PI_PROVIDER")
        self._pi_model = os.getenv("PI_MODEL")
        self._pi_no_session = _env_flag("PI_NO_SESSION", default=False)
        self._pi_session_root = Path(os.getenv("PI_SESSION_ROOT", ".pi_sessions")).resolve()

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

        LOGGER.info("Processing queued message %s for %s", message.id, conversation_key)
        try:
            async with message.channel.typing():
                response_text = await asyncio.to_thread(self._run_pi_prompt, conversation_key, prompt)
        except PiRPCError as exc:
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

    def _get_pi_client(self, conversation_key: str) -> PiRPCClient:
        existing = self._pi_clients.get(conversation_key)
        if existing is not None:
            return existing

        session_dir = self._pi_session_root / conversation_key
        session_dir.mkdir(parents=True, exist_ok=True)
        client = PiRPCClient(
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
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    discord.utils.setup_logging(level=level, root=True)

    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN is required")

    bot = PersonalAssistantBot()
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
