from __future__ import annotations

import asyncio
import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from configuration import BotConfig

LOGGER = logging.getLogger("assistant.bot")

_PI_RPC_CLIENT_CLASS: Any | None = None
_PI_RPC_ERROR_CLASS: type[Exception] = Exception

DeltaHandler = Callable[[str], Awaitable[None]]


def load_pi_sdk() -> tuple[Any, type[Exception]]:
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

@dataclass(slots=True)
class PiClientState:
    client: Any
    created_at_monotonic: float
    last_used_monotonic: float
    session_dir: Path


class PiRuntime:
    def __init__(self, config: BotConfig) -> None:
        self._pi_clients: dict[str, PiClientState] = {}
        self._timeout_count = 0

        self._pi_executable = config.pi_executable
        self._pi_provider = config.pi_provider
        self._pi_model = config.pi_model
        self._pi_data_dir = config.pi_data_dir
        self._pi_session_root = config.pi_session_root
        self._pi_session_ttl_seconds = config.pi_session_ttl_seconds
        self._pi_call_timeout_seconds = config.pi_call_timeout_seconds
        self._session_cleanup_ttl_seconds = (
            config.pi_session_ttl_seconds if config.pi_session_ttl_seconds > 0 else 1800
        )

        self._pi_session_root.mkdir(parents=True, exist_ok=True)

    @property
    def timeout_count(self) -> int:
        return self._timeout_count

    @property
    def active_session_count(self) -> int:
        return len(self._pi_clients)

    def session_snapshots(self) -> list[dict[str, str | float]]:
        now = time.monotonic()
        sessions: list[dict[str, str | float]] = []
        for conversation_key, state in self._pi_clients.items():
            sessions.append(
                {
                    "conversation_key": conversation_key,
                    "session_dir": str(state.session_dir),
                    "created_age_seconds": round(max(0.0, now - state.created_at_monotonic), 2),
                    "idle_age_seconds": round(max(0.0, now - state.last_used_monotonic), 2),
                }
            )
        sessions.sort(key=lambda item: item["idle_age_seconds"], reverse=True)
        return sessions

    async def run_prompt(
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

        run_coro = self._run_prompt_once(
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
            await self.evict_client(conversation_key, reason="timeout")
            raise

    async def sweep_stale_sessions(self, *, is_conversation_locked: Callable[[str], bool]) -> tuple[int, int]:
        now = time.monotonic()
        expired_keys: list[str] = []

        for conversation_key, state in list(self._pi_clients.items()):
            if is_conversation_locked(conversation_key):
                continue
            if now - state.last_used_monotonic >= self._session_cleanup_ttl_seconds:
                expired_keys.append(conversation_key)

        for conversation_key in expired_keys:
            await self.evict_client(conversation_key, reason="sweeper-idle-expired")

        active_session_dirs = {state.session_dir.resolve() for state in self._pi_clients.values()}
        deleted_dirs = await asyncio.to_thread(
            self._prune_stale_session_directories,
            active_session_dirs,
            self._session_cleanup_ttl_seconds,
        )
        return len(expired_keys), deleted_dirs

    async def close_all_clients(self) -> None:
        await asyncio.to_thread(self._close_pi_clients)

    async def evict_client(self, conversation_key: str, *, reason: str) -> None:
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
        LOGGER.debug("Evicted pi client for %s reason=%s", conversation_key, reason)

    async def _run_prompt_once(
        self,
        conversation_key: str,
        prompt: str,
        *,
        force_ephemeral: bool,
        force_session: bool,
        on_delta: DeltaHandler | None,
    ) -> str:
        if force_ephemeral or (self._pi_session_ttl_seconds == 0 and not force_session):
            client, session_dir = await asyncio.to_thread(
                self._create_pi_client,
                conversation_key,
                force_no_session=True,
            )
            try:
                response_text = await self._stream_text_from_client_async(
                    conversation_key,
                    client,
                    prompt,
                    on_delta=on_delta,
                )
                LOGGER.debug("Returned ephemeral pi response for %s", conversation_key)
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

        LOGGER.debug("Returned pi response for %s", conversation_key)
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
        stop_signal = threading.Event()

        def emit(event: str, payload: object) -> None:
            try:
                loop.call_soon_threadsafe(events.put_nowait, (event, payload))
            except RuntimeError:
                return

        def stream_worker() -> None:
            try:
                if self._pi_call_timeout_seconds > 0:
                    client.prompt(prompt, timeout=self._pi_call_timeout_seconds)
                else:
                    client.prompt(prompt)

                while not stop_signal.is_set():
                    try:
                        event = client.next_event(timeout=1.0)
                    except Exception as exc:  # noqa: BLE001
                        if exc.__class__.__name__ == "PiRPCTimeoutError":
                            continue
                        raise

                    if not isinstance(event, dict):
                        continue
                    if event.get("type") == "agent_end":
                        break
                    if event.get("type") != "message_update":
                        continue

                    assistant_event = event.get("assistantMessageEvent")
                    if not isinstance(assistant_event, dict):
                        continue
                    if assistant_event.get("type") != "text_delta":
                        continue

                    delta = assistant_event.get("delta")
                    if isinstance(delta, str):
                        emit("delta", delta)
            except Exception as exc:  # noqa: BLE001
                emit("error", exc)
            else:
                emit("done", None)

        threading.Thread(
            target=stream_worker,
            name=f"pi-stream-{conversation_key}",
            daemon=True,
        ).start()

        chunks: list[str] = []
        try:
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
        finally:
            stop_signal.set()

        return "".join(chunks)

    async def _get_pi_client(self, conversation_key: str, *, force_session: bool = False) -> PiClientState:
        if self._pi_session_ttl_seconds == 0 and not force_session:
            raise RuntimeError("Session TTL of 0 requires per-message client creation")

        now = time.monotonic()
        existing = self._pi_clients.get(conversation_key)
        if existing is not None:
            if force_session and self._pi_session_ttl_seconds == 0:
                existing.last_used_monotonic = now
                LOGGER.debug("Resuming forced pi session for %s", conversation_key)
                return existing

            if self._pi_session_ttl_seconds > 0 and now - existing.last_used_monotonic < self._pi_session_ttl_seconds:
                existing.last_used_monotonic = now
                LOGGER.debug("Resuming pi session for %s", conversation_key)
                return existing

            await self.evict_client(conversation_key, reason="ttl-expired")

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
            pi_rpc_client_class, _ = load_pi_sdk()

        session_dir = self._pi_session_root / conversation_key / f"session-{time.time_ns()}"
        session_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.debug("Starting pi session for %s", conversation_key)
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

    @staticmethod
    def _close_pi_client(
        conversation_key: str,
        client: Any,
        session_dir: Path | None = None,
        delete_session_dir: bool = False,
    ) -> None:
        try:
            client.close()
            LOGGER.debug("Finished pi session for %s", conversation_key)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed closing pi client for %s", conversation_key)

        if delete_session_dir and session_dir is not None:
            try:
                shutil.rmtree(session_dir)
            except FileNotFoundError:
                return
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed deleting session directory %s for %s", session_dir, conversation_key)
