from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Awaitable, Callable

from aiohttp import WSMsgType, web

from bot_runtime import PersonalAssistantBot
from configuration import BotConfig

LOGGER = logging.getLogger("assistant.dashboard")


class DashboardServer:
    def __init__(self, config: BotConfig, bot: PersonalAssistantBot) -> None:
        self._enabled = config.dashboard_enabled
        self._host = config.dashboard_host
        self._port = config.dashboard_port
        self._base_path = config.dashboard_base_path
        self._access_token = config.dashboard_access_token or ""
        self._ws_push_interval_seconds = config.dashboard_ws_push_interval_ms / 1000
        self._bot = bot
        self._frontend_dist = Path(__file__).resolve().parent / "web" / "dist"

        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._event_task: asyncio.Task[None] | None = None
        self._websocket_clients: set[web.WebSocketResponse] = set()

    async def start(self) -> None:
        if not self._enabled or self._runner is not None:
            return

        app = web.Application(middlewares=[self._auth_middleware])
        self._register_routes(app)

        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, host=self._host, port=self._port)
        await site.start()

        self._app = app
        self._runner = runner
        self._site = site
        self._event_task = asyncio.create_task(self._event_worker(), name="dashboard-event-worker")

        LOGGER.info(
            "Dashboard server listening on http://%s:%s%s",
            self._host,
            self._port,
            self._base_path,
        )

    async def stop(self) -> None:
        if self._runner is None:
            return

        if self._event_task is not None:
            self._event_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._event_task
            self._event_task = None

        for websocket in list(self._websocket_clients):
            with contextlib.suppress(Exception):
                await websocket.close()
        self._websocket_clients.clear()

        await self._runner.cleanup()
        self._site = None
        self._runner = None
        self._app = None
        LOGGER.info("Dashboard server stopped")

    def _register_routes(self, app: web.Application) -> None:
        base_path = self._base_path
        api_prefix = f"{base_path}/api"

        app.router.add_get(f"{api_prefix}/overview", self._json_handler(self._bot.dashboard_overview_snapshot))
        app.router.add_get(f"{api_prefix}/queue", self._json_handler(self._bot.dashboard_queue_snapshot))
        app.router.add_get(f"{api_prefix}/sessions", self._json_handler(self._bot.dashboard_session_snapshot))
        app.router.add_get(f"{api_prefix}/config", self._handle_config)
        app.router.add_get(f"{base_path}/ws/events", self._handle_events_ws)
        app.router.add_get(base_path, self._handle_dashboard)
        app.router.add_get(f"{base_path}/", self._handle_dashboard)
        app.router.add_get(f"{base_path}/{{tail:.*}}", self._handle_static_or_dashboard)

    def _json_handler(
        self,
        snapshot: Callable[[], dict[str, object]],
    ) -> Callable[[web.Request], Awaitable[web.Response]]:
        async def _handler(_request: web.Request) -> web.Response:
            return web.json_response(snapshot())

        return _handler

    @web.middleware
    async def _auth_middleware(
        self,
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        path = request.path
        api_prefix = f"{self._base_path}/api"
        ws_prefix = f"{self._base_path}/ws"
        if path.startswith(api_prefix) or path.startswith(ws_prefix):
            if not self._is_authorized(request):
                return web.json_response({"error": "unauthorized"}, status=401)
        return await handler(request)

    def _is_authorized(self, request: web.Request) -> bool:
        expected = f"Bearer {self._access_token}"
        header = request.headers.get("Authorization", "")
        if header == expected:
            return True
        # Browsers do not allow arbitrary WS headers, so WS auth may use token query param.
        return request.query.get("token", "") == self._access_token

    async def _handle_config(self, _request: web.Request) -> web.Response:
        config = {
            "base_path": self._base_path,
            "api_paths": {
                "overview": f"{self._base_path}/api/overview",
                "queue": f"{self._base_path}/api/queue",
                "sessions": f"{self._base_path}/api/sessions",
                "config": f"{self._base_path}/api/config",
            },
            "ws_path": f"{self._base_path}/ws/events",
            "ws_push_interval_ms": int(self._ws_push_interval_seconds * 1000),
        }
        return web.json_response(config)

    async def _handle_events_ws(self, request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse(heartbeat=30)
        await websocket.prepare(request)
        self._websocket_clients.add(websocket)
        await websocket.send_json({"type": "overview", "payload": self._bot.dashboard_overview_snapshot()})
        try:
            async for message in websocket:
                if message.type == WSMsgType.ERROR:
                    LOGGER.debug("Dashboard websocket closed with error: %s", websocket.exception())
        finally:
            self._websocket_clients.discard(websocket)
        return websocket

    async def _event_worker(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._ws_push_interval_seconds)
                if not self._websocket_clients:
                    continue
                await self._broadcast_json(
                    {
                        "type": "overview",
                        "payload": self._bot.dashboard_overview_snapshot(),
                    }
                )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            LOGGER.exception("Dashboard event worker failed")

    async def _broadcast_json(self, payload: dict[str, object]) -> None:
        stale_clients: list[web.WebSocketResponse] = []
        for websocket in self._websocket_clients:
            if websocket.closed:
                stale_clients.append(websocket)
                continue
            try:
                await websocket.send_json(payload)
            except Exception:  # noqa: BLE001
                stale_clients.append(websocket)
        for websocket in stale_clients:
            self._websocket_clients.discard(websocket)

    async def _handle_dashboard(self, _request: web.Request) -> web.StreamResponse:
        index_file = self._frontend_dist / "index.html"
        if not index_file.is_file():
            return self._frontend_not_built_response()
        return web.FileResponse(index_file)

    async def _handle_static_or_dashboard(self, request: web.Request) -> web.StreamResponse:
        tail = request.match_info.get("tail", "")
        if not tail:
            return await self._handle_dashboard(request)

        dist_root = self._frontend_dist.resolve()
        candidate = (dist_root / tail).resolve()
        try:
            candidate.relative_to(dist_root)
        except ValueError:
            raise web.HTTPNotFound from None

        if candidate.is_file():
            return web.FileResponse(candidate)
        return await self._handle_dashboard(request)

    def _frontend_not_built_response(self) -> web.Response:
        return web.Response(
            status=503,
            text=(
                "Dashboard frontend is not built yet.\n"
                "Run: cd web && npm install && npm run build\n"
            ),
            content_type="text/plain",
        )
