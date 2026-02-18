# Telegram Personal Assistant Bot

Telegram bot using `python-telegram-bot` with a bounded concurrent queue worker pool and `pi_sdk` backend.

## Behavior

- Accepts messages only from the bot owner and IDs in `[telegram].trusted_user_ids`.
- In group chats, the message must mention the bot (or reply to one of its messages).
- In DMs, mention is not required.
- Pushes each message into an in-memory queue.
- Multiple workers process queued messages concurrently.
- Per-conversation locks preserve message ordering within each DM/chat.
- Streams response deltas into a progress message before posting the final response.

## Requirements

- Python 3.14+
- `bot.config.toml` with your Telegram bot token
- `pi` CLI available on `PATH` (or set `[pi].executable` in config)

## Setup

```bash
uv sync
```

## Configuration

Create a local config file from the example:

```bash
cp bot.config.toml.example bot.config.toml
```

Set your Telegram token in `[telegram].bot_token`, your owner user ID in
`[telegram].bot_owner_user_id`, and optional trusted users in
`[telegram].trusted_user_ids`. Runtime settings are read from this file.

Session behavior is configured under `[pi]`:

- `data_dir` sets the working directory (`cwd`) used for all `pi` sessions.
- When `data_dir` is set, the bot also sets `PI_CODING_AGENT_DIR` to that same path so PI uses isolated settings/extensions/auth instead of your global `~/.pi/agent`.
- `session_root` stores conversation state on disk.
- `session_ttl_seconds` controls session lifetime.
- `session_ttl_seconds = 0` starts a brand-new session for every message (except forced session paths).
- `session_ttl_seconds > 0` keeps one active session per DM/chat and expires idle sessions after the TTL.
- `call_timeout_seconds` enforces a hard timeout around PI calls (`0` disables timeout).
- `session_sweeper_interval_seconds` controls periodic cleanup of stale in-memory clients and stale on-disk session directories (`0` disables sweeper).

Bot runtime behavior is configured under `[bot]`:

- `queue_maxsize` sets max pending inbound messages.
- `worker_concurrency` controls queue worker pool size.
- `heartbeat_interval_seconds` controls heartbeat cadence (`0` disables heartbeat).
- `metrics_interval_seconds` controls periodic health telemetry logs (`0` disables this worker).
- `stream_edit_interval_ms` throttles how frequently progress message edits are sent.
- `latency_window_size` sets the sample window used for per-conversation latency percentiles.

Dashboard behavior is configured under `[dashboard]`:

- `enabled` toggles the local dashboard server.
- `host` must stay local-only (`127.0.0.1`, `localhost`, or `::1`) when enabled.
- `port` sets the HTTP listener.
- `base_path` sets where the dashboard and API are mounted (for example `/dashboard`).
- `access_token` is required and used as a Bearer token for API and websocket endpoints.
- `ws_push_interval_ms` controls periodic websocket overview updates.

## Run

```bash
uv run python3 main.py
```

When dashboard is enabled, API endpoints are available under:

- `/dashboard/api/overview`
- `/dashboard/api/queue`
- `/dashboard/api/sessions`
- `/dashboard/api/config`
- `/dashboard/ws/events` (websocket)

## Dashboard Frontend

The dashboard frontend is a Solid.js + Vite app in `web/`.

```bash
cd web
npm install
npm run build
```

After build, the bot serves the static dashboard at the configured `[dashboard].base_path`.
At runtime, enter your `[dashboard].access_token` in the dashboard UI to connect.

## Notes

- In group chats, mention the bot username (for example `@your_bot`) or reply to a bot message to trigger processing.
- Health logs report queue depth, oldest queue age, active session count, timeout count, and per-conversation latency percentiles.
