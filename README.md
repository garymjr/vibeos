# Discord Personal Assistant Bot

Discord bot using `discord.py` with a bounded concurrent queue worker pool and `pi_sdk` backend.

## Behavior

- Accepts messages only from the bot owner and IDs in `[discord].trusted_user_ids`.
- In guild channels, the message must mention the bot.
- In DMs, mention is not required.
- Pushes each message into an in-memory queue.
- Multiple workers process queued messages concurrently.
- Per-conversation locks preserve message ordering within each DM/channel.
- Streams response deltas into an in-channel progress message before posting the final response.

## Requirements

- Python 3
- `bot.config.toml` with your Discord bot token
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

Set your Discord token in `[discord].bot_token`, your owner user ID in
`[discord].bot_owner_user_id`, and optional trusted users in
`[discord].trusted_user_ids`. Runtime settings are read from this file.

Session behavior is configured under `[pi]`:

- `data_dir` sets the working directory (`cwd`) used for all `pi` sessions.
- When `data_dir` is set, the bot also sets `PI_CODING_AGENT_DIR` to that same path so PI uses isolated settings/extensions/auth instead of your global `~/.pi/agent`.
- `session_root` stores conversation state on disk.
- `session_ttl_seconds` controls session lifetime.
- `session_ttl_seconds = 0` starts a brand-new session for every message (except forced session paths).
- `session_ttl_seconds > 0` keeps one active session per DM/channel and expires idle sessions after the TTL.
- `call_timeout_seconds` enforces a hard timeout around PI calls (`0` disables timeout).
- `session_sweeper_interval_seconds` controls periodic cleanup of stale in-memory clients and stale on-disk session directories (`0` disables sweeper).

Bot runtime behavior is configured under `[bot]`:

- `queue_maxsize` sets max pending inbound messages.
- `worker_concurrency` controls queue worker pool size.
- `heartbeat_interval_seconds` controls heartbeat cadence (`0` disables heartbeat).
- `metrics_interval_seconds` controls periodic health telemetry logs (`0` disables this worker).
- `stream_edit_interval_ms` throttles how frequently progress message edits are sent.
- `latency_window_size` sets the sample window used for per-conversation latency percentiles.

## Run

```bash
uv run python3 main.py
```

## Notes

- Enable the Message Content Intent for your bot in the Discord Developer Portal.
- Health logs report queue depth, oldest queue age, active session count, timeout count, and per-conversation latency percentiles.
