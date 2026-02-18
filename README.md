# Discord Personal Assistant Bot

Discord bot using `discord.py` with a message queue and `pi_sdk` backend.

## Behavior

- Listens for incoming Discord messages (guild channels and DMs).
- Pushes each message into an in-memory queue.
- A queue worker reads messages one at a time and sends prompts to `pi` via `pi_sdk`.
- Sends the `pi` response back to the same channel or DM where the message originated.

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

Set your Discord token in `[discord].bot_token` and your owner user ID in
`[discord].bot_owner_user_id`. Runtime settings are read from this file.

Session behavior is configured under `[pi]`:

- `data_dir` sets the working directory (`cwd`) used for all `pi` sessions.
- When `data_dir` is set, the bot also sets `PI_CODING_AGENT_DIR` to that same path so PI uses isolated settings/extensions/auth instead of your global `~/.pi/agent`.
- `session_root` stores conversation state on disk.
- `session_ttl_seconds` controls session lifetime.
- `session_ttl_seconds = 0` starts a brand-new session for every message.
- `session_ttl_seconds > 0` keeps one active session per DM/channel and rotates it after the TTL expires.

Heartbeat behavior is configured under `[bot]`:

- `heartbeat_interval_seconds` controls heartbeat cadence.
- `heartbeat_interval_seconds = 0` disables heartbeat.
- `heartbeat_interval_seconds > 0` starts a periodic ephemeral heartbeat prompt, then forwards the heartbeat response into the owner DM PI session.

## Run

```bash
uv run python3 main.py
```

## Notes

- Enable the Message Content Intent for your bot in the Discord Developer Portal.
- The queue worker processes messages sequentially so responses are generated in order.
