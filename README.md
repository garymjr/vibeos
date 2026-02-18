# Discord Personal Assistant Bot

Discord bot using `discord.py` with a message queue and `pi_sdk` backend.

## Behavior

- Listens for incoming Discord messages (guild channels and DMs).
- Pushes each message into an in-memory queue.
- A queue worker reads messages one at a time and sends prompts to `pi` via `pi_sdk`.
- Sends the `pi` response back to the same channel or DM where the message originated.

## Requirements

- Python 3
- Discord bot token
- Local `pi_sdk` source directory (default lookup order):
  - `PI_SDK_PATH` env var (if set)
  - `~/Developer/pi_sdk`
  - `~/Developer/pi-py` (fallback)
- `pi` CLI available on `PATH` (or set `PI_EXECUTABLE`)

## Setup

```bash
python3 -m pip install -e .
```

If needed, install local SDK in editable mode too:

```bash
python3 -m pip install -e ~/Developer/pi_sdk
```

Or, for the fallback path used in this workspace:

```bash
python3 -m pip install -e ~/Developer/pi-py
```

## Environment Variables

- `DISCORD_BOT_TOKEN` (required)
- `PI_SDK_PATH` (optional, local sdk path)
- `PI_EXECUTABLE` (default: `pi`)
- `PI_PROVIDER` (optional)
- `PI_MODEL` (optional)
- `PI_NO_SESSION` (optional, default: `false`)
- `PI_SESSION_ROOT` (optional, default: `.pi_sessions`)
- `BOT_QUEUE_MAXSIZE` (optional, default: `1000`)
- `LOG_LEVEL` (optional, default: `INFO`)

## Run

```bash
DISCORD_BOT_TOKEN=... python3 main.py
```

## Notes

- Enable the Message Content Intent for your bot in the Discord Developer Portal.
- The queue worker processes messages sequentially so responses are generated in order.
