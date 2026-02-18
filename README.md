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
- Local `pi_sdk` source directory (default lookup order):
  - `[pi].sdk_path` in `bot.config.toml` (if set)
  - `~/Developer/pi_sdk`
  - `~/Developer/pi-py` (fallback)
- `pi` CLI available on `PATH` (or set `[pi].executable` in config)

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

## Configuration

Create a local config file from the example:

```bash
cp bot.config.toml.example bot.config.toml
```

Set your Discord token in `[discord].bot_token`. Runtime settings are read from this file.

## Run

```bash
python3 main.py
```

## Notes

- Enable the Message Content Intent for your bot in the Discord Developer Portal.
- The queue worker processes messages sequentially so responses are generated in order.
