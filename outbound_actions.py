from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SUPPORTED_CHANNEL = "telegram"
SUPPORTED_SEND_ACTIONS = {"sendmessage", "send_message", "send"}
SUPPORTED_SEND_TOOL_NAMES = {"message", "send", "message_send", "telegram_send"}


class OutboundActionParseError(ValueError):
    """Raised when a send action is recognized but invalid for this runtime."""


@dataclass(slots=True)
class TelegramSendAction:
    to: int | str
    text: str
    reply_to_message_id: int | None = None
    message_thread_id: int | None = None


def extract_openclaw_send_action(
    *,
    tool_name: str,
    args: dict[str, Any],
    default_chat_id: int,
) -> TelegramSendAction | None:
    """Parse an OpenClaw-style send action from a tool execution payload.

    Supported patterns:
    - tool name `message` (or `send`) with args.action == "sendMessage"
    - tool name `send`/`message_send`/`telegram_send` with direct `message`/`text`
    - optional `replyTo` and `threadId` for Telegram reply/topic routing
    """
    normalized_tool_name = tool_name.strip().lower()
    action = _read_optional_string(args, "action")
    if action is not None:
        if action.lower() not in SUPPORTED_SEND_ACTIONS:
            return None
    elif normalized_tool_name not in SUPPORTED_SEND_TOOL_NAMES:
        return None

    channel = _read_optional_string(args, "channel")
    if channel and channel.lower() != SUPPORTED_CHANNEL:
        raise OutboundActionParseError(
            f"Unsupported outbound channel '{channel}' (this runtime only supports {SUPPORTED_CHANNEL})."
        )

    raw_target = (
        args.get("to")
        or args.get("chat_id")
        or args.get("chatId")
        or args.get("target")
    )
    target = _normalize_telegram_target(raw_target) if raw_target is not None else default_chat_id

    text = (
        _read_optional_string(args, "message")
        or _read_optional_string(args, "text")
        or _read_optional_string(args, "content")
        or _read_optional_string(args, "body")
        or _read_optional_string(args, "caption")
    )
    if text is None:
        raise OutboundActionParseError("Outbound send action is missing message/text content.")

    reply_to_message_id = _read_optional_int(args, ("replyTo", "reply_to", "replyToId", "reply_to_id"))
    message_thread_id = _read_optional_int(args, ("threadId", "thread_id"))

    return TelegramSendAction(
        to=target,
        text=text,
        reply_to_message_id=reply_to_message_id,
        message_thread_id=message_thread_id,
    )


def _read_optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise OutboundActionParseError(f"Expected '{key}' to be a string.")
    stripped = value.strip()
    return stripped or None


def _read_optional_int(data: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            raise OutboundActionParseError(f"Expected '{key}' to be an integer.")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                continue
            if stripped.lstrip("-").isdigit():
                return int(stripped)
        raise OutboundActionParseError(f"Expected '{key}' to be an integer.")
    return None


def _normalize_telegram_target(value: Any) -> int | str:
    if isinstance(value, bool):
        raise OutboundActionParseError("Telegram target cannot be a boolean.")
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise OutboundActionParseError("Telegram target must be an integer chat id or a string.")
    stripped = value.strip()
    if not stripped:
        raise OutboundActionParseError("Telegram target cannot be empty.")
    if stripped.lstrip("-").isdigit():
        return int(stripped)
    return stripped
