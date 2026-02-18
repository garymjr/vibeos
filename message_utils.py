from __future__ import annotations

from dataclasses import dataclass

from telegram import Message, MessageEntity

TELEGRAM_MESSAGE_LIMIT = 4096


@dataclass(slots=True)
class QueuedMessage:
    message: Message
    content: str
    enqueued_at_monotonic: float


def extract_mentions(text: str | None, entities: tuple[MessageEntity, ...] | None) -> set[str]:
    if not text or not entities:
        return set()

    mentions: set[str] = set()
    for entity in entities:
        if entity.type != MessageEntity.MENTION:
            continue
        start = entity.offset
        end = start + entity.length
        if start < 0 or end > len(text):
            continue
        token = text[start:end].strip().lower()
        if token.startswith("@"):
            mentions.add(token)
    return mentions


def is_bot_mentioned(message: Message, *, bot_username: str | None, bot_user_id: int | None) -> bool:
    if bot_username is not None:
        mention = f"@{bot_username}"
        mentions = extract_mentions(message.text, message.entities)
        mentions.update(extract_mentions(message.caption, message.caption_entities))
        if mention in mentions:
            return True

    if bot_user_id is not None:
        replied = message.reply_to_message
        if replied is not None and replied.from_user is not None and replied.from_user.id == bot_user_id:
            return True

    return False


def message_key(message: Message) -> tuple[int, int]:
    return (message.chat_id, message.message_id)


def dm_conversation_key(user_id: int, channel_id: int) -> str:
    return f"dm-user-{user_id}-channel-{channel_id}"


def conversation_key(message: Message) -> str:
    if message.chat.type == "private":
        sender_id = message.from_user.id if message.from_user is not None else 0
        return dm_conversation_key(sender_id, message.chat_id)
    return f"chat-{message.chat_id}"


def prompt_from_message(queued: QueuedMessage) -> str:
    message = queued.message
    if message.chat.type == "private":
        location = "direct message"
    else:
        location = f"chat_type={message.chat.type} chat={message.chat.title or message.chat_id}"

    sender = message.from_user
    sender_name = sender.full_name if sender is not None else "unknown"
    sender_id = sender.id if sender is not None else 0
    return (
        "You are a helpful personal assistant on Telegram.\n"
        f"Author: {sender_name} (id={sender_id})\n"
        f"Location: {location}\n"
        f"Message: {queued.content}"
    )


def stream_preview(text: str, *, limit: int = TELEGRAM_MESSAGE_LIMIT) -> str:
    if not text:
        return "Thinking..."
    if len(text) <= limit:
        return text

    header = f"[streaming, {len(text)} chars]\n"
    tail_size = max(1, limit - len(header))
    return header + text[-tail_size:]


def split_for_telegram(text: str, *, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
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
