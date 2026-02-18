from __future__ import annotations

import logging

from telegram.ext import ApplicationBuilder, MessageHandler, filters

from bot_runtime import PersonalAssistantBot
from configuration import configure_logging, configure_pi_environment, ensure_event_loop, load_config
from pi_runtime import load_pi_sdk

LOGGER = logging.getLogger("assistant.bot")


def main() -> None:
    config = load_config()
    configure_logging(config.log_level)
    provider = config.pi_provider or "<default>"
    model = config.pi_model or "<default>"
    LOGGER.info("Starting bot with pi provider=%s model=%s", provider, model)
    configure_pi_environment(config)

    load_pi_sdk()

    bot = PersonalAssistantBot(config)
    application = (
        ApplicationBuilder()
        .token(config.telegram_bot_token)
        .post_init(bot.on_ready)
        .post_shutdown(bot.shutdown)
        .build()
    )
    application.add_handler(MessageHandler(filters.TEXT | filters.CAPTION, bot.on_message))
    application.add_error_handler(bot.on_error)
    ensure_event_loop()
    application.run_polling()


if __name__ == "__main__":
    main()
