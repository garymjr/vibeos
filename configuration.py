from __future__ import annotations

import asyncio
import logging
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger("assistant.bot")
DEFAULT_CONFIG_PATH = Path("bot.config.toml")
_VALID_LOG_LEVELS: tuple[str, ...] = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


class _ComponentNameFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        component = record.name
        if component.startswith("assistant."):
            component = component.removeprefix("assistant.")
        elif component.startswith("telegram.ext."):
            component = f"telegram.{component.removeprefix('telegram.ext.')}"
        elif component.startswith("telegram."):
            component = component.removeprefix("telegram.")

        record.component = component
        return True


@dataclass(slots=True)
class BotConfig:
    telegram_bot_token: str
    telegram_bot_owner_user_id: int
    telegram_trusted_user_ids: tuple[int, ...]
    pi_executable: str
    pi_provider: str | None
    pi_model: str | None
    pi_data_dir: Path | None
    pi_session_root: Path
    pi_session_ttl_seconds: int
    pi_call_timeout_seconds: int
    pi_session_sweeper_interval_seconds: int
    bot_queue_maxsize: int
    bot_worker_concurrency: int
    bot_heartbeat_interval_seconds: int
    bot_metrics_interval_seconds: int
    bot_stream_edit_interval_ms: int
    bot_latency_window_size: int
    dashboard_enabled: bool
    dashboard_host: str
    dashboard_port: int
    dashboard_base_path: str
    dashboard_access_token: str | None
    dashboard_ws_push_interval_ms: int
    log_level: str


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level)
    handler = logging.StreamHandler()
    handler.addFilter(_ComponentNameFilter())
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(component)-24s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger()
    for existing in root.handlers[:]:
        root.removeHandler(existing)
        existing.close()
    root.setLevel(level)
    root.addHandler(handler)

    for logger_name in ("httpx", "httpcore"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> BotConfig:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise RuntimeError(
            f"Config file not found at {resolved_path}. "
            "Create bot.config.toml from bot.config.toml.example."
        )

    with resolved_path.open("rb") as handle:
        config_data = tomllib.load(handle)

    if not isinstance(config_data, dict):
        raise RuntimeError("Config root must be a TOML table")

    telegram_config = _as_table(config_data.get("telegram"), name="telegram")
    if not telegram_config:
        raise RuntimeError("Missing required [telegram] section in bot.config.toml")

    pi_config = _as_table(config_data.get("pi"), name="pi")
    bot_config = _as_table(config_data.get("bot"), name="bot")
    dashboard_config = _as_table(config_data.get("dashboard"), name="dashboard")

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing required TELEGRAM_BOT_TOKEN environment variable")
    owner_user_id = _get_required_positive_int(telegram_config, "bot_owner_user_id", section="telegram")
    trusted_user_ids = _get_positive_int_list(telegram_config, "trusted_user_ids", section="telegram")

    pi_executable = _get_optional_string(pi_config, "executable", section="pi") or "pi"
    pi_provider = _get_optional_string(pi_config, "provider", section="pi")
    pi_model = _get_optional_string(pi_config, "model", section="pi")
    pi_data_dir_raw = _get_optional_string(pi_config, "data_dir", section="pi")
    pi_session_root = _get_optional_string(pi_config, "session_root", section="pi") or ".pi_sessions"
    pi_session_ttl_seconds = _get_non_negative_int(
        pi_config,
        "session_ttl_seconds",
        section="pi",
        default=0,
    )
    pi_call_timeout_seconds = _get_non_negative_int(
        pi_config,
        "call_timeout_seconds",
        section="pi",
        default=180,
    )
    pi_session_sweeper_interval_seconds = _get_non_negative_int(
        pi_config,
        "session_sweeper_interval_seconds",
        section="pi",
        default=60,
    )

    bot_queue_maxsize = _get_positive_int(bot_config, "queue_maxsize", section="bot", default=1000)
    bot_worker_concurrency = _get_positive_int(bot_config, "worker_concurrency", section="bot", default=4)
    bot_heartbeat_interval_seconds = _get_non_negative_int(
        bot_config,
        "heartbeat_interval_seconds",
        section="bot",
        default=0,
    )
    bot_metrics_interval_seconds = _get_non_negative_int(
        bot_config,
        "metrics_interval_seconds",
        section="bot",
        default=60,
    )
    bot_stream_edit_interval_ms = _get_positive_int(
        bot_config,
        "stream_edit_interval_ms",
        section="bot",
        default=700,
    )
    bot_latency_window_size = _get_positive_int(
        bot_config,
        "latency_window_size",
        section="bot",
        default=200,
    )
    dashboard_enabled = _get_bool(dashboard_config, "enabled", section="dashboard", default=False)
    dashboard_host = _get_optional_string(dashboard_config, "host", section="dashboard") or "127.0.0.1"
    dashboard_port = _get_positive_int(dashboard_config, "port", section="dashboard", default=8765)
    dashboard_base_path_raw = (
        _get_optional_string(dashboard_config, "base_path", section="dashboard") or "/dashboard"
    )
    dashboard_base_path = _normalize_path_prefix(
        dashboard_base_path_raw,
        section="dashboard",
        key="base_path",
    )
    dashboard_access_token = _get_optional_string(dashboard_config, "access_token", section="dashboard")
    dashboard_ws_push_interval_ms = _get_positive_int(
        dashboard_config,
        "ws_push_interval_ms",
        section="dashboard",
        default=1000,
    )
    _validate_dashboard_host(dashboard_host, enabled=dashboard_enabled)
    if dashboard_enabled and dashboard_access_token is None:
        raise RuntimeError("[dashboard] access_token must be a non-empty string when dashboard is enabled")

    log_level = (_get_optional_string(bot_config, "log_level", section="bot") or "INFO").upper()
    if log_level not in _VALID_LOG_LEVELS:
        allowed = ", ".join(_VALID_LOG_LEVELS)
        raise RuntimeError(f"[bot] log_level must be one of: {allowed}")

    pi_data_dir = Path(pi_data_dir_raw).expanduser().resolve() if pi_data_dir_raw else None
    if pi_data_dir is not None:
        pi_data_dir.mkdir(parents=True, exist_ok=True)

    return BotConfig(
        telegram_bot_token=token,
        telegram_bot_owner_user_id=owner_user_id,
        telegram_trusted_user_ids=tuple(trusted_user_ids),
        pi_executable=pi_executable,
        pi_provider=pi_provider,
        pi_model=pi_model,
        pi_data_dir=pi_data_dir,
        pi_session_root=Path(pi_session_root).expanduser().resolve(),
        pi_session_ttl_seconds=pi_session_ttl_seconds,
        pi_call_timeout_seconds=pi_call_timeout_seconds,
        pi_session_sweeper_interval_seconds=pi_session_sweeper_interval_seconds,
        bot_queue_maxsize=bot_queue_maxsize,
        bot_worker_concurrency=bot_worker_concurrency,
        bot_heartbeat_interval_seconds=bot_heartbeat_interval_seconds,
        bot_metrics_interval_seconds=bot_metrics_interval_seconds,
        bot_stream_edit_interval_ms=bot_stream_edit_interval_ms,
        bot_latency_window_size=bot_latency_window_size,
        dashboard_enabled=dashboard_enabled,
        dashboard_host=dashboard_host,
        dashboard_port=dashboard_port,
        dashboard_base_path=dashboard_base_path,
        dashboard_access_token=dashboard_access_token,
        dashboard_ws_push_interval_ms=dashboard_ws_push_interval_ms,
        log_level=log_level,
    )


def configure_pi_environment(config: BotConfig) -> None:
    if config.pi_data_dir is None:
        return

    os.environ["PI_CODING_AGENT_DIR"] = str(config.pi_data_dir)
    LOGGER.info("Configured PI_CODING_AGENT_DIR=%s", config.pi_data_dir)


def ensure_event_loop() -> None:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
        return

    if loop.is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())


def _as_table(value: object, *, name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeError(f"[{name}] must be a TOML table")
    return value


def _get_required_string(table: dict[str, object], key: str, *, section: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"[{section}] {key} must be a non-empty string")
    return value.strip()


def _get_optional_string(table: dict[str, object], key: str, *, section: str) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError(f"[{section}] {key} must be a string when set")
    stripped = value.strip()
    return stripped or None


def _get_required_positive_int(table: dict[str, object], key: str, *, section: str) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(f"[{section}] {key} must be a positive integer")
    return value


def _get_positive_int(table: dict[str, object], key: str, *, section: str, default: int) -> int:
    value = table.get(key)
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(f"[{section}] {key} must be a positive integer")
    return value


def _get_non_negative_int(table: dict[str, object], key: str, *, section: str, default: int) -> int:
    value = table.get(key)
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"[{section}] {key} must be a non-negative integer")
    return value


def _get_positive_int_list(table: dict[str, object], key: str, *, section: str) -> list[int]:
    value = table.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise RuntimeError(f"[{section}] {key} must be a list of positive integers when set")

    values: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise RuntimeError(f"[{section}] {key}[{index}] must be a positive integer")
        values.append(item)

    return list(dict.fromkeys(values))


def _get_bool(table: dict[str, object], key: str, *, section: str, default: bool) -> bool:
    value = table.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise RuntimeError(f"[{section}] {key} must be a boolean")
    return value


def _normalize_path_prefix(value: str, *, section: str, key: str) -> str:
    normalized = f"/{value.strip('/')}" if value.strip("/") else "/"
    if normalized == "/":
        raise RuntimeError(f"[{section}] {key} cannot be /")
    return normalized


def _validate_dashboard_host(host: str, *, enabled: bool) -> None:
    if not enabled:
        return
    allowed_hosts = {"127.0.0.1", "localhost", "::1"}
    if host not in allowed_hosts:
        allowed = ", ".join(sorted(allowed_hosts))
        raise RuntimeError(f"[dashboard] host must be local-only when enabled ({allowed})")
