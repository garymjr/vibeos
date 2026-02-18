from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, TypedDict
from zoneinfo import ZoneInfo

from croniter import croniter

LOGGER = logging.getLogger("assistant.cron")
DEFAULT_CRON_STORE_PATH = Path("~/.vibeos/cron/jobs.json").expanduser().resolve()

CronRunStatus = Literal["ok", "error", "skipped"]
CronSessionTarget = Literal["owner", "isolated"]


class CronScheduleAt(TypedDict):
    kind: Literal["at"]
    at: str


class CronScheduleEvery(TypedDict):
    kind: Literal["every"]
    every_seconds: int
    anchor_unix: float


class CronScheduleExpr(TypedDict):
    kind: Literal["cron"]
    expr: str
    timezone: str


CronSchedule = CronScheduleAt | CronScheduleEvery | CronScheduleExpr


class CronPayload(TypedDict):
    prompt: str
    session_target: CronSessionTarget
    notify_owner: bool
    force_ephemeral: bool
    force_session: bool


class CronJobState(TypedDict):
    next_run_at_unix: float | None
    running_at_unix: float | None
    last_run_at_unix: float | None
    last_status: CronRunStatus | None
    last_error: str | None
    last_duration_ms: int | None
    consecutive_errors: int


class CronJob(TypedDict):
    id: str
    name: str
    enabled: bool
    delete_after_run: bool
    created_at_unix: float
    updated_at_unix: float
    schedule: CronSchedule
    payload: CronPayload
    state: CronJobState


class CronStoreFile(TypedDict):
    version: int
    jobs: list[CronJob]


@dataclass(slots=True)
class CronRunOutcome:
    status: CronRunStatus
    summary: str | None = None
    error: str | None = None


@dataclass(slots=True)
class CronJobCreateInput:
    name: str
    schedule: CronSchedule
    prompt: str
    session_target: CronSessionTarget = "owner"
    notify_owner: bool = True
    delete_after_run: bool | None = None
    force_ephemeral: bool = False
    force_session: bool = False
    enabled: bool = True


CronJobExecutor = Callable[[CronJob], Awaitable[CronRunOutcome]]


def resolve_cron_store_path(raw_path: str | None) -> Path:
    if raw_path and raw_path.strip():
        return Path(raw_path.strip()).expanduser().resolve()
    return DEFAULT_CRON_STORE_PATH


def parse_absolute_time_unix(value: str) -> float:
    text = value.strip()
    if not text:
        raise ValueError("schedule.at must be a non-empty timestamp")
    normalized = text.replace("Z", "+00:00") if text.endswith("Z") else text
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _format_iso_utc(unix_seconds: float) -> str:
    return datetime.fromtimestamp(unix_seconds, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_timezone(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name.strip() or "UTC")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid timezone: {tz_name}") from exc


def _compute_schedule_next_run_at(schedule: CronSchedule, now_unix: float) -> float | None:
    if schedule["kind"] == "at":
        return parse_absolute_time_unix(schedule["at"])

    if schedule["kind"] == "every":
        every_seconds = max(1, int(schedule["every_seconds"]))
        anchor_unix = float(schedule.get("anchor_unix", now_unix))
        if now_unix < anchor_unix:
            return anchor_unix
        elapsed = now_unix - anchor_unix
        steps = math.floor(elapsed / every_seconds) + 1
        return anchor_unix + (steps * every_seconds)

    expr = schedule["expr"].strip()
    if not expr:
        return None
    tz_name = schedule.get("timezone", "UTC")
    tz = _resolve_timezone(tz_name)
    base_dt = datetime.fromtimestamp(now_unix, tz=tz)
    itr = croniter(expr, base_dt, ret_type=datetime)
    for _ in range(4):
        candidate = itr.get_next(datetime).timestamp()
        if candidate > now_unix:
            return candidate
    return None


def compute_job_next_run_at(job: CronJob, now_unix: float) -> float | None:
    if not job["enabled"]:
        return None
    schedule = job["schedule"]
    if schedule["kind"] == "at":
        # One-shot jobs remain due until they run once.
        if job["state"].get("last_status") in {"ok", "error", "skipped"}:
            return None
        return _compute_schedule_next_run_at(schedule, now_unix)
    return _compute_schedule_next_run_at(schedule, now_unix)


def load_cron_store(store_path: Path) -> CronStoreFile:
    try:
        raw = store_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"version": 1, "jobs": []}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse cron store at {store_path}: {exc}") from exc

    if not isinstance(data, dict):
        return {"version": 1, "jobs": []}
    jobs = data.get("jobs")
    if not isinstance(jobs, list):
        jobs = []
    sanitized: list[CronJob] = []
    for raw_job in jobs:
        if isinstance(raw_job, dict):
            sanitized.append(copy.deepcopy(raw_job))  # type: ignore[arg-type]
    return {"version": 1, "jobs": sanitized}


def save_cron_store(store_path: Path, store: CronStoreFile) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = store_path.with_suffix(f"{store_path.suffix}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(store, indent=2), encoding="utf-8")
    tmp_path.replace(store_path)
    with contextlib.suppress(Exception):
        backup_path = store_path.with_suffix(f"{store_path.suffix}.bak")
        backup_path.write_text(store_path.read_text(encoding="utf-8"), encoding="utf-8")


def create_cron_job(input_data: CronJobCreateInput, now_unix: float | None = None) -> CronJob:
    now = float(now_unix if now_unix is not None else time.time())
    schedule: CronSchedule
    if input_data.schedule["kind"] == "every":
        schedule = {
            "kind": "every",
            "every_seconds": max(1, int(input_data.schedule["every_seconds"])),
            "anchor_unix": float(input_data.schedule.get("anchor_unix", now)),
        }
    elif input_data.schedule["kind"] == "cron":
        timezone_name = input_data.schedule.get("timezone", "UTC")
        _resolve_timezone(timezone_name)
        schedule = {
            "kind": "cron",
            "expr": input_data.schedule["expr"].strip(),
            "timezone": timezone_name,
        }
    else:
        schedule = {
            "kind": "at",
            "at": _format_iso_utc(parse_absolute_time_unix(input_data.schedule["at"])),
        }

    if input_data.force_ephemeral and input_data.force_session:
        raise ValueError("force_ephemeral and force_session cannot both be true")

    delete_after_run = (
        input_data.delete_after_run
        if input_data.delete_after_run is not None
        else (schedule["kind"] == "at")
    )
    job: CronJob = {
        "id": uuid.uuid4().hex,
        "name": input_data.name.strip() or "Cron job",
        "enabled": bool(input_data.enabled),
        "delete_after_run": bool(delete_after_run),
        "created_at_unix": now,
        "updated_at_unix": now,
        "schedule": schedule,
        "payload": {
            "prompt": input_data.prompt.strip(),
            "session_target": input_data.session_target,
            "notify_owner": bool(input_data.notify_owner),
            "force_ephemeral": bool(input_data.force_ephemeral),
            "force_session": bool(input_data.force_session),
        },
        "state": {
            "next_run_at_unix": None,
            "running_at_unix": None,
            "last_run_at_unix": None,
            "last_status": None,
            "last_error": None,
            "last_duration_ms": None,
            "consecutive_errors": 0,
        },
    }
    job["state"]["next_run_at_unix"] = compute_job_next_run_at(job, now)
    return job


class CronScheduler:
    def __init__(
        self,
        *,
        enabled: bool,
        store_path: Path,
        max_sleep_seconds: int,
        max_concurrent_runs: int,
        default_timeout_seconds: int,
        executor: CronJobExecutor,
    ) -> None:
        self._enabled = enabled
        self._store_path = store_path
        self._max_sleep_seconds = max(1, max_sleep_seconds)
        self._default_timeout_seconds = max(0, default_timeout_seconds)
        self._executor = executor

        self._jobs: list[CronJob] = []
        self._store_mtime_ns: int | None = None
        self._lock = asyncio.Lock()
        self._wake_event = asyncio.Event()
        self._runner_task: asyncio.Task[None] | None = None
        self._execution_tasks: set[asyncio.Task[None]] = set()
        self._run_semaphore = asyncio.Semaphore(max(1, max_concurrent_runs))

    async def start(self) -> None:
        async with self._lock:
            self._load_store_locked(force=True)
            changed = self._normalize_jobs_locked(time.time())
            if changed:
                self._save_store_locked()

        if not self._enabled:
            LOGGER.info("Cron scheduler disabled")
            return

        if self._runner_task is None or self._runner_task.done():
            self._runner_task = asyncio.create_task(self._runner_loop(), name="cron-scheduler")
            LOGGER.info("Cron scheduler started store=%s jobs=%s", self._store_path, len(self._jobs))
        self._wake_event.set()

    async def stop(self) -> None:
        if self._runner_task is not None:
            self._runner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner_task
            self._runner_task = None

        for task in list(self._execution_tasks):
            task.cancel()
        if self._execution_tasks:
            await asyncio.gather(*self._execution_tasks, return_exceptions=True)
        self._execution_tasks.clear()

    async def add_job(self, input_data: CronJobCreateInput) -> CronJob:
        async with self._lock:
            self._load_store_locked()
            now = time.time()
            job = create_cron_job(input_data, now_unix=now)
            self._jobs.append(job)
            self._save_store_locked()
        self._wake_event.set()
        return copy.deepcopy(job)

    async def remove_job(self, job_id: str) -> bool:
        async with self._lock:
            self._load_store_locked()
            before = len(self._jobs)
            self._jobs = [job for job in self._jobs if job["id"] != job_id]
            removed = len(self._jobs) != before
            if removed:
                self._save_store_locked()
        if removed:
            self._wake_event.set()
        return removed

    async def set_enabled(self, job_id: str, enabled: bool) -> bool:
        async with self._lock:
            self._load_store_locked()
            job = next((entry for entry in self._jobs if entry["id"] == job_id), None)
            if job is None:
                return False
            job["enabled"] = bool(enabled)
            job["updated_at_unix"] = time.time()
            if enabled:
                job["state"]["next_run_at_unix"] = compute_job_next_run_at(job, time.time())
            else:
                job["state"]["next_run_at_unix"] = None
                job["state"]["running_at_unix"] = None
            self._save_store_locked()
        self._wake_event.set()
        return True

    async def run_job_now(self, job_id: str) -> bool:
        async with self._lock:
            self._load_store_locked()
            job = next((entry for entry in self._jobs if entry["id"] == job_id), None)
            if job is None:
                return False
            if not job["enabled"]:
                job["enabled"] = True
            now = time.time()
            job["state"]["next_run_at_unix"] = now
            job["updated_at_unix"] = now
            self._save_store_locked()
        self._wake_event.set()
        return True

    def dashboard_snapshot(self) -> dict[str, object]:
        now = time.time()
        jobs = sorted(
            self._jobs,
            key=lambda job: (
                1 if job["state"].get("next_run_at_unix") is None else 0,
                job["state"].get("next_run_at_unix") or 0,
            ),
        )
        entries: list[dict[str, object]] = []
        due_count = 0
        for job in jobs:
            state = job["state"]
            next_run = state.get("next_run_at_unix")
            if isinstance(next_run, (int, float)) and next_run <= now and job["enabled"]:
                due_count += 1
            entries.append(
                {
                    "id": job["id"],
                    "name": job["name"],
                    "enabled": job["enabled"],
                    "schedule_kind": job["schedule"]["kind"],
                    "schedule_text": _schedule_to_text(job["schedule"]),
                    "next_run_at_unix": next_run,
                    "running": state.get("running_at_unix") is not None,
                    "last_status": state.get("last_status"),
                    "last_error": state.get("last_error"),
                    "last_run_at_unix": state.get("last_run_at_unix"),
                }
            )
        return {
            "enabled": self._enabled,
            "store_path": str(self._store_path),
            "job_count": len(entries),
            "due_count": due_count,
            "jobs": entries,
        }

    async def _runner_loop(self) -> None:
        while True:
            try:
                await self._reload_store_if_changed()
                due_job_ids = await self._collect_due_job_ids()
                for job_id in due_job_ids:
                    self._spawn_execution_task(job_id)
                sleep_seconds = await self._next_sleep_seconds()
                self._wake_event.clear()
                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=sleep_seconds)
                except asyncio.TimeoutError:
                    continue
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                LOGGER.exception("Cron scheduler loop failed")
                await asyncio.sleep(1)

    async def _reload_store_if_changed(self) -> None:
        async with self._lock:
            current_mtime = _file_mtime_ns(self._store_path)
            if self._store_mtime_ns is not None and current_mtime == self._store_mtime_ns:
                return
            self._load_store_locked(force=True)
            changed = self._normalize_jobs_locked(time.time())
            if changed:
                self._save_store_locked()

    async def _collect_due_job_ids(self) -> list[str]:
        async with self._lock:
            now = time.time()
            due_ids: list[str] = []
            for job in self._jobs:
                if not job["enabled"]:
                    continue
                if job["state"].get("running_at_unix") is not None:
                    continue
                next_run = job["state"].get("next_run_at_unix")
                if isinstance(next_run, (int, float)) and next_run <= now:
                    due_ids.append(job["id"])
                    job["state"]["running_at_unix"] = now
                    job["state"]["last_error"] = None
            if due_ids:
                self._save_store_locked()
            return due_ids

    def _spawn_execution_task(self, job_id: str) -> None:
        task = asyncio.create_task(self._execute_job(job_id), name=f"cron-job-{job_id}")
        self._execution_tasks.add(task)
        task.add_done_callback(lambda done: self._execution_tasks.discard(done))

    async def _execute_job(self, job_id: str) -> None:
        async with self._run_semaphore:
            started_at = time.time()
            async with self._lock:
                job = next((entry for entry in self._jobs if entry["id"] == job_id), None)
                if job is None:
                    return
                payload = copy.deepcopy(job)

            try:
                if self._default_timeout_seconds > 0:
                    outcome = await asyncio.wait_for(
                        self._executor(payload),
                        timeout=self._default_timeout_seconds,
                    )
                else:
                    outcome = await self._executor(payload)
            except asyncio.TimeoutError:
                outcome = CronRunOutcome(status="error", error="cron job execution timed out")
            except Exception as exc:  # noqa: BLE001
                outcome = CronRunOutcome(status="error", error=str(exc))

            await self._apply_job_outcome(
                job_id=job_id,
                started_at=started_at,
                ended_at=time.time(),
                outcome=outcome,
            )

    async def _apply_job_outcome(
        self,
        *,
        job_id: str,
        started_at: float,
        ended_at: float,
        outcome: CronRunOutcome,
    ) -> None:
        async with self._lock:
            job = next((entry for entry in self._jobs if entry["id"] == job_id), None)
            if job is None:
                return
            state = job["state"]
            state["running_at_unix"] = None
            state["last_run_at_unix"] = started_at
            state["last_status"] = outcome.status
            state["last_error"] = outcome.error
            state["last_duration_ms"] = max(0, int((ended_at - started_at) * 1000))
            state["consecutive_errors"] = (
                state.get("consecutive_errors", 0) + 1 if outcome.status == "error" else 0
            )
            job["updated_at_unix"] = ended_at

            if job["schedule"]["kind"] == "at":
                if job["delete_after_run"] and outcome.status == "ok":
                    self._jobs = [entry for entry in self._jobs if entry["id"] != job_id]
                else:
                    job["enabled"] = False
                    state["next_run_at_unix"] = None
            else:
                next_run = compute_job_next_run_at(job, ended_at)
                if outcome.status == "error":
                    backoff_seconds = _error_backoff_seconds(state["consecutive_errors"])
                    next_run = max(next_run or 0, ended_at + backoff_seconds)
                state["next_run_at_unix"] = next_run

            self._save_store_locked()

        self._wake_event.set()

    async def _next_sleep_seconds(self) -> float:
        async with self._lock:
            now = time.time()
            next_run_values = [
                entry["state"]["next_run_at_unix"]
                for entry in self._jobs
                if entry["enabled"] and isinstance(entry["state"].get("next_run_at_unix"), (int, float))
            ]
            if not next_run_values:
                return float(self._max_sleep_seconds)
            next_run = min(float(value) for value in next_run_values)
            delay = max(0.0, next_run - now)
            return min(delay, float(self._max_sleep_seconds))

    def _load_store_locked(self, *, force: bool = False) -> None:
        current_mtime = _file_mtime_ns(self._store_path)
        if not force and self._jobs and current_mtime == self._store_mtime_ns:
            return
        loaded = load_cron_store(self._store_path)
        self._jobs = loaded["jobs"]
        self._store_mtime_ns = current_mtime

    def _save_store_locked(self) -> None:
        save_cron_store(self._store_path, {"version": 1, "jobs": self._jobs})
        self._store_mtime_ns = _file_mtime_ns(self._store_path)

    def _normalize_jobs_locked(self, now_unix: float) -> bool:
        changed = False
        for job in list(self._jobs):
            state = job.get("state")
            if not isinstance(state, dict):
                job["state"] = {
                    "next_run_at_unix": None,
                    "running_at_unix": None,
                    "last_run_at_unix": None,
                    "last_status": None,
                    "last_error": None,
                    "last_duration_ms": None,
                    "consecutive_errors": 0,
                }
                state = job["state"]
                changed = True

            if state.get("running_at_unix") is not None:
                state["running_at_unix"] = None
                changed = True

            expected_next = compute_job_next_run_at(job, now_unix)
            if state.get("next_run_at_unix") != expected_next:
                state["next_run_at_unix"] = expected_next
                changed = True
        return changed


def _error_backoff_seconds(consecutive_errors: int) -> int:
    schedule = [30, 60, 5 * 60, 15 * 60, 60 * 60]
    index = max(0, min(consecutive_errors - 1, len(schedule) - 1))
    return schedule[index]


def _schedule_to_text(schedule: CronSchedule) -> str:
    if schedule["kind"] == "at":
        return schedule["at"]
    if schedule["kind"] == "every":
        return f"every {int(schedule['every_seconds'])}s"
    return f"{schedule['expr']} ({schedule.get('timezone', 'UTC')})"


def _file_mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None
    except Exception:
        return None
