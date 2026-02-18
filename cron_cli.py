from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from cron_runtime import (
    CronJob,
    CronJobCreateInput,
    compute_job_next_run_at,
    create_cron_job,
    load_cron_store,
    save_cron_store,
    resolve_cron_store_path,
)


def _parse_store_path(raw_path: str | None) -> Path:
    return resolve_cron_store_path(raw_path)


def _default_state() -> dict[str, Any]:
    return {
        "next_run_at_unix": None,
        "running_at_unix": None,
        "last_run_at_unix": None,
        "last_status": None,
        "last_error": None,
        "last_duration_ms": None,
        "consecutive_errors": 0,
    }


def _normalize_job(job: CronJob, now_unix: float) -> None:
    state = job.get("state")
    if not isinstance(state, dict):
        job["state"] = _default_state()
        state = job["state"]

    if not job.get("enabled", True):
        state["next_run_at_unix"] = None
        state["running_at_unix"] = None
        return

    if state.get("running_at_unix") is None:
        state["next_run_at_unix"] = compute_job_next_run_at(job, now_unix)


def _load_jobs(store_path: Path) -> list[CronJob]:
    store = load_cron_store(store_path)
    jobs = store.get("jobs", [])
    now_unix = time.time()
    for job in jobs:
        _normalize_job(job, now_unix)
    return jobs


def _save_jobs(store_path: Path, jobs: list[CronJob]) -> None:
    save_cron_store(store_path, {"version": 1, "jobs": jobs})


def _find_job(jobs: list[CronJob], job_id: str) -> CronJob | None:
    return next((job for job in jobs if job.get("id") == job_id), None)


def cmd_list(args: argparse.Namespace) -> dict[str, Any]:
    store_path = _parse_store_path(args.store_path)
    jobs = _load_jobs(store_path)
    jobs.sort(
        key=lambda job: (
            1 if job["state"].get("next_run_at_unix") is None else 0,
            job["state"].get("next_run_at_unix") or 0,
        )
    )
    return {
        "ok": True,
        "store_path": str(store_path),
        "job_count": len(jobs),
        "jobs": jobs,
    }


def cmd_add(args: argparse.Namespace) -> dict[str, Any]:
    store_path = _parse_store_path(args.store_path)
    jobs = _load_jobs(store_path)
    now_unix = time.time()

    if args.cron_expr:
        schedule: dict[str, Any] = {
            "kind": "cron",
            "expr": args.cron_expr,
            "timezone": args.timezone,
        }
    elif args.every_seconds is not None:
        schedule = {
            "kind": "every",
            "every_seconds": args.every_seconds,
            "anchor_unix": now_unix,
        }
    else:
        schedule = {
            "kind": "at",
            "at": args.at,
        }

    job = create_cron_job(
        CronJobCreateInput(
            name=args.name,
            schedule=schedule,
            prompt=args.prompt,
            session_target=args.session_target,
            notify_owner=args.notify_owner,
            delete_after_run=args.delete_after_run,
            force_ephemeral=args.force_ephemeral,
            force_session=args.force_session,
            enabled=args.enabled,
        ),
        now_unix=now_unix,
    )

    jobs.append(job)
    _save_jobs(store_path, jobs)
    return {"ok": True, "store_path": str(store_path), "job": job}


def cmd_remove(args: argparse.Namespace) -> dict[str, Any]:
    store_path = _parse_store_path(args.store_path)
    jobs = _load_jobs(store_path)
    before = len(jobs)
    jobs = [job for job in jobs if job.get("id") != args.id]
    if len(jobs) == before:
        return {"ok": False, "error": f"job not found: {args.id}", "store_path": str(store_path)}
    _save_jobs(store_path, jobs)
    return {"ok": True, "store_path": str(store_path), "removed_job_id": args.id}


def cmd_enable(args: argparse.Namespace) -> dict[str, Any]:
    return _set_enabled(args, enabled=True)


def cmd_disable(args: argparse.Namespace) -> dict[str, Any]:
    return _set_enabled(args, enabled=False)


def _set_enabled(args: argparse.Namespace, *, enabled: bool) -> dict[str, Any]:
    store_path = _parse_store_path(args.store_path)
    jobs = _load_jobs(store_path)
    job = _find_job(jobs, args.id)
    if job is None:
        return {"ok": False, "error": f"job not found: {args.id}", "store_path": str(store_path)}

    now_unix = time.time()
    job["enabled"] = enabled
    job["updated_at_unix"] = now_unix
    if enabled:
        job["state"]["next_run_at_unix"] = compute_job_next_run_at(job, now_unix)
    else:
        job["state"]["next_run_at_unix"] = None
        job["state"]["running_at_unix"] = None

    _save_jobs(store_path, jobs)
    return {"ok": True, "store_path": str(store_path), "job": job}


def cmd_run_now(args: argparse.Namespace) -> dict[str, Any]:
    store_path = _parse_store_path(args.store_path)
    jobs = _load_jobs(store_path)
    job = _find_job(jobs, args.id)
    if job is None:
        return {"ok": False, "error": f"job not found: {args.id}", "store_path": str(store_path)}

    now_unix = time.time()
    job["enabled"] = True
    job["updated_at_unix"] = now_unix
    job["state"]["running_at_unix"] = None
    job["state"]["next_run_at_unix"] = now_unix

    _save_jobs(store_path, jobs)
    return {"ok": True, "store_path": str(store_path), "job": job}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage vibeos cron job store.")
    parser.add_argument(
        "--store-path",
        help="Cron JSON store path (defaults to ~/.vibeos/cron/jobs.json).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List cron jobs.")
    list_parser.set_defaults(handler=cmd_list)

    add_parser = subparsers.add_parser("add", help="Add a cron job.")
    add_parser.add_argument("--name", required=True, help="Human-friendly job name.")
    add_parser.add_argument("--prompt", required=True, help="Prompt executed for the job.")
    add_parser.add_argument(
        "--session-target",
        choices=("owner", "isolated"),
        default="owner",
        help="Run in owner DM session or an isolated cron session.",
    )
    add_parser.add_argument(
        "--notify-owner",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send cron run results to the owner DM.",
    )
    add_parser.add_argument(
        "--enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the new job immediately.",
    )
    add_parser.add_argument(
        "--delete-after-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Delete one-shot jobs after successful execution.",
    )
    add_parser.add_argument("--force-ephemeral", action="store_true", help="Force an ephemeral PI session.")
    add_parser.add_argument("--force-session", action="store_true", help="Force session persistence.")
    schedule_group = add_parser.add_mutually_exclusive_group(required=True)
    schedule_group.add_argument("--cron-expr", help="Cron expression, for example */15 * * * *.")
    schedule_group.add_argument("--every-seconds", type=int, help="Fixed interval schedule in seconds.")
    schedule_group.add_argument("--at", help="One-shot timestamp (ISO-8601).")
    add_parser.add_argument(
        "--timezone",
        default="UTC",
        help="IANA timezone for --cron-expr schedule (default: UTC).",
    )
    add_parser.set_defaults(handler=cmd_add)

    remove_parser = subparsers.add_parser("remove", help="Remove a cron job.")
    remove_parser.add_argument("--id", required=True, help="Job ID.")
    remove_parser.set_defaults(handler=cmd_remove)

    enable_parser = subparsers.add_parser("enable", help="Enable a cron job.")
    enable_parser.add_argument("--id", required=True, help="Job ID.")
    enable_parser.set_defaults(handler=cmd_enable)

    disable_parser = subparsers.add_parser("disable", help="Disable a cron job.")
    disable_parser.add_argument("--id", required=True, help="Job ID.")
    disable_parser.set_defaults(handler=cmd_disable)

    run_parser = subparsers.add_parser("run-now", help="Mark job as due now.")
    run_parser.add_argument("--id", required=True, help="Job ID.")
    run_parser.set_defaults(handler=cmd_run_now)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = args.handler(args)
    except Exception as exc:  # noqa: BLE001
        json.dump({"ok": False, "error": str(exc)}, sys.stdout)
        sys.stdout.write("\n")
        return 1

    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
