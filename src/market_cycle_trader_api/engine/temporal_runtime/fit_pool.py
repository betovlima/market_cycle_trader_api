from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
from typing import Callable, TypeVar

T = TypeVar("T")


def _cgroup_cpu_quota() -> int | None:
    path = "/sys/fs/cgroup/cpu.max"
    try:
        quota_text, period_text = open(path, "r", encoding="utf-8").read().strip().split()[:2]
        if quota_text == "max":
            return None
        quota = int(quota_text)
        period = int(period_text)
        if quota <= 0 or period <= 0:
            return None
        return max(1, int(math.ceil(quota / period)))
    except (OSError, ValueError, IndexError):
        return None


def _available_cpus() -> int:
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if callable(process_cpu_count):
        value = process_cpu_count()
        if value:
            return max(1, int(value))
    quota = _cgroup_cpu_quota()
    if quota is not None:
        return quota
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))


def temporal_fit_worker_count(task_count: int) -> int:
    override = str(os.getenv("MCT_TEMPORAL_FIT_WORKERS") or "").strip()
    if override:
        try:
            requested = max(1, int(override))
        except ValueError:
            requested = _available_cpus()
    else:
        requested = _available_cpus()
    return max(1, min(int(task_count), 5, requested))


def run_independent_fit_tasks(
    tasks: dict[str, Callable[[], T]],
    *,
    cancel_check: Callable[[], None] | None = None,
    completed_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, T]:
    if not tasks:
        return {}
    workers = temporal_fit_worker_count(len(tasks))
    if workers <= 1:
        results: dict[str, T] = {}
        total = len(tasks)
        for position, (name, task) in enumerate(tasks.items(), start=1):
            if cancel_check is not None:
                cancel_check()
            results[name] = task()
            if completed_callback is not None:
                completed_callback(name, position, total)
        return results
    results: dict[str, T] = {}
    total = len(tasks)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mct-temporal-fit") as executor:
        futures = {executor.submit(task): name for name, task in tasks.items()}
        completed = 0
        for future in as_completed(futures):
            name = futures[future]
            results[name] = future.result()
            completed += 1
            if completed_callback is not None:
                completed_callback(name, completed, total)
            if cancel_check is not None:
                cancel_check()
    return results
