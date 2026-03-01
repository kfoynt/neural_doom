#!/usr/bin/env python3
"""Run a headless enemy-backend regression pass and parse metric output."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


METRIC_RE = re.compile(
    r"Enemy regression metrics:\s*"
    r"ticks=(?P<ticks>\d+)\s+"
    r"shots_per_tick=(?P<shots>[0-9.]+)\s+"
    r"target_switches_per_tick=(?P<switch_tick>[0-9.]+)\s+"
    r"target_switches_per_enemy_tick=(?P<switch_enemy>[0-9.]+)\s+"
    r"close_pairs_per_tick=(?P<close_pairs>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run headless enemy-backend regression")
    p.add_argument("--python", default=sys.executable, help="Python executable")
    p.add_argument("--backend", default="e1m1_transformer_backend.py", help="Backend script path")
    p.add_argument("--wad", default="DOOM.WAD", help="Path to DOOM.WAD")
    p.add_argument("--map", default="E1M1", help="Map name")
    p.add_argument("--enemy-backend-mod", default="enemy_nn_backend_mod.pk3", help="Enemy backend mod path")
    p.add_argument("--enemy-slots", type=int, default=16, help="Enemy slots")
    p.add_argument("--max-ticks", type=int, default=256, help="Headless ticks")
    p.add_argument("--log-interval", type=int, default=32, help="Tick log interval")
    p.add_argument("--max-close-pairs-per-tick", type=float, default=4.0)
    p.add_argument("--max-target-switches-per-enemy-tick", type=float, default=0.20)
    p.add_argument("--min-shots-per-tick", type=float, default=0.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    backend = Path(args.backend)
    wad = Path(args.wad)
    mod = Path(args.enemy_backend_mod)

    cmd = [
        args.python,
        str(backend),
        "--wad",
        str(wad),
        "--map",
        args.map,
        "--headless",
        "--max-ticks",
        str(args.max_ticks),
        "--log-interval",
        str(args.log_interval),
        "--enemy-backend-transformer",
        "--enemy-backend-mod",
        str(mod),
        "--enemy-slots",
        str(args.enemy_slots),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)

    match = METRIC_RE.search(proc.stdout)
    if match is None:
        print("FAIL: Could not find 'Enemy regression metrics' line in output.", file=sys.stderr)
        return 2

    ticks = int(match.group("ticks"))
    shots_per_tick = float(match.group("shots"))
    switches_per_enemy_tick = float(match.group("switch_enemy"))
    close_pairs_per_tick = float(match.group("close_pairs"))

    failures: list[str] = []
    if shots_per_tick < args.min_shots_per_tick:
        failures.append(
            f"shots_per_tick {shots_per_tick:.3f} < min_shots_per_tick {args.min_shots_per_tick:.3f}"
        )
    if switches_per_enemy_tick > args.max_target_switches_per_enemy_tick:
        failures.append(
            "target_switches_per_enemy_tick "
            f"{switches_per_enemy_tick:.4f} > max_target_switches_per_enemy_tick "
            f"{args.max_target_switches_per_enemy_tick:.4f}"
        )
    if close_pairs_per_tick > args.max_close_pairs_per_tick:
        failures.append(
            f"close_pairs_per_tick {close_pairs_per_tick:.3f} > max_close_pairs_per_tick {args.max_close_pairs_per_tick:.3f}"
        )

    print(
        "Parsed metrics: "
        f"ticks={ticks} shots_per_tick={shots_per_tick:.3f} "
        f"target_switches_per_enemy_tick={switches_per_enemy_tick:.4f} "
        f"close_pairs_per_tick={close_pairs_per_tick:.3f}"
    )

    if failures:
        print("FAIL:", file=sys.stderr)
        for line in failures:
            print(f"- {line}", file=sys.stderr)
        return 3

    if proc.returncode != 0:
        print(f"FAIL: Backend exited with status {proc.returncode}", file=sys.stderr)
        return proc.returncode

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
