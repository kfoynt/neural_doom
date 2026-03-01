#!/usr/bin/env python3
"""Run headless enemy-backend regression suites and enforce release gates."""

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
    r"close_pairs_per_tick=(?P<close_pairs>[0-9.]+)\s+"
    r"(?:identity_churn_rate=(?P<identity_churn>[0-9.]+)\s+)?"
    r"player_max_stuck_ticks=(?P<stuck>\d+)\s+"
    r"mse_mean=(?P<mse_mean>[0-9.]+)\s+"
    r"mse_drift=(?P<mse_drift>-?[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run headless enemy-backend regression")
    p.add_argument("--python", default=sys.executable, help="Python executable")
    p.add_argument("--backend", default="e1m1_transformer_backend.py", help="Backend script path")
    p.add_argument("--wad", default="DOOM.WAD", help="Path to DOOM.WAD")
    p.add_argument("--map", default="E1M1", help="Single map name (legacy shortcut)")
    p.add_argument("--maps", default="", help="Comma-separated map list (e.g. E1M1,E1M2)")
    p.add_argument("--seeds", default="1,2,3", help="Comma-separated seed list")
    p.add_argument(
        "--tick-suites",
        default="",
        help="Comma-separated max-tick suite list (e.g. 5000,20000).",
    )
    p.add_argument(
        "--release-gate",
        action="store_true",
        help=(
            "Run release criteria suite by default: multi-map, multi-seed, and 5000+20000 tick gates "
            "(unless overridden with --maps/--seeds/--tick-suites)."
        ),
    )
    p.add_argument("--enemy-backend-mod", default="enemy_nn_backend_mod.pk3", help="Enemy backend mod path")
    p.add_argument("--enemy-slots", type=int, default=16, help="Enemy slots")
    p.add_argument("--max-ticks", type=int, default=5000, help="Headless ticks (use 5000-20000 for long-run gates)")
    p.add_argument("--log-interval", type=int, default=256, help="Tick log interval")
    p.add_argument("--max-close-pairs-per-tick", type=float, default=4.0)
    p.add_argument("--max-target-switches-per-enemy-tick", type=float, default=0.35)
    p.add_argument("--min-shots-per-tick", type=float, default=0.02)
    p.add_argument("--max-shots-per-tick", type=float, default=1.25)
    p.add_argument("--max-identity-churn-rate", type=float, default=0.15)
    p.add_argument("--max-player-stuck-ticks", type=int, default=96)
    p.add_argument("--max-mse-mean", type=float, default=0.20)
    p.add_argument("--max-mse-drift", type=float, default=0.03)
    p.add_argument("--fail-fast", action="store_true", help="Stop on first failing map/seed case")
    return p.parse_args()


def _parse_maps(args: argparse.Namespace) -> list[str]:
    raw = args.maps.strip()
    if raw:
        maps = [m.strip() for m in raw.split(",") if m.strip()]
        if maps:
            return maps
    if args.release_gate:
        return ["E1M1", "E1M2", "E1M3"]
    return [args.map]


def _parse_seeds(seed_text: str) -> list[int]:
    seeds: list[int] = []
    for token in seed_text.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        seeds = [1]
    return seeds


def _parse_tick_suites(args: argparse.Namespace) -> list[int]:
    raw = args.tick_suites.strip()
    if raw:
        suites: list[int] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            suites.append(max(1, int(token)))
        if suites:
            return suites
    if args.release_gate:
        return [5000, 20000]
    return [max(1, int(args.max_ticks))]


def _evaluate_case(args: argparse.Namespace, map_name: str, seed: int, ticks: int) -> tuple[list[str], int]:
    backend = Path(args.backend)
    wad = Path(args.wad)
    mod = Path(args.enemy_backend_mod)

    cmd = [
        args.python,
        str(backend),
        "--wad",
        str(wad),
        "--map",
        map_name,
        "--headless",
        "--max-ticks",
        str(ticks),
        "--log-interval",
        str(args.log_interval),
        "--enemy-backend-transformer",
        "--enemy-backend-mod",
        str(mod),
        "--enemy-slots",
        str(args.enemy_slots),
        "--doom-seed",
        str(seed),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(f"=== ticks={ticks} map={map_name} seed={seed} ===")
    print(proc.stdout)

    failures: list[str] = []
    if proc.returncode != 0:
        failures.append(f"backend exit status {proc.returncode}")
        return failures, proc.returncode

    match = METRIC_RE.search(proc.stdout)
    if match is None:
        failures.append("missing 'Enemy regression metrics' line")
        return failures, 2

    ticks = int(match.group("ticks"))
    shots_per_tick = float(match.group("shots"))
    switches_per_enemy_tick = float(match.group("switch_enemy"))
    close_pairs_per_tick = float(match.group("close_pairs"))
    identity_churn_rate = float(match.group("identity_churn") or 0.0)
    player_max_stuck_ticks = int(match.group("stuck"))
    mse_mean = float(match.group("mse_mean"))
    mse_drift = float(match.group("mse_drift"))

    if shots_per_tick < args.min_shots_per_tick:
        failures.append(
            f"shots_per_tick {shots_per_tick:.3f} < min_shots_per_tick {args.min_shots_per_tick:.3f}"
        )
    if shots_per_tick > args.max_shots_per_tick:
        failures.append(
            f"shots_per_tick {shots_per_tick:.3f} > max_shots_per_tick {args.max_shots_per_tick:.3f}"
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
    if identity_churn_rate > args.max_identity_churn_rate:
        failures.append(
            f"identity_churn_rate {identity_churn_rate:.4f} > max_identity_churn_rate {args.max_identity_churn_rate:.4f}"
        )
    if player_max_stuck_ticks > args.max_player_stuck_ticks:
        failures.append(
            f"player_max_stuck_ticks {player_max_stuck_ticks} > max_player_stuck_ticks {args.max_player_stuck_ticks}"
        )
    if mse_mean > args.max_mse_mean:
        failures.append(
            f"mse_mean {mse_mean:.6f} > max_mse_mean {args.max_mse_mean:.6f}"
        )
    if abs(mse_drift) > args.max_mse_drift:
        failures.append(
            f"|mse_drift| {abs(mse_drift):.6f} > max_mse_drift {args.max_mse_drift:.6f}"
        )

    print(
        "Parsed metrics: "
        f"ticks={ticks} shots_per_tick={shots_per_tick:.3f} "
        f"target_switches_per_enemy_tick={switches_per_enemy_tick:.4f} "
        f"close_pairs_per_tick={close_pairs_per_tick:.3f} "
        f"identity_churn_rate={identity_churn_rate:.4f} "
        f"player_max_stuck_ticks={player_max_stuck_ticks} "
        f"mse_mean={mse_mean:.6f} mse_drift={mse_drift:.6f}"
    )
    return failures, 0


def main() -> int:
    args = parse_args()
    maps = _parse_maps(args)
    seeds = _parse_seeds(args.seeds)
    tick_suites = _parse_tick_suites(args)

    all_failures: list[str] = []
    case_count = 0
    for ticks in tick_suites:
        for map_name in maps:
            for seed in seeds:
                case_count += 1
                failures, status = _evaluate_case(args, map_name, seed, ticks)
                if failures:
                    for failure in failures:
                        all_failures.append(f"[ticks={ticks} map={map_name} seed={seed}] {failure}")
                    if args.fail_fast:
                        break
                if status != 0 and args.fail_fast:
                    break
            if all_failures and args.fail_fast:
                break
        if all_failures and args.fail_fast:
            break

    print(f"Suite summary: cases={case_count} failures={len(all_failures)}")

    if all_failures:
        print("FAIL:", file=sys.stderr)
        for line in all_failures:
            print(f"- {line}", file=sys.stderr)
        return 3

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
