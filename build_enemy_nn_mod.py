#!/usr/bin/env python3
"""Build enemy_nn_backend_mod.pk3 from enemy_nn_backend_mod_src."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def build_pk3(src_dir: Path, output_pk3: Path) -> None:
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    with ZipFile(output_pk3, "w", compression=ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_dir():
                continue
            arcname = path.relative_to(src_dir).as_posix()
            zf.write(path, arcname=arcname)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build enemy NN backend mod PK3.")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("enemy_nn_backend_mod_src"),
        help="Directory containing mod source files (CVARINFO/MAPINFO/ZSCRIPT).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("enemy_nn_backend_mod.pk3"),
        help="Output PK3 path.",
    )
    args = parser.parse_args()

    src_dir = args.src.resolve()
    out_pk3 = args.out.resolve()
    build_pk3(src_dir, out_pk3)
    print(f"Built: {out_pk3}")


if __name__ == "__main__":
    main()
