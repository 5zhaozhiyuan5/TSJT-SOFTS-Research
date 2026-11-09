"""TSJT-PGrad 的独立命令行入口。

用法：``python -m tsjt_pgrad.run --config configs/pgrad_pemsbay.yaml``
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .training.trainer import run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TSJT-PGrad runner")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    artifact = run(args.config)
    print(json.dumps({"artifact": artifact}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run"]
