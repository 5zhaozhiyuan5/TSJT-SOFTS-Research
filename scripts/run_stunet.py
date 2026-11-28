#!/usr/bin/env python
"""Launch the pinned upstream STUNet code with diagnostic hooks enabled."""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--entry", choices=("train", "eval"), default="train")
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    repo_root = Path(__file__).resolve().parents[1]
    upstream = repo_root / "third_party" / "STUNet"
    if not (upstream / "src" / "train.py").exists():
        raise RuntimeError("STUNet submodule missing. Run: git submodule update --init --recursive")

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(upstream))
    os.environ.setdefault("STUNET_DIAG_DIR", str(repo_root / "runs" / "stunet_diagnostics"))

    from stunet_diagnostics.install import install
    run_dir = install()
    print(f"[STUNet diagnostics] writing logs to: {run_dir}")

    os.chdir(upstream)
    if args.entry == "train":
        from src.train import main as hydra_main
    else:
        from src.eval import main as hydra_main
    hydra_main()


if __name__ == "__main__":
    main()
