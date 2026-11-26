#!/usr/bin/env python3
"""Read-only D1/D2 diagnostic from a previously trained ORIGINAL F2 checkpoint.

Loads the Stage-I F2 weights; never creates target test_dataset and never
executes optimizer.step(). Each audit pass takes a newly sampled target
SUPPORT batch and one source batch per city. Stage-I epochs in the audit
are synthetic trigger indices, not recorded checkpoint-training epochs.

Command:
python -u audit_f2_checkpoint.py --checkpoint /path/to/F2/l1_pems-bay_SOFTS_seed1.pth \
  --probe-batches 16 --output diagnostics/f2_d1d2_seed1.jsonl
"""
import argparse
import json
import io
import contextlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "model" / "Meta_Models"))

from datasets import traffic_dataset
from utils import get_data_list
from PMME import STRep


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Existing ORIGINAL F2 Stage-I state_dict checkpoint")
    parser.add_argument("--config-filename", default="configs/config.yaml")
    parser.add_argument("--data-list", default="chengdu_shenzhen_metr")
    parser.add_argument("--test-dataset", default="pems-bay")
    parser.add_argument("--target-days", type=int, default=3)
    parser.add_argument("--probe-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--source-budget", type=int, default=192)
    parser.add_argument("--target-budget", type=int, default=96)
    parser.add_argument("--daepoch", type=int, default=1500)
    parser.add_argument("--ot-scale", type=float, default=0.01)
    parser.add_argument("--output", type=Path,
                        default=Path("diagnostics/f2_d1d2_seed1.jsonl"))
    args = parser.parse_args()
    if args.probe_batches < 1 or args.target_days != 3:
        parser.error("Require probe-batches >=1 and three source cities")
    if not args.checkpoint.is_file():
        parser.error("Missing original F2 checkpoint: %s" % args.checkpoint)
    if args.source_budget < 8 or args.target_budget < 2:
        parser.error("Diagnostic subset budget is too small")
    return args


def source_target_batch(source_dataset, target_dataset, n_sources=3):
    data = []
    matrix = []
    target, adj = target_dataset[0]
    data.append(target)
    matrix.append(adj)
    used_shapes = {adj.shape[1]}
    for city_idx in range(n_sources):
        found = False
        for _ in range(50):
            sample, adjacency = source_dataset[city_idx]
            if adjacency.shape[1] not in used_shapes:
                data.append(sample)
                matrix.append(adjacency)
                used_shapes.add(adjacency.shape[1])
                found = True
                break
        if not found:
            raise RuntimeError("Could not select three distinct source cities")
    return data, matrix


def verify_unchanged(model, saved_params):
    for name, p in model.named_parameters():
        if not torch.equal(p.detach().cpu(), saved_params[name]):
            raise RuntimeError("Readonly audit unexpectedly changed parameter "+name)


def main():
    args = parse_args()
    # Resolve user paths BEFORE switching to the repository directory.
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    os.chdir(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.save_dir = args.output.parent
    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    args.STmodel = "SOFTS"
    args.new = 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Explicitly force the original F2 and prohibit all training mutations.
    os.environ.update({
        "PMME_TRANSFER_VARIANT": "gradient",
        "PMME_F2_METHOD": "original",
        "PMME_PAIR_VARIANT": "f2",
        "PMME_PAIR_STRENGTH": "0",
        "PMME_FEEDBACK_MODE": "g0",
        "PMME_F2_DIAGNOSTIC": "d1d2",
        "PMME_F2_DIAG_READONLY": "1",
        "PMME_F2_DIAG_INTERVAL": "1",
        "PMME_F2_DIAG_SOURCE_BUDGET": str(args.source_budget),
        "PMME_F2_DIAG_TARGET_BUDGET": str(args.target_budget),
        "PMME_TRANSFER_OT_SCALE": str(args.ot_scale),
        "PMME_TRANSFER_TEMPERATURE": "8",
    })
    with open(args.config_filename, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if int(config["task"]["source"]["batch_size"]) != 16:
        raise ValueError("Expected batch=16 from the matched experiment config")
    data_args, task_args, model_args = (
        config["data"], config["task"], config["model"]
    )
    names = get_data_list(args.data_list)
    if len(names) != 3:
        raise ValueError("Expected exactly three source cities")
    patch_config = {
        "data_list": args.data_list, "base_dir": ROOT,
        "device": args.device
    }
    source_dataset = traffic_dataset(
        data_args, task_args["source"], names, "source",
        test_data=args.test_dataset
    )
    target_dataset = traffic_dataset(
        data_args, task_args["target_few_shot"], names, "target_maml",
        test_data=args.test_dataset
    )
    model = STRep(
        0.5, args.daepoch, 1, 100, 0, 1, "SOFTS", "OT",
        data_args, task_args, model_args, patch_config, args
    )
    # PyTorch weights_only prevents arbitrary pickle execution from a
    # potentially untrusted checkpoint.
    checkpoint = torch.load(
        str(args.checkpoint), map_location=args.device, weights_only=True
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected Stage-I STRep checkpoint with model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.current_epoch = int(checkpoint.get("epoch", -1))
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    if checkpoint_epoch < 1500:
        raise ValueError("Checkpoint predates F2 OT activation")
    del checkpoint

    saved_params = {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
    }
    # Some network buffers update in train mode even without optimizer.step;
    # reset these after EACH probe to preserve the loaded checkpoint.
    saved_buffers = {
        name: b.detach().clone() for name, b in model.named_buffers()
    }
    print("F2_DIAGNOSTIC_CHECKPOINT", args.checkpoint,
          "checkpoint_training_epoch", checkpoint_epoch,
          "probe_batches", args.probe_batches,
          "target_set", "three-day-support-only",
          "training_updates", 0, flush=True)

    # Save machine-readable diagnostics directly, while also preserving
    # the original verbose stdout for troubleshooting.
    record_count = 0
    with args.output.open("w", encoding="utf-8") as output_stream:
      for iteration in range(args.probe_batches):
        data, matrices = source_target_batch(
            source_dataset, target_dataset, n_sources=3
        )
        print("F2_DIAGNOSTIC_PROBE", iteration,
              "checkpoint_training_epoch", checkpoint_epoch,
              "virtual_trigger_epoch", args.daepoch + iteration,
              flush=True)
        collected_stdout = io.StringIO()
        with contextlib.redirect_stdout(collected_stdout):
            model.meta_train_revise(
                data, matrices, epoch=args.daepoch + iteration
            )
        verbose_text = collected_stdout.getvalue()
        sys.stdout.write(verbose_text)
        sys.stdout.flush()
        records_this_batch = 0
        for line in verbose_text.splitlines():
            prefix = "f2_d1d2_audit_json "
            if line.startswith(prefix):
                item = json.loads(line[len(prefix):])
                item["probe_batch_index"] = iteration
                item["checkpoint_training_epoch"] = checkpoint_epoch
                item["training_updates"] = 0
                output_stream.write(
                    json.dumps(item, sort_keys=True, allow_nan=False) + "\n"
                )
                records_this_batch += 1
        output_stream.flush()
        if records_this_batch != 3:
            raise RuntimeError(
                "Expected exactly 3 audited source cities, got %d" %
                records_this_batch
            )
        record_count += records_this_batch
        with torch.no_grad():
            for name, b in model.named_buffers():
                b.copy_(saved_buffers[name])
        verify_unchanged(model, saved_params)
    print("F2_DIAGNOSTIC_FINISHED",
          "records", record_count, "jsonl", args.output,
          "probe_batches", args.probe_batches,
          "checkpoint_training_epoch", checkpoint_epoch,
          "training_updates", 0, flush=True)


if __name__ == "__main__":
    main()
