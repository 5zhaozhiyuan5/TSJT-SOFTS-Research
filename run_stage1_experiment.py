"""Versioned Stage-I experiments for the PMME research branch.

This runner evaluates the official PMME Stage-I and the research Stage-I with
the same source/target batches.  It never constructs the held-out target test
dataset until the Stage-I checkpoint has been selected.  Stage-II is run by
``run_official_stage2.py`` and is intentionally not imported here.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from datasets import traffic_dataset
from utils import calc_metric, get_data_list

sys.path.append("./model")
sys.path.append("./model/Meta_Models")
import PMME
from stage1_innovation import FrequencyPrototypeStage1


PAPER_HORIZONS = [1, 11, 23, 35]
PAPER_REFERENCE = {
    "rmse": [5.05, 8.16, 9.30, 9.77],
    "mae": [2.71, 4.05, 4.69, 5.02],
}


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def log(handle, message: str) -> None:
    print(message, flush=True)
    handle.write(str(message) + "\n")
    handle.flush()


def make_stage2_batch(target_dataset, source_dataset):
    target_data, target_matrix = target_dataset[0]
    data = [target_data]
    matrices = [target_matrix]
    used_shapes = {target_matrix.shape[1]}
    source_index = 0
    while len(data) < 4:
        source_data, source_matrix = source_dataset[source_index]
        source_index += 1
        if source_matrix.shape[1] in used_shapes:
            continue
        used_shapes.add(source_matrix.shape[1])
        data.append(source_data)
        matrices.append(source_matrix)
    return data, matrices


def aggregate_metrics(container, dataset, device, pred_num):
    total = [0 for _ in range(pred_num)]
    mse_sum = [0.0 for _ in range(pred_num)]
    mae_sum = [0.0 for _ in range(pred_num)]
    mape_sum = [0.0 for _ in range(pred_num)]
    container.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            data_i, _matrix = dataset[index]
            if data_i.x.shape[0] == 0:
                continue
            out, y, _rep, _unscaled, _memory = container.model(
                data_i, stage="test"
            )
            mse, _rmse, mae, mape = calc_metric(out, y, stage="test")
            batch_size = int(out.shape[0])
            for horizon in range(pred_num):
                total[horizon] += batch_size
                mse_sum[horizon] += float(mse[horizon]) * batch_size
                mae_sum[horizon] += float(mae[horizon]) * batch_size
                mape_sum[horizon] += float(mape[horizon]) * batch_size
    metrics = []
    for horizon in range(pred_num):
        mse_value = mse_sum[horizon] / max(total[horizon], 1)
        metrics.append(
            {
                "horizon_index": horizon,
                "mse": mse_value,
                "rmse": float(np.sqrt(mse_value)),
                "mae": mae_sum[horizon] / max(total[horizon], 1),
                "mape": mape_sum[horizon] / max(total[horizon], 1),
            }
        )
    return metrics


def build_models(args, data_args, task_args, model_args, patch_cfg):
    if args.stage1_method == "pmme":
        return PMME.STRep(
            args.classifier,
            args.DAepoch,
            args.DAcoef,
            args.xySquaredRatio,
            0,
            args.projection,
            args.model_name,
            args.DA,
            data_args,
            task_args,
            model_args,
            patch_cfg,
            args,
        )
    backbone = PMME.PatchFSL(
        data_args,
        model_args,
        task_args,
        0,
        0,
        args.model_name,
        patch_cfg,
        args,
    )
    return FrequencyPrototypeStage1(
        backbone=backbone,
        feature_dim=128,
        horizon=int(task_args["source"]["pred_num"]),
        prototype_count=args.prototype_count,
        route_temperature=args.route_temperature,
        uniform_route_mix=args.uniform_route_mix,
        target_weight=args.target_weight,
        frequency_weight=args.frequency_weight,
        trend_weight=args.trend_weight,
        prototype_weight=args.prototype_weight,
        route_smooth_weight=args.route_smooth_weight,
        learning_rate=model_args["STnet"]["lr"],
    )


def checkpoint_state(model, args, epoch, selection_score, selection_metric):
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        optimizer = getattr(model, "main_optim", None)
    return {
        "version": args.version,
        "epoch": int(epoch),
        "stage1_method": args.stage1_method,
        "selection_metric": selection_metric,
        "selection_score": float(selection_score),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "git_revision": git_revision(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", default="./configs/config.yaml")
    parser.add_argument("--test_dataset", default="metr-la")
    parser.add_argument("--data_list", default="chengdu_shenzhen_pems")
    parser.add_argument("--target_days", type=int, default=3)
    parser.add_argument("--train_epochs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--version", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--stage1_method", choices=["pmme", "fpp"], default="pmme")
    parser.add_argument("--stage1_selection_metric", choices=["source_mae", "target_support_mae"], default="source_mae")
    parser.add_argument("--model_name", default="SOFTS")
    parser.add_argument("--STmodel", default="SOFTS")
    parser.add_argument("--classifier", type=float, default=0.3)
    parser.add_argument("--DAepoch", type=int, default=1500)
    parser.add_argument("--DAcoef", type=float, default=0.01)
    parser.add_argument("--DAwarmup", type=int, default=500)
    parser.add_argument("--xySquaredRatio", type=float, default=100.0)
    parser.add_argument("--projection", type=float, default=1.0)
    parser.add_argument("--DA", default="OT")
    parser.add_argument("--prototype_count", type=int, default=16)
    parser.add_argument("--route_temperature", type=float, default=2.0)
    parser.add_argument("--uniform_route_mix", type=float, default=0.35)
    parser.add_argument("--target_weight", type=float, default=0.10)
    parser.add_argument("--frequency_weight", type=float, default=0.05)
    parser.add_argument("--trend_weight", type=float, default=0.03)
    parser.add_argument("--prototype_weight", type=float, default=0.01)
    parser.add_argument("--route_smooth_weight", type=float, default=0.05)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    args.save_dir = save_dir
    args.device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    args.new = 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    with open(args.config_filename, "r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    config["task"]["source"]["train_epochs"] = args.train_epochs
    config["task"]["source"]["finetune_epochs"] = 0
    data_args = config["data"]
    task_args = config["task"]
    model_args = config["model"]
    data_list = get_data_list(args.data_list)
    patch_cfg = {
        "data_list": args.data_list,
        "base_dir": Path(sys.path[0]),
        "device": args.device,
    }

    source_dataset = traffic_dataset(
        data_args,
        task_args["source"],
        data_list.copy(),
        "source",
        test_data=args.test_dataset,
        add_target=False,
        target_days=args.target_days,
    )
    support_dataset = traffic_dataset(
        data_args,
        task_args["target_few_shot"],
        data_list.copy(),
        "target_maml",
        test_data=args.test_dataset,
        target_days=args.target_days,
    )

    model = build_models(args, data_args, task_args, model_args, patch_cfg).to(args.device)
    best_score = float("inf")
    best_epoch = -1
    checkpoint_path = save_dir / "stage1_best.pth"
    log_path = save_dir / "log.txt"
    diagnostics_path = save_dir / "diagnostics.jsonl"

    with open(log_path, "w", encoding="utf-8") as log_handle:
        log(log_handle, f"version={args.version}")
        log(log_handle, f"stage1_method={args.stage1_method}")
        log(log_handle, f"git_revision={git_revision()}")
        log(log_handle, f"source_datasets={list(source_dataset.data_list)}")
        log(log_handle, f"target_dataset={args.test_dataset}")
        log(log_handle, json.dumps(json_safe(vars(args)), ensure_ascii=False))
        with open(save_dir / "config.json", "w", encoding="utf-8") as config_handle:
            json.dump(
                {
                    "args": json_safe(vars(args)),
                    "config": json_safe(config),
                    "paper_horizons": PAPER_HORIZONS,
                    "paper_reference": PAPER_REFERENCE,
                },
                config_handle,
                ensure_ascii=False,
                indent=2,
            )

        for epoch in range(args.train_epochs):
            data, matrices = make_stage2_batch(support_dataset, source_dataset)
            start = time.time()
            if args.stage1_method == "pmme":
                loss, _mse, _rmse, source_mae, _mape = model.meta_train_revise(
                    data, matrices, epoch
                )
                train_record = {
                    "loss": float(loss),
                    "source_mae": float(source_mae),
                    "target_mae": None,
                }
            else:
                model.train()
                train_record = model.train_step(data, epoch)
                with open(diagnostics_path, "a", encoding="utf-8") as diag_handle:
                    diag_handle.write(
                        json.dumps(model.last_diagnostics, ensure_ascii=False) + "\n"
                    )

            if epoch % 25 == 0:
                log(
                    log_handle,
                    "epoch={} loss={:.6f} source_mae={:.6f} seconds={:.3f}".format(
                        epoch,
                        train_record["loss"],
                        train_record["source_mae"],
                        time.time() - start,
                    ),
                )

            if epoch == 0 or (epoch + 1) % 100 == 0:
                if args.stage1_selection_metric == "target_support_mae":
                    support_metrics = aggregate_metrics(
                        model, support_dataset, args.device, task_args["source"]["pred_num"]
                    )
                    selection_score = float(
                        np.mean([support_metrics[index]["mae"] for index in PAPER_HORIZONS])
                    )
                else:
                    selection_score = float(train_record["source_mae"])
                if selection_score < best_score:
                    best_score = selection_score
                    best_epoch = epoch
                    torch.save(
                        checkpoint_state(
                            model,
                            args,
                            epoch,
                            selection_score,
                            args.stage1_selection_metric,
                        ),
                        checkpoint_path,
                    )
                    log(
                        log_handle,
                        f"best_stage1 epoch={epoch} score={selection_score:.6f}",
                    )

        if not checkpoint_path.exists():
            torch.save(
                checkpoint_state(
                    model, args, args.train_epochs - 1, best_score,
                    args.stage1_selection_metric,
                ),
                checkpoint_path,
            )
            best_epoch = args.train_epochs - 1

        best_state = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(best_state["model_state_dict"])
        test_dataset = traffic_dataset(
            data_args,
            task_args["source"],
            data_list.copy(),
            "test",
            test_data=args.test_dataset,
            target_days=args.target_days,
        )
        metrics = aggregate_metrics(
            model, test_dataset, args.device, task_args["source"]["pred_num"]
        )
        result = {
            "version": args.version,
            "stage1_method": args.stage1_method,
            "seed": args.seed,
            "target_dataset": args.test_dataset,
            "source_datasets": [str(item) for item in source_dataset.data_list],
            "stage1_best_epoch": int(best_epoch),
            "stage1_best_selection_score": float(best_score),
            "stage1_checkpoint": str(checkpoint_path),
            "paper_horizon_indices": PAPER_HORIZONS,
            "paper_reference": PAPER_REFERENCE,
            "paper_metrics": [metrics[index] for index in PAPER_HORIZONS],
            "metrics_all_horizons": metrics,
            "git_revision": git_revision(),
        }
        with open(save_dir / "stage1_metrics_seed1.json", "w", encoding="utf-8") as result_handle:
            json.dump(result, result_handle, ensure_ascii=False, indent=2)
        log(log_handle, json.dumps({"paper_metrics": result["paper_metrics"]}, ensure_ascii=False))
        log(log_handle, f"result={save_dir / 'stage1_metrics_seed1.json'}")


if __name__ == "__main__":
    main()
