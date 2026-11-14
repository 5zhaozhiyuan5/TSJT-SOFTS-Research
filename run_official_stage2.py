"""Run the unmodified PMME paper Stage-II on a saved Stage-I checkpoint."""

from __future__ import annotations

import argparse
import json
import random
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
import PMME_2ndStage


PAPER_HORIZONS = [1, 11, 23, 35]
PAPER_REFERENCE = {
    "rmse": [5.05, 8.16, 9.30, 9.77],
    "mae": [2.71, 4.05, 4.69, 5.02],
}


def log(handle, message):
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


def load_stage1_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    old_state = checkpoint["model_state_dict"]
    source_state = {
        key[len("model."):]: value
        for key, value in old_state.items()
        if key.startswith("model.")
    }
    target_state = model.model.state_dict()
    filtered = {
        key: value
        for key, value in source_state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    target_state.update(filtered)
    model.model.load_state_dict(target_state)
    return checkpoint, sorted(filtered)


def aggregate_metrics(model, dataset, pred_num):
    total = [0 for _ in range(pred_num)]
    mse_sum = [0.0 for _ in range(pred_num)]
    mae_sum = [0.0 for _ in range(pred_num)]
    mape_sum = [0.0 for _ in range(pred_num)]
    with torch.no_grad():
        for index in range(len(dataset)):
            data_i, _matrix = dataset[index]
            if data_i.x.shape[0] == 0:
                continue
            out, y, _rep, _unscaled, _memory = model.model(data_i, stage="test")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", default="./configs/config.yaml")
    parser.add_argument("--test_dataset", default="metr-la")
    parser.add_argument("--data_list", default="chengdu_shenzhen_pems")
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--train_epochs", type=int, default=6000)
    parser.add_argument("--target_days", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--classifier", type=float, default=0.3)
    parser.add_argument("--DAepoch", type=int, default=1500)
    parser.add_argument("--DAcoef", type=float, default=0.01)
    parser.add_argument("--xySquaredRatio", type=float, default=100.0)
    parser.add_argument("--model_name", default="SOFTS")
    parser.add_argument("--STmodel", default="SOFTS")
    parser.add_argument("--MemSize", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    args.save_dir = save_dir
    args.device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    args.curr_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    with open(args.config_filename, "r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    config["task"]["source"]["train_epochs"] = 0
    config["task"]["source"]["finetune_epochs"] = args.train_epochs
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
    model = PMME_2ndStage.STRep(
        args.classifier,
        args.DAepoch,
        args.DAcoef,
        args.xySquaredRatio,
        args.MemSize,
        args.temperature,
        args.model_name,
        data_args,
        task_args,
        model_args,
        patch_cfg,
        args,
    ).to(args.device)
    stage1_checkpoint, loaded = load_stage1_weights(
        model, args.stage1_checkpoint, args.device
    )

    best_score = float("inf")
    best_epoch = -1
    best_path = save_dir / "stage2_best.pth"
    log_path = save_dir / "log.txt"
    with open(log_path, "w", encoding="utf-8") as log_handle:
        log(log_handle, f"version={args.version}")
        log(log_handle, "official_stage2=true")
        log(log_handle, f"stage1_checkpoint={args.stage1_checkpoint}")
        log(log_handle, f"loaded_stage1_parameters={len(loaded)}")
        log(log_handle, json.dumps(vars(args), default=str, ensure_ascii=False))

        for epoch in range(args.train_epochs):
            data, matrices = make_stage2_batch(support_dataset, source_dataset)
            loss, _mse, rmse, mae, _mape = model.meta_train_revise(
                data, matrices, epoch
            )
            if epoch % 25 == 0:
                log(
                    log_handle,
                    "epoch={} loss={:.6f} source_mae={:.6f} source_rmse={:.6f}".format(
                        epoch, float(loss), float(mae), float(rmse)
                    ),
                )
            if epoch == 0 or (epoch + 1) % 100 == 0:
                support_metrics = aggregate_metrics(
                    model, support_dataset, task_args["source"]["pred_num"]
                )
                selection_score = float(
                    np.mean([support_metrics[index]["mae"] for index in PAPER_HORIZONS])
                )
                if selection_score < best_score:
                    best_score = selection_score
                    best_epoch = epoch
                    torch.save(
                        {
                            "version": args.version,
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": model.main_optim.state_dict(),
                            "selection_metric": "target_support_paper4_mae",
                            "selection_score": selection_score,
                            "stage1_checkpoint": str(args.stage1_checkpoint),
                        },
                        best_path,
                    )
                    log(
                        log_handle,
                        f"best_stage2 epoch={epoch} support_score={selection_score:.6f}",
                    )

        if not best_path.exists():
            torch.save(
                {
                    "version": args.version,
                    "epoch": args.train_epochs - 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": model.main_optim.state_dict(),
                    "selection_metric": "target_support_paper4_mae",
                    "selection_score": best_score,
                    "stage1_checkpoint": str(args.stage1_checkpoint),
                },
                best_path,
            )
            best_epoch = args.train_epochs - 1

        best_state = torch.load(best_path, map_location=args.device)
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
            model, test_dataset, task_args["source"]["pred_num"]
        )
        result = {
            "version": args.version,
            "official_stage2": True,
            "stage1_checkpoint": str(args.stage1_checkpoint),
            "stage1_checkpoint_epoch": int(stage1_checkpoint.get("epoch", -1)),
            "stage2_best_epoch": int(best_epoch),
            "stage2_best_support_score": float(best_score),
            "target_dataset": args.test_dataset,
            "source_datasets": [str(item) for item in source_dataset.data_list],
            "paper_horizon_indices": PAPER_HORIZONS,
            "paper_reference": PAPER_REFERENCE,
            "paper_metrics": [metrics[index] for index in PAPER_HORIZONS],
            "metrics_all_horizons": metrics,
            "config": vars(args),
        }
        with open(save_dir / "stage2_metrics_seed1.json", "w", encoding="utf-8") as result_handle:
            json.dump(result, result_handle, ensure_ascii=False, indent=2, default=str)
        log(log_handle, json.dumps({"paper_metrics": result["paper_metrics"]}, ensure_ascii=False))
        log(log_handle, f"result={save_dir / 'stage2_metrics_seed1.json'}")


if __name__ == "__main__":
    main()
