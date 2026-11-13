import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_MODEL_DIR = os.path.join(ROOT, "model", "Meta_Models")
sys.path.insert(0, ROOT)
sys.path.insert(0, META_MODEL_DIR)

from datasets import traffic_dataset
import PMME_2ndStage


def write_synthetic_city(root, city, mean, std, length, nodes=2, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(length, dtype=np.float32)
    speed = (
        mean
        + 0.35 * std * np.sin(time[:, None] / 31.0)
        + 0.20 * std * np.sin(time[:, None] / 97.0)
        + std * rng.normal(size=(length, nodes)).astype(np.float32) * 0.25
    )
    values = np.zeros((length, nodes, 4), dtype=np.float32)
    values[:, :, 0] = speed
    values[:, :, -1] = (time[:, None] % 2016)
    dataset_path = os.path.join(root, f"{city}_dataset.npy")
    adjacency_path = os.path.join(root, f"{city}_matrix.npy")
    np.save(dataset_path, values)
    np.save(adjacency_path, np.ones((nodes, nodes), dtype=np.float32))
    return {
        "dataset_path": dataset_path,
        "adjacency_matrix_path": adjacency_path,
        "node_num": nodes,
        "time_step": length,
    }


class PmmeTargetResidualLowRankIntegrationTest(unittest.TestCase):
    def test_heldout_test_evaluation_is_deferred_until_after_stage2(self):
        runner = (Path(ROOT) / "train_DA_OT.py").read_text(encoding="utf-8")
        adaptation_start = runner.index(
            "adapter_stats = rep_model.fit_target_residual_adapter(finetune_dataset)"
        )
        test_dataset_creation = runner.index("test_dataset = traffic_dataset")
        self.assertGreater(test_dataset_creation, adaptation_start)
        self.assertNotIn(".test_evaluation(", runner[:adaptation_start])
        self.assertEqual(runner.count(".test_evaluation("), 2)
        self.assertNotIn("initialize_horizon_memory", runner)
        self.assertIn("LowRankResidualAdapter", runner)
        self.assertIn("[Stage-II][LowRankResidualAdapter] fit_complete", runner)
        self.assertIn("fit_target_residual_adapter(finetune_dataset)", runner)
        self.assertNotIn("total_stage2_epochs", runner)
        self.assertNotIn("[Stage-II][Epoch {}/{}", runner)
        for launcher in (
            "train_one_OT-pems.sh",
            "train_one_OT-chengdu.sh",
            "train_one_OT-metr.sh",
            "train_one_OT-shenzhen.sh",
        ):
            launcher_text = (Path(ROOT) / launcher).read_text(encoding="utf-8")
            self.assertIn("finetune_epochs=1", launcher_text)
            self.assertNotIn("finetune_epochs=300", launcher_text)
            self.assertNotIn("finetune_epochs=6000", launcher_text)

    def test_target_normalization_uses_only_three_day_support_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            target_path = write_synthetic_city(
                root, "target", 55.0, 9.0, 1300, seed=17
            )
            data_args = {
                "data_keys": ["target"],
                "target": target_path,
            }
            task_args = {
                "his_num": 288,
                "pred_num": 36,
                "batch_size": 46,
            }
            support_dataset = traffic_dataset(
                data_args, task_args, None, "target_maml",
                test_data="target", target_days=3,
            )
            heldout_dataset = traffic_dataset(
                data_args, task_args, None, "test",
                test_data="target", target_days=3,
            )
            raw = np.load(target_path["dataset_path"])
            expected_speed = raw[:864, :, 0].astype(np.float64)
            expected_mean = float(expected_speed.mean())
            expected_std = float(expected_speed.std())

            self.assertAlmostEqual(
                float(support_dataset.means_list["target"][0]),
                expected_mean,
                places=5,
            )
            self.assertAlmostEqual(
                float(support_dataset.stds_list["target"][0]),
                expected_std,
                places=5,
            )
            self.assertAlmostEqual(
                float(heldout_dataset.means_list["target"][0]),
                expected_mean,
                places=5,
            )
            self.assertAlmostEqual(
                float(heldout_dataset.stds_list["target"][0]),
                expected_std,
                places=5,
            )

    def test_stage1_pcgrad_training_path_is_preserved(self):
        torch.manual_seed(13)
        with tempfile.TemporaryDirectory() as root:
            data_args = {
                "data_keys": ["target"],
                "target": write_synthetic_city(root, "target", 55.0, 9.0, 864, seed=4),
            }
            task_args = {
                "source": {
                    "his_num": 288,
                    "pred_num": 36,
                    "batch_size": 16,
                    "task_num": 2,
                    "finetune_epochs": 1,
                },
                "target_few_shot": {
                    "his_num": 288,
                    "pred_num": 36,
                    "batch_size": 46,
                    "task_num": 2,
                },
            }
            target_dataset = traffic_dataset(
                data_args,
                task_args["target_few_shot"],
                ["target"],
                "target_maml",
                test_data="target",
                target_days=3,
            )
            args = SimpleNamespace(
                test_dataset="target",
                STmodel="SOFTS",
                save_dir=root,
                stage2_feature_bins=12,
                stage2_residual_rank=6,
                stage2_residual_smooth_rank=3,
                stage2_residual_ridge=10.0,
            )
            trainer = PMME_2ndStage.STRep(
                0.3, 50000, 0.1, 10.0,
                0, 0, "SOFTS",
                data_args, task_args, {"STnet": {"lr": 1e-3}},
                {"device": torch.device("cpu")}, args,
            )
            self.assertIsNone(trainer.model.residual_net)
            target_data = SimpleNamespace(
                x=target_dataset.x_list["target"][:2],
                y=target_dataset.y_list["target"][:2],
                means=target_dataset.means_list["target"],
                stds=target_dataset.stds_list["target"],
            )
            before = {
                name: value.detach().clone()
                for name, value in trainer.model.forecasting_model.state_dict().items()
            }
            stats = trainer.meta_train_revise(
                [target_data, target_data], epoch=0
            )
            self.assertEqual(len(stats), 5)
            self.assertTrue(
                all(torch.isfinite(torch.as_tensor(value)).all() for value in stats)
            )
            self.assertTrue(any(
                not torch.equal(before[name], value.detach())
                for name, value in trainer.model.forecasting_model.state_dict().items()
            ))

    def test_target_only_residual_network_trains_and_keeps_stage1_frozen(self):
        torch.manual_seed(11)
        with tempfile.TemporaryDirectory() as root:
            data_args = {
                "data_keys": ["source_a", "source_b", "target"],
                "source_a": write_synthetic_city(root, "source_a", 60.0, 10.0, 1200, seed=1),
                "source_b": write_synthetic_city(root, "source_b", 30.0, 8.0, 1200, seed=2),
                "target": write_synthetic_city(root, "target", 55.0, 9.0, 864, seed=3),
            }
            task_args = {
                "source": {
                    "his_num": 288,
                    "pred_num": 36,
                    "batch_size": 16,
                    "task_num": 2,
                    "finetune_epochs": 1,
                },
                "target_few_shot": {
                    "his_num": 288,
                    "pred_num": 36,
                    "batch_size": 46,
                    "task_num": 2,
                },
            }

            source_dataset = traffic_dataset(
                data_args,
                task_args["source"],
                ["source_a", "source_b"],
                "source",
                test_data="target",
                add_target=False,
                target_days=3,
            )
            target_dataset = traffic_dataset(
                data_args,
                task_args["target_few_shot"],
                ["source_a", "source_b"],
                "target_maml",
                test_data="target",
                target_days=3,
            )
            self.assertEqual(list(source_dataset.data_list), ["source_a", "source_b"])
            self.assertNotIn("target", source_dataset.data_list)
            self.assertEqual(target_dataset.x_list["target"].shape[0], 46)

            args = SimpleNamespace(
                test_dataset="target",
                STmodel="SOFTS",
                save_dir=root,
                stage2_residual_hidden=32,
            )
            model_args = {"STnet": {"lr": 1e-3}}
            patch_cfg = {"device": torch.device("cpu")}

            trainer = PMME_2ndStage.STRep(
                0.3, 50000, 0.1, 10.0,
                12, 1, "SOFTS",
                data_args, task_args, model_args, patch_cfg, args,
            )

            self.assertIsNotNone(trainer.model.residual_net)
            self.assertNotIn("adjacency", trainer.model.residual_net.forward.__code__.co_varnames)
            self.assertEqual(sum(p.numel() for p in trainer.model.residual_net.parameters()), 0)
            self.assertEqual(trainer.trainable_params, [])
            self.assertIsNone(trainer.main_optim)
            self.assertTrue(
                all(not p.requires_grad for p in trainer.model.forecasting_model.parameters())
            )

            target_data = SimpleNamespace(
                x=target_dataset.x_list["target"][:2],
                y=target_dataset.y_list["target"][:2],
                means=target_dataset.means_list["target"],
                stds=target_dataset.stds_list["target"],
            )
            backbone_before = {
                name: value.detach().clone()
                for name, value in trainer.model.forecasting_model.state_dict().items()
            }
            adapter_before = {
                name: value.detach().clone()
                for name, value in trainer.model.residual_net.state_dict().items()
            }

            fit_stats = trainer.fit_target_residual_adapter(target_dataset)
            self.assertTrue(trainer.model.residual_net.is_fitted)
            self.assertEqual(fit_stats["support_rows"], 92)
            self.assertEqual(fit_stats["feature_dim"], 40)
            self.assertEqual(fit_stats["rank"], 6)
            for key in (
                "total_loss",
                "residual_loss",
                "forecast_loss",
                "correction_l2",
                "train_MSE",
                "train_MAE",
                "train_RMSE",
                "fit_seconds",
                "captured_energy_ratio",
            ):
                self.assertTrue(np.isfinite(fit_stats[key]))

            out, y, _, _, correction = trainer.model(target_data, stage="test")
            self.assertEqual(tuple(out.shape), tuple(y.shape))
            self.assertEqual(tuple(correction.shape), tuple(y.shape))
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(any(
                not torch.equal(adapter_before[name], value.detach())
                for name, value in trainer.model.residual_net.state_dict().items()
            ))
            for name, value in trainer.model.forecasting_model.state_dict().items():
                self.assertTrue(torch.equal(backbone_before[name], value.detach()))


if __name__ == "__main__":
    unittest.main()
