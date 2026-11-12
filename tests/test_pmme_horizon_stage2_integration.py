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


class PmmeTargetResidualTemporalIntegrationTest(unittest.TestCase):
    def test_heldout_test_evaluation_is_deferred_until_after_stage2(self):
        runner = (Path(ROOT) / "train_DA_OT.py").read_text(encoding="utf-8")
        adaptation_start = runner.index(
            "for i in range(task_args['source']['finetune_epochs']):"
        )
        test_dataset_creation = runner.index("test_dataset = traffic_dataset")
        self.assertGreater(test_dataset_creation, adaptation_start)
        self.assertNotIn(".test_evaluation(", runner[:adaptation_start])
        self.assertEqual(runner.count(".test_evaluation("), 2)
        self.assertNotIn("initialize_horizon_memory", runner)
        self.assertIn("TargetResidualTemporal", runner)
        self.assertIn("[Stage-II][Epoch {}/{}", runner)
        self.assertIn('mean_epoch_component("residual_loss")', runner)
        for launcher in (
            "train_one_OT-pems.sh",
            "train_one_OT-chengdu.sh",
            "train_one_OT-metr.sh",
            "train_one_OT-shenzhen.sh",
        ):
            launcher_text = (Path(ROOT) / launcher).read_text(encoding="utf-8")
            self.assertIn("finetune_epochs=300", launcher_text)
            self.assertNotIn("finetune_epochs=6000", launcher_text)

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
                stage2_residual_hidden=32,
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
            self.assertTrue(
                all(not p.requires_grad for p in trainer.model.forecasting_model.parameters())
            )
            self.assertTrue(
                all(p.requires_grad for p in trainer.model.residual_net.parameters())
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
            residual_before = {
                name: value.detach().clone()
                for name, value in trainer.model.residual_net.state_dict().items()
            }

            stats = trainer.meta_train_revise(
                target_data, epoch=0, target=True
            )

            self.assertEqual(len(stats), 5)
            self.assertTrue(
                all(torch.isfinite(torch.as_tensor(value)).all() for value in stats)
            )
            loss_components = trainer.last_stage2_loss_components
            self.assertIsNotNone(loss_components)
            for key in (
                "total_loss",
                "residual_loss",
                "forecast_loss",
                "correction_l2",
                "grad_norm",
            ):
                self.assertTrue(np.isfinite(loss_components[key]))
            self.assertAlmostEqual(
                float(np.asarray(stats[0]).item()),
                loss_components["total_loss"],
                places=5,
            )
            self.assertTrue(any(
                not torch.equal(residual_before[name], value.detach())
                for name, value in trainer.model.residual_net.state_dict().items()
            ))
            for name, value in trainer.model.forecasting_model.state_dict().items():
                self.assertTrue(torch.equal(backbone_before[name], value.detach()))


if __name__ == "__main__":
    unittest.main()
