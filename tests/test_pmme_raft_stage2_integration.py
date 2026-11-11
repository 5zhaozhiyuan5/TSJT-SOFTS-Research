import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_MODEL_DIR = os.path.join(ROOT, "model", "Meta_Models")
sys.path.insert(0, ROOT)
sys.path.insert(0, META_MODEL_DIR)

import PMME_2ndStage


def make_city(mean, std, samples=4, nodes=2, horizon=36):
    # Channel 0 is globally standardized speed; channel 1 is an unused time feature.
    history = torch.randn(samples, nodes, 288)
    future_global = history[:, :, -1:] + 0.15 * torch.randn(samples, nodes, horizon)
    future_raw = future_global * std + mean
    x = torch.stack([history, torch.zeros_like(history)], dim=-1)
    return x, future_raw


class PmmeRaftStage2IntegrationTest(unittest.TestCase):
    def test_bank_build_and_one_real_stage2_optimizer_step(self):
        torch.manual_seed(11)
        source_x_a, source_y_a = make_city(60.0, 10.0)
        source_x_b, source_y_b = make_city(30.0, 8.0)

        source_dataset = SimpleNamespace(
            data_list=["source_a", "source_b"],
            x_list={"source_a": source_x_a, "source_b": source_x_b},
            y_list={"source_a": source_y_a, "source_b": source_y_b},
            means_list={"source_a": [60.0], "source_b": [30.0]},
            stds_list={"source_a": [10.0], "source_b": [8.0]},
        )

        with tempfile.TemporaryDirectory() as save_dir:
            args = SimpleNamespace(
                test_dataset="target",
                STmodel="SOFTS",
                save_dir=save_dir,
            )
            data_args = {
                "data_keys": ["source_a", "source_b", "target"],
            }
            task_args = {
                "source": {"pred_num": 36, "finetune_epochs": 1},
                "target_few_shot": {"batch_size": 1, "his_num": 288, "pred_num": 36},
            }
            model_args = {"STnet": {"lr": 1e-3}}
            patch_cfg = {"device": torch.device("cpu")}

            trainer = PMME_2ndStage.STRep(
                0.3, 50000, 0.1, 10.0,
                12, 1, "SOFTS",
                data_args, task_args, model_args, patch_cfg, args,
            )

            bank_count, city_counts = trainer.model.build_retrieval_bank(source_dataset)
            self.assertEqual(bank_count, 12)
            self.assertEqual(city_counts, {"source_a": 6, "source_b": 6})
            self.assertNotIn("target", trainer.model.raft_adapter.source_city_names)

            # Exercise the production PatchFSL.forward + STRep.meta_train_revise path.
            target_history = torch.randn(1, 3, 288)
            target_future = (
                target_history[:, :, -1:] + 0.1 * torch.randn(1, 3, 36)
            ) * 9.0 + 55.0
            target_data = SimpleNamespace(
                x=torch.stack([target_history, torch.zeros_like(target_history)], dim=-1),
                y=target_future,
                means=[55.0],
                stds=[9.0],
            )

            backbone_before = {
                name: value.detach().clone()
                for name, value in trainer.model.forecasting_model.state_dict().items()
            }
            fusion_before = trainer.model.raft_adapter.fusion.weight.detach().clone()

            stats = trainer.meta_train_revise(target_data, None, epoch=0)

            self.assertEqual(len(stats), 5)
            self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all() for value in stats))
            self.assertFalse(torch.equal(
                fusion_before,
                trainer.model.raft_adapter.fusion.weight.detach(),
            ))
            for name, value in trainer.model.forecasting_model.state_dict().items():
                self.assertTrue(torch.equal(backbone_before[name], value.detach()))

    def test_missing_target_args_is_not_silently_accepted(self):
        # The production builder reads args.test_dataset; PatchFSL must retain args.
        with tempfile.TemporaryDirectory() as save_dir:
            args = SimpleNamespace(test_dataset="target", STmodel="SOFTS", save_dir=save_dir)
            model = PMME_2ndStage.PatchFSL(
                {"data_keys": ["source_a", "target"]},
                {"STnet": {"lr": 1e-3}},
                {"source": {"pred_num": 36}},
                8, 1, "SOFTS", {"device": torch.device("cpu")}, args,
            )
            self.assertIs(model.args, args)


if __name__ == "__main__":
    unittest.main()
