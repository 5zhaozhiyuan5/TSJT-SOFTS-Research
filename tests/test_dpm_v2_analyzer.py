"""Verify post-run DPM-V2 log analyzer with a real filesystem fixture."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestDPMV2Analyzer(unittest.TestCase):
    def test_logs_to_csv_and_json(self):
        root=Path(__file__).resolve().parents[1]
        analyzer=root/"summarize_dpm_v2_diagnostics.py"
        with tempfile.TemporaryDirectory() as folder:
            run_root=Path(folder)
            run=run_root/"v2b"/"seed1"
            run.mkdir(parents=True)
            (run/"COMPLETED").touch()
            (run/"run_config.txt").write_text(
                "mode=v2b\ncommit=abc123\nseed=1\ndpm_strength=0.35\n",
                encoding="utf-8"
            )
            city={
                "event":"dpm_v2_attention_cost","epoch":1500,
                "city":"metr-la","target_group":0,"mode":"v2b",
                "strength":.35,"source_count":16,"target_count":12,
                "sample_bonus_mean":.4,"sample_bonus_std":.2,
                "sample_extra_mean":.1,"sample_source_rank_change":.05,
                "subset_sinkhorn":{"epsilon":.1,"transport_l1_change":.03}
            }
            bank={
                "event":"dpm_v2_bank_update","epoch":1500,"step":1,
                "cities":3,"momentum":.08,
                "q_band_before":[.5,.5,.5],
                "q_band_after":[.51,.50,.49],
                "r_band_before":[1,1,1],
                "r_band_after":[.98,.99,.97]
            }
            lines=[
                "DPM_V2_DIAG "+json.dumps(city),
                "DPM_V2_BANK_UPDATE "+json.dumps(bank),
            ]
            (run/"train.out").write_text("\n".join(lines)+"\n",encoding="utf-8")
            result=subprocess.run(
                [sys.executable,str(analyzer),"--run-root",str(run_root)],
                text=True,capture_output=True
            )
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue((run_root/"dpm_v2_pair_cost_diagnostics.csv").is_file())
            self.assertTrue((run_root/"dpm_v2_bank_update_history.csv").is_file())
            report=json.loads((run_root/"dpm_v2_analysis.json").read_text())
            self.assertEqual(report["runs"][0]["mode"],"v2b")
            self.assertEqual(report["runs"][0]["pair_cost_records"],1)
            self.assertEqual(report["runs"][0]["bank_update_records"],1)
            self.assertEqual(report["by_source_city"][0]["city"],"metr-la")


if __name__=="__main__":
    unittest.main()
