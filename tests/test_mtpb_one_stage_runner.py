"""Static regression guards for the one-stage MTPB runner.

These tests do not train on real traffic data.
"""
from pathlib import Path
import ast

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_mtpb_integrated.py"


def test_one_stage_is_integrated_only_and_has_no_finetuning():
    code = RUNNER.read_text(encoding="utf-8")
    ast.parse(code)
    assert "if args.one_stage and args.mode != 'integrated':" in code
    assert "if args.one_stage:" in code
    assert "elif not args.skip_finetune:" in code
    assert "rep.finetuning(target_train,target_test,args.finetune_epochs)" in code
    assert "target_finetune_steps':0" in code


def test_direct_evaluation_is_read_only_and_checkpoint_precedes_test():
    code = RUNNER.read_text(encoding="utf-8")
    start = code.index("    if args.one_stage:\n",code.index("with open(args.output/'diagnostics.jsonl'"))
    end = code.index("    elif not args.skip_finetune:",start)
    block = code[start:end]
    assert block.index("torch.save(rep.model.state_dict(),checkpoint)") < block.index("rep.test_batch(")
    assert "with torch.no_grad():" in block
    assert "one_stage_test_metrics.json" in block
    assert ".backward(" not in block
    assert ".step(" not in block


def test_source_target_data_split_remains_explicit():
    code = RUNNER.read_text(encoding="utf-8")
    assert "add_target=False" in code
    assert "apply_train_only_target_normalization(" in code
    assert "target_days=3" in code
    assert "history=288" not in code or "'history':288" in code
