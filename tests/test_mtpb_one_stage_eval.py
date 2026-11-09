"""One-stage target evaluation: no optimizer and no target fine-tuning."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

_SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_mtpb_one_stage.py'
_SPEC=importlib.util.spec_from_file_location('mtpb_one_stage_runner',_SCRIPT)
runner=importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


class ToyPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta=torch.nn.Parameter(torch.tensor(2.))
        self.register_buffer('counter',torch.tensor(0))
    def forward(self,data,graph,stage='test'):
        x=data.x[...,0]
        pred=(x[:,:,-1]*self.theta).unsqueeze(-1).expand(-1,-1,36)
        return pred,data.y,None


class ToyRep:
    device='cpu'
    def __init__(self):
        self.model=ToyPredictor()
    def finetuning(self,*args):
        raise AssertionError('target finetuning is forbidden')


class ToyTargetTest:
    def __len__(self):
        return 2
    def __getitem__(self,i):
        x=torch.ones(3,4,288,2)*(i+1)
        y=torch.ones(3,4,36)*(i+1)
        return SimpleNamespace(x=x,y=y),torch.eye(4)


def test_direct_target_evaluation_does_not_finetune():
    rep=ToyRep()
    params=[p.detach().clone() for p in rep.model.parameters()]
    result=runner.evaluate_without_adaptation(rep,ToyTargetTest())
    assert result['protocol']=='direct_target_test_no_finetune'
    assert result['target_finetune_epochs']==0
    assert result['test_node_window_pairs']==24
    assert len(result['mae_by_horizon'])==36
    assert len(result['rmse_by_horizon'])==36
    assert result['overall_mae'] == 1.5
    assert result['overall_rmse'] > 0
    assert all(torch.equal(a,b) for a,b in zip(params,rep.model.parameters()))


def test_direct_target_eval_rejects_weight_mutation():
    rep=ToyRep()
    original=rep.model.forward
    def bad(data,graph,stage='test'):
        with torch.no_grad():
            rep.model.theta.add_(1)
        return original(data,graph,stage)
    rep.model.forward=bad
    try:
        runner.evaluate_without_adaptation(rep,ToyTargetTest())
    except AssertionError as e:
        assert 'modified learned model parameters' in str(e)
    else:
        raise AssertionError('mutation must be detected')


def test_runner_never_calls_finetuning():
    body=_SCRIPT.read_text(encoding='utf-8')
    assert 'rep.finetuning(' not in '\n'.join(
        l for l in body.splitlines() if not l.strip().startswith('#'))
    assert 'if not args.skip_eval:' in body
    assert 'evaluate_without_adaptation(rep,target_test)' in body
