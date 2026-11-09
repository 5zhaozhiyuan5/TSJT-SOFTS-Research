"""Training-only target normalization is invariant to test-period values."""
import numpy as np
import torch
from tsjt_pgrad.mtpb.train_only_norm import apply_train_only_target_normalization


class TinyTarget:
    def __init__(self):
        self.x_list={'pems-bay':torch.ones((2,2,288,2))*3.}
        self.means_list={'pems-bay':np.array([7.])}
        self.stds_list={'pems-bay':np.array([2.])}


def test_normalization_uses_first_three_days_only(tmp_path):
    raw=np.ones((1000,2,1),dtype=np.float32)*5
    raw[:864,0,0]=np.arange(864)%9
    raw[864:]=999999.
    path=tmp_path/'dataset_expand.npy'
    np.save(path,raw)
    train=TinyTarget();test=TinyTarget()
    before=train.x_list['pems-bay'].clone()
    result=apply_train_only_target_normalization(train,test,path)
    assert result['mean']<20
    assert result['mean']==train.means_list['pems-bay'][0]
    assert result['mean']==test.means_list['pems-bay'][0]
    assert not torch.equal(before,train.x_list['pems-bay'])
    raw[864:]=-999999.
    np.save(path,raw)
    train2=TinyTarget();test2=TinyTarget()
    again=apply_train_only_target_normalization(train2,test2,path)
    assert again==result
    assert torch.equal(train2.x_list['pems-bay'],train.x_list['pems-bay'])
