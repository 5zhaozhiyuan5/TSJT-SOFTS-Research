"""Apply target-three-day-only normalization to official MTPB datasets.

Upstream MTPB calculates PEMS-BAY speed mean/std before separating target
train and test. This conversion prevents target future-label statistics from
informing the model, while leaving prediction targets in original units.
"""
from __future__ import annotations
import numpy as np


def apply_train_only_target_normalization(
    target_train, target_test, raw_path,
    target_city='pems-bay', train_days=3,
):
    raw = np.load(raw_path, mmap_mode='r')
    if raw.ndim != 3:
        raise ValueError('expected [T,N,F] MTPB dataset_expand.npy')
    length = int(train_days)*288
    if length > raw.shape[0]:
        raise ValueError('target training window exceeds dataset length')
    train_speed = np.asarray(raw[:length,:,0],dtype=np.float64)
    mean = float(train_speed.mean())
    std = float(train_speed.std())
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 1e-12:
        raise ValueError('invalid target train-only statistics')
    for dataset in (target_train,target_test):
        previous_mean = float(np.asarray(
            dataset.means_list[target_city]).reshape(-1)[0])
        previous_std = float(np.asarray(
            dataset.stds_list[target_city]).reshape(-1)[0])
        if previous_std <= 0:
            raise ValueError('invalid upstream target standard deviation')
        x = dataset.x_list[target_city]
        x[...,0] = ((x[...,0]*previous_std+previous_mean)-mean)/std
        dataset.means_list[target_city] = np.array([mean])
        dataset.stds_list[target_city] = np.array([std])
    return {'mean':mean,'std':std,'train_days':train_days}
