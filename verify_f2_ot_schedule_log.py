#!/usr/bin/env python3
"""Fail closed if Stage-I logged OT coefficients do not match T0/T1/T2.

The training run script invokes this verifier before writing COMPLETED.
"""
import argparse
import math
import re
from pathlib import Path

from model.Meta_Models.f2_ot_schedule import effective_ot_scale


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
MARK = re.compile(
    r"^f2_ot_schedule epoch (\d+) policy (t[012]) phase (\S+)"
    + r" base_scale (" + FLOAT + r")"
    + r" effective_scale (" + FLOAT + r")"
    + r" DAcoef (" + FLOAT + r")"
    + r" effective_ot_coefficient (" + FLOAT + r")"
)


def verify(text, *, mode, stage1, daepoch, decay_start, decay_end,
           base_scale, tail_fraction):
    if "Source training Ending!!!" not in text:
        raise ValueError("Stage-I completion marker missing")
    first_stage=text.split("Source training Ending!!!",1)[0]
    found={}
    for line in first_stage.splitlines():
        if not line.startswith("f2_ot_schedule "):
            continue
        m=MARK.match(line)
        if m is None:
            raise ValueError("Malformed schedule line: "+line)
        epoch=int(m.group(1))
        actual_mode=m.group(2)
        actual_base=float(m.group(4))
        actual_scale=float(m.group(5))
        dacoef=float(m.group(6))
        coefficient=float(m.group(7))
        if actual_mode != mode:
            raise ValueError("Mismatch schedule mode at epoch "+str(epoch))
        if epoch in found:
            raise ValueError("Duplicate OT schedule record for epoch "+str(epoch))
        if not math.isclose(actual_base,base_scale,rel_tol=1e-8,abs_tol=1e-10):
            raise ValueError("Base scale does not match run config")
        expected=effective_ot_scale(
            epoch,mode=mode,base_scale=base_scale,daepoch=daepoch,
            decay_start=decay_start,decay_end=decay_end,
            tail_fraction=tail_fraction
        )
        if not math.isclose(actual_scale,expected,rel_tol=1e-8,abs_tol=1e-10):
            raise ValueError(
                "Bad OT scale at epoch %d: got %.12g expected %.12g" %
                (epoch,actual_scale,expected)
            )
        if not math.isclose(coefficient,actual_scale*dacoef,
                            rel_tol=1e-8,abs_tol=1e-10):
            raise ValueError("Incorrect DAcoef * OT scale at epoch "+str(epoch))
        found[epoch]=actual_scale
    # Require exact transitions, not only a few easy-to-match log samples.
    required={0,daepoch,decay_start,decay_end,stage1-1}
    missing=sorted(required.difference(found))
    if missing:
        raise ValueError("Missing OT schedule records at epochs: "+str(missing))
    if mode == "t1" and found[decay_end] != 0.0:
        raise ValueError("T1 has not reached exact zero OT coefficient")
    if mode == "t2" and not math.isclose(
        found[decay_end],base_scale*tail_fraction,
        rel_tol=1e-8,abs_tol=1e-10
    ):
        raise ValueError("T2 did not reach 25% OT strength")
    if mode == "t0" and found[decay_end] != base_scale:
        raise ValueError("T0 original F2 OT strength changed")
    return found


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--log",type=Path,required=True)
    p.add_argument("--mode",choices=("t0","t1","t2"),required=True)
    p.add_argument("--stage1",type=int,required=True)
    p.add_argument("--daepoch",type=int,required=True)
    p.add_argument("--decay-start",type=int,required=True)
    p.add_argument("--decay-end",type=int,required=True)
    p.add_argument("--base-scale",type=float,required=True)
    p.add_argument("--tail-fraction",type=float,default=0.25)
    a=p.parse_args()
    found=verify(a.log.read_text(encoding="utf-8",errors="replace"),
                 mode=a.mode,stage1=a.stage1,daepoch=a.daepoch,
                 decay_start=a.decay_start,decay_end=a.decay_end,
                 base_scale=a.base_scale,tail_fraction=a.tail_fraction)
    print("OT_DECAY_VERIFIED",a.mode,"samples",len(found),
          "DAepoch",found[a.daepoch],"start",found[a.decay_start],
          "end",found[a.decay_end],"last",found[a.stage1-1])


if __name__=="__main__":
    main()
