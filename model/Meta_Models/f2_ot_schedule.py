"""F2 Stage-I OT-strength schedule; no new parameters and no change to OT marginals.

t0: original fixed F2 OT strength (strict baseline).
t1: keep F2 OT strength until decay_start; linearly anneal to ZERO at decay_end.
t2: same schedule but linearly anneal to 25% of original OT strength.
The helper is independent of torch and is easy to test on CPU.
"""
import math


SCHEDULES = ("t0", "t1", "t2")


def validate_schedule(mode, base_scale, daepoch, decay_start, decay_end,
                      tail_fraction=0.25):
    if mode not in SCHEDULES:
        raise ValueError("PMME_F2_OT_SCHEDULE must be t0, t1 or t2")
    if not math.isfinite(base_scale) or base_scale <= 0.0:
        raise ValueError("Base OT scale must be positive and finite")
    if not isinstance(daepoch, int) or daepoch < 0:
        raise ValueError("OT activation epoch must be a nonnegative integer")
    if not isinstance(decay_start, int) or not isinstance(decay_end, int):
        raise ValueError("Decay epochs must be integers")
    if not 0 <= decay_start < decay_end:
        raise ValueError("Require 0 <= decay_start < decay_end")
    # T0 is the untouched fixed-F2 baseline. It must also remain valid for
    # unrelated legacy workflows where DAepoch can be larger than 2999.
    if mode != "t0" and daepoch > decay_start:
        raise ValueError("For T1/T2, OT activation must precede decay start")
    if not math.isfinite(tail_fraction) or not 0.0 <= tail_fraction <= 1.0:
        raise ValueError("T2 tail fraction must be in [0,1]")


def effective_ot_scale(epoch, *, mode, base_scale, daepoch, decay_start,
                       decay_end, tail_fraction=0.25):
    """Return current Stage-I OT loss scale (excluding DAcoef).

    At epoch < daepoch, OT is OFF in the original PMME Stage-I too.
    At epoch=decay_start, T0/T1/T2 have exactly the same scale.
    At epoch>=decay_end, T1 is exactly 0 and T2 is 0.25*base_scale.
    """
    validate_schedule(mode, base_scale, daepoch, decay_start, decay_end,
                      tail_fraction)
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Epoch must be a nonnegative integer")
    if epoch < daepoch:
        return 0.0
    if mode == "t0" or epoch <= decay_start:
        return float(base_scale)
    progress = min(1.0, (epoch - decay_start) / (decay_end - decay_start))
    fraction = 0.0 if mode == "t1" else tail_fraction
    return float(base_scale * (1.0 - (1.0 - fraction) * progress))


def phase(epoch, *, mode, daepoch, decay_start, decay_end):
    if epoch < daepoch:
        return "off_before_daepoch"
    if mode == "t0":
        return "fixed"
    if epoch < decay_start:
        return "pre_decay"
    if epoch >= decay_end:
        return "tail"
    return "linear_decay"
