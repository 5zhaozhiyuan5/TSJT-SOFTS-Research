# PMME F2 gradient marginal four methods - run logs

- Branch: `experiment/pmme-f2-gradient-marginal-four-20261126`
- Source commit: `46b6c4c56ee73d79fbc54448823ee725add403de`
- Run: batch 16, seed 1, target PEMS-BAY; methods `mag`, `sign`, `ema`, `biot` run sequentially.
- Stage I: 3000 epochs; Stage II: 6000 epochs; DAepoch 1500; OT scale 0.01.
- Existing server PMME environment; no server downloads or package installation.

Status at this snapshot: `mag/seed1` and `sign/seed1` completed Stage I and Stage II. `ema/seed1` reached Stage I epoch 2725/3000; `biot/seed1` is queued. Logs are gzip-compressed as `train.out.gz`.
