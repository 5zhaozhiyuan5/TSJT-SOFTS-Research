# PMME F2 gradient marginal four methods - run logs

- Branch: `experiment/pmme-f2-gradient-marginal-four-20261126`
- Source commit: `46b6c4c56ee73d79fbc54448823ee725add403de`
- Server run: batch 16, seed 1, target PEMS-BAY; methods run sequentially as `mag`, `sign`, `ema`, `biot`.
- Stage I: 3000 epochs; Stage II: 6000 epochs; DAepoch 1500; OT scale 0.01.
- Existing server PMME environment; no downloads or package installation on the server.

This is a live snapshot. `mag/seed1` was training at the time of this commit. Logs for later methods will be added as their runs proceed. Training output is gzip-compressed as `train.out.gz`.
