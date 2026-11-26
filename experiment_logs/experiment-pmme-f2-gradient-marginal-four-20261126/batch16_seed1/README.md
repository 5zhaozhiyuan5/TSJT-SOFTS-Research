# PMME F2 gradient marginal four methods - run logs

- Branch: `experiment/pmme-f2-gradient-marginal-four-20261126`
- Source commit: `46b6c4c56ee73d79fbc54448823ee725add403de`
- Run: batch 16, seed 1, target PEMS-BAY; sequential methods `mag`, `sign`, `ema`, `biot`.
- Stage I: 3000 epochs; Stage II: 6000 epochs; DAepoch 1500; OT scale 0.01.
- Environment: existing server PMME environment; no downloads or package installation on the server.

Live snapshot: `mag/seed1` has reached Stage I epoch 300/3000. The other three methods are queued. Training logs are stored as gzip-compressed `train.out.gz` files.
