Save the data in `./data`
For standard (non-domain adaptation) methods, run `normal_baseline.sh`.
For domain adaptation (DA) methods, run `DA_baseline.sh`.
For PMME, run `PMME.sh`.

## PMME Stage-II (experimental replacement)

The current `main` branch keeps PMME Stage-I and the SOFTS forecasting backbone unchanged. Stage-II uses a source-only RAFT-style multi-scale history/future bank and a SARAF/TimeFuse-inspired reliability-aware horizon gate. The gate replaces the earlier global dense fusion layer and adapts only the retrieval scale projections and gate on the target three-day support set.

- Literature review, v2 failure analysis, and gate design: [docs/stage2_raft_replacement.md](docs/stage2_raft_replacement.md)
- Adapter implementation: [model/Meta_Models/raft_cross_city.py](model/Meta_Models/raft_cross_city.py)
- Smoke tests: [tests/test_raft_cross_city.py](tests/test_raft_cross_city.py)

Run both smoke suites from the repository root:

```bash
python tests/test_raft_cross_city.py
python tests/test_pmme_raft_stage2_integration.py
```

The second suite exercises the real dataset preprocessing code, the SOFTS forecasting backbone, source-only memory construction, and one Stage-II optimizer step using generated synthetic arrays. It is an integration smoke test, not a real four-city forecasting benchmark. Full real-data training still requires the four datasets and the matching Stage-I checkpoint.

See [the Stage-II runtime review](docs/stage2_raft_replacement.md#tests-and-what-they-do-not-prove) for the bugs found and CI evidence.


## PMME Stage-II (current experiment)

The current Stage-II implementation uses three horizon-conditioned queries over one shared residual-memory matrix, with separate 12-step generators for short/mid/long forecast bands. The shared memory is initialized from frozen SOFTS representations of the authorized three-day target support inputs; Stage-I remains frozen and unchanged.

- Design and protocol: [docs/stage2_horizon_memory.md](docs/stage2_horizon_memory.md)
- Implementation: [model/Meta_Models/horizon_residual_memory.py](model/Meta_Models/horizon_residual_memory.py)
- Unit test: [tests/test_horizon_residual_memory.py](tests/test_horizon_residual_memory.py)
- Integration test: [tests/test_pmme_horizon_stage2_integration.py](tests/test_pmme_horizon_stage2_integration.py)

Run the two tests from the repository root:
- `python tests/test_horizon_residual_memory.py`
- `python tests/test_pmme_horizon_stage2_integration.py`

The integration test uses synthetic arrays and does not establish a real four-city benchmark improvement.
