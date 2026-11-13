# PMME upstream baseline and modification boundary

The clean baseline for this research branch is the authors' official repository:

- Upstream repository: https://github.com/serre20/PMME
- Upstream commit: 038d8d6958c13384c34b2e706e43fc0f421ea03c
- Upstream source directory: PMME_and_Others/
- Paper: PMME: Spatio-Temporal Few-Shot Learning via Pattern Matching with Memory Enhancement, PAKDD 2026, LNCS 16598, pp. 132–144.

The 27 source/configuration/shell files from the upstream PMME_and_Others directory were mirrored to this repository root and verified by Git blob SHA. The official Stage-II implementation remains byte-for-byte identical to upstream:

- model/Meta_Models/PMME_2ndStage.py
- Official Stage-II residual memory, retrieval, gating, optimization, and training behavior are not modified.

## Research modification boundary

Only the PMME Stage-I training path is being extended:

- model/Meta_Models/PMME.py: adds an auxiliary multi-horizon traffic response-profile supervision head to the Stage-I trainer.
- train_DA_OT.py: adds the --profile_coef argument. The original Stage-II training/evaluation block is unchanged.

The response-profile head is outside the PatchFSL forecasting model. It is used only to supervise Stage-I shared representations and is not loaded by the official Stage-II model. Stage-II continues to load the Stage-I forecasting backbone weights using the original checkpoint filtering behavior.

No claim of benchmark improvement or full experiment reproduction is made until the real cross-city experiments are run.