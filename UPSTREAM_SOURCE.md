# Upstream source provenance

This repository mirrors the authors' original `PMME_and_Others` implementation.

- Upstream repository: https://github.com/serre20/PMME
- Upstream commit: `038d8d6958c13384c34b2e706e43fc0f421ea03c`
- Source directory: `PMME_and_Others/`
- Paper: **PMME: Spatio-Temporal Few-Shot Learning via Pattern Matching with Memory Enhancement**, PAKDD 2026, LNCS 16598, pp. 132–144.

All Python, YAML, and shell files from that upstream directory are mirrored without algorithmic changes. Generated `__pycache__` bytecode and external datasets/checkpoints are excluded. The root `README.md` is the upstream `PMME_and_Others/README.md`.

The upstream repository was freshly cloned and re-verified on 2026-11-14;
all 27 non-bytecode source files in `PMME_and_Others/` match this project by
SHA-256. The official model entry points are
`model/Meta_Models/PMME.py` (Stage-I) and
`model/Meta_Models/PMME_2ndStage.py` (Stage-II).

The authors' top-level README lists the following main dependencies: torch 2.6.0, torch-geometric 2.6.1, gpytorch 1.14, PyYAML 6.0.2, geomloss 0.2.6, and einops 0.8.1.

No training or benchmark reproduction is claimed by this source import itself.
