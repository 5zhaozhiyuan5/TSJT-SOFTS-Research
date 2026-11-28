Save the data in `./data`
For standard (non-domain adaptation) methods, run `normal_baseline.sh`.
For domain adaptation (DA) methods, run `DA_baseline.sh`.
For PMME, run `PMME.sh`.

## STUNet official research baseline

The official STUNet implementation is pinned as a Git submodule in the dedicated branch \`baseline/stunet-official-diagnostics-20261128\`. It is kept separate from the PMME/TSJT experiments on \`main\`.

See:
- \`UPSTREAM_STUNET.md\` for source provenance and pinned commit.
- \`diagnostics/STUNET_UPSTREAM_AUDIT.md\` for source-level questions to test.
- \`diagnostics/STUNET_DIAGNOSTICS.md\` for setup, commands and log interpretation.

Clone the branch with \`--recurse-submodules\` to obtain the official source.
