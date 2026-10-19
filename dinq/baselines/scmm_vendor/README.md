# SCMM source files (not included)

The SCMM baseline (`dinq/baselines/scmm.py`, `dinq/baselines/scmm_train.py`) reuses
SCMM's self-supervised losses and masking unchanged. The SCMM repository
(<https://github.com/Kyler-RL/SCMM>; Liu et al., "EEG-SCMM: Soft Contrastive Masked
Modeling for Cross-Corpus EEG-Based Emotion Recognition", ACM MM 2025) carries no
licence, so its source is not redistributed here.

To run the SCMM baseline, copy these three files from the SCMM repository into this
folder:

| File | Symbols used by `scmm.py` |
|---|---|
| `augmentation.py` | `masking` |
| `loss.py` | `AutoWeightLoss`, `SoftContrastiveLoss`, `AggregateReconstruct`, `ReconstructiveLoss` |
| `utils.py` | `generate_soft_assignment` (imported by `loss.py`) |

`loss.py` imports `generate_soft_assignment` with `from .utils import ...`. If the
upstream file uses an absolute import, change it to the relative form.

Which scripts need these files:
- `scripts/run_baselines.py` (cross-dataset reference baselines) imports `SCMMConfig`
  when it starts, so it needs them for every model, not only `--models scmm`.
- Nothing else does. The DINQ model, the FACED subject-axis baselines
  (`subject_baselines.py`), the sweeps, the permutation test, the cross-dataset DINQ
  runs (`cross_corpus.py`) and `make_paper_figures.py` all run without SCMM.
