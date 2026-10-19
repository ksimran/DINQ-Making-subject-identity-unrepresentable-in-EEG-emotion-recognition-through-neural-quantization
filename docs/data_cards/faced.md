# FACED — data card (onboarding, mini Gate-0)

- **Source (preprocessed):** `<DATA_ROOT>/Faced/Clisa_data.zip` → `Clisa_data/sub{000..122}.pkl`,
  read per-subject straight from the zip (no full extraction). Also present: raw
  `Data.zip` (.bdf, 1000 Hz), `EEG_Features` (DE/PSD, 5 bands).
- **Subjects:** 123 (S000–S122) → offset ids **100..222** (disjoint from DEAP 1..32, SEED-V 33..48). Domain id **2**.
- **Signal:** preprocessed `.pkl` = **(28 videos, 32 ch, 7500 samples)** float64,
  **250 Hz**, 30 s/trial. Channels 16–17 are **A1/A2 mastoids (dropped)** → **30 EEG**.
- **Channels:** `.bdf`/`.pkl` order **validated against sub000 header** (header
  guard passes). Old 10-20 names T3/T4/T5/T6 modernized to T7/T8/P7/P8.
  - **DEAP∩FACED = 30** (DEAP minus AF3/AF4, which FACED lacks)
  - **FACED∩SEED-V = 30** · **DEAP∩SEED-V∩FACED = 30**
- **Label mapping (binary):** Stimuli_info Valence column, .pkl videos in
  Stimuli_info index order — videos 1–12 Negative → **0**; 13–16 Neutral →
  **drop**; 17–28 Positive → **1** (4 pos: amusement/inspiration/joy/tenderness;
  4 neg: anger/disgust/fear/sadness). Per subject: **24 kept trials, balanced
  12 neg / 12 pos** (cleaner than SEED-V's 20.7% pos). *Flag: .pkl video ordering
  is per the FACED Readme; worth a cross-check vs behavioural ratings.*
- **Loader:** `dinq/data/faced.py` → channel-aligned Trials, neutral dropped,
  ready for the same wavelet→cov→shrinkage→**recenter**→tangent (+DE) pipeline.

## Status: pipeline-ready; HELD before certificate
Loader + channel reconciliation (header-guarded) + label mapping DONE and
validated. **Deferred (heavy, and to protect the running baselines):** building
the covariance/tangent + DE caches for all 123 subjects and the 100%-SPD
validation — these run once the baseline sweeps free the CPU. Per the onboarding
plan, FACED does **not** enter the certificate / headline experiments until the
certificate is locked and thresholds frozen. Then FACED = the primary
cross-subject headline (123 subjects) + DEAP↔FACED / SEED-V↔FACED pairs.
