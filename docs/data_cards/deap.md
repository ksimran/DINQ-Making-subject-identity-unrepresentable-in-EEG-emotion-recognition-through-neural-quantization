# DEAP — data card

- **Source:** `$DINQ_DEAP_ROOT` (default `$DINQ_DATA_ROOT/DEAP_dataset`; data_preprocessed_python, .dat)
- **Subjects:** 32 (ids 1..32)
- **Trials:** 1280 total, 40/subject
- **Sampling:** 128 Hz; 60 s stimulus (3 s baseline dropped)
- **Label mapping:** valence median-split -> binary; trial balance: 638 pos / 642 neg (49.8% pos)
- **Channels:** 32 EEG (Geneva order) -> common montage C=32
- **Bands (four):** ['theta', 'alpha', 'beta', 'gamma']  (delta absent: DEAP is 4-45 Hz)
- **DE feature dim:** C*B = 128
- **Covariance/tangent dim:** B * C(C+1)/2 = 2112
- **Domain id:** 0
