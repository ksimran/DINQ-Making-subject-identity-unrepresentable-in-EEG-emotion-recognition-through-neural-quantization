# SEED-V — data card

- **Source (DE):** `$DINQ_SEEDV_DE_ROOT` (default `$DINQ_DATA_ROOT/SEED_V/EEG_DE_features`; precomputed DE, 4 s windows)
- **Source (raw):** `/run/user/1000/gvfs/sftp:host=192.168.1.14/mnt/Data/My_Space/Datasets/SEED_V/EEG_raw` — NOT mounted (SFTP)
- **Subjects:** 16 (offset ids 33..48, +32)
- **Trials:** 45/subject (3 sessions x 15)
- **Native channels:** 62 (10-20) -> common montage C=32
- **Label mapping:** happy->pos; {sad,fear,disgust}->neg; neutral dropped.
  Kept-segment balance: 4816 pos / 18480 neg (20.7% pos)
- **Bands (four):** ['theta', 'alpha', 'beta', 'gamma']
- **DE feature dim:** C*B = 128 (channel-major reshape(n,62,5))
- **Covariance/tangent dim:** B * C(C+1)/2 = 2112 (needs raw .cnt)
- **Domain id:** 1
