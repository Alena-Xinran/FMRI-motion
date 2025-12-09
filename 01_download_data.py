import os
from nilearn import datasets

DATA_DIR = '/nfs/roberts/project/pi_lhs7/xl754/abide/abide_raw'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

datasets.fetch_abide_pcp(
    data_dir=DATA_DIR,
    pipeline='cpac',
    quality_checked=True,
    n_subjects=50,
    verbose=1
)

print(f"Download complete. Data stored in {DATA_DIR}")