import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import datasets
from scipy.ndimage import center_of_mass

RAW_DIR = '/nfs/roberts/project/pi_lhs7/xl754/abide/abide_raw'
BASE_DIR = '/nfs/roberts/project/pi_lhs7/xl754/abide'
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIG_DIR = os.path.join(BASE_DIR, 'figures')

if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)

print("Loading dataset for validation...")
abide = datasets.fetch_abide_pcp(data_dir=RAW_DIR, pipeline='cpac', quality_checked=True, n_subjects=50)

pheno_data = abide.phenotypic
sub_ids = pheno_data['SUB_ID'] if isinstance(pheno_data, pd.DataFrame) else pheno_data['SUB_ID']

correlations = []

for i, img_path in enumerate(abide.func_preproc):
    sub_id = sub_ids.iloc[i] if hasattr(sub_ids, 'iloc') else sub_ids[i]

    print(f"Validating Subject {sub_id}...")
    my_result_path = os.path.join(RESULTS_DIR, f'sub_{sub_id}_motion.csv')
    
    if not os.path.exists(my_result_path):
        continue

    try:

        my_df = pd.read_csv(my_result_path)
        my_flow = my_df['optical_flow_mag'].values
        
        img = nib.load(img_path)
        data = img.get_fdata()
        
        coms = []
        for t in range(data.shape[-1]):
            coms.append(center_of_mass(data[..., t]))
        coms = np.array(coms)
        gt_motion = np.sqrt(np.sum(np.diff(coms, axis=0)**2, axis=1))

        min_len = min(len(my_flow), len(gt_motion))
        corr = np.corrcoef(my_flow[:min_len], gt_motion[:min_len])[0, 1]
        correlations.append(corr)
        print(f"  -> Correlation: {corr:.4f}")
        
        plt.figure(figsize=(10, 4))
        p1 = (my_flow[:min_len] - np.mean(my_flow[:min_len])) / np.std(my_flow[:min_len])
        p2 = (gt_motion[:min_len] - np.mean(gt_motion[:min_len])) / np.std(gt_motion[:min_len])
        
        plt.plot(p1, label='Optical Flow (Blur)', alpha=0.8)
        plt.plot(p2, label='Ground Truth (CoM)', alpha=0.6)
        plt.title(f"Subject {sub_id} (Corr={corr:.3f})")
        plt.legend()
        plt.savefig(os.path.join(FIG_DIR, f'sub_{sub_id}_val.png'))
        plt.close()

    except Exception as e:
        print(f"Error: {e}")

if correlations:
    print(f"AVERAGE CORRELATION: {np.mean(correlations):.4f}")