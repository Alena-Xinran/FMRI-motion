import os
import cv2
import time
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets
from scipy.ndimage import center_of_mass

# --- Paths ---
RAW_DIR = '/nfs/roberts/project/pi_lhs7/xl754/abide/abide_raw'
BASE_DIR = '/nfs/roberts/project/pi_lhs7/xl754/abide'
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
COMPARISON_DIR = os.path.join(BASE_DIR, 'comparison_results')

if not os.path.exists(COMPARISON_DIR):
    os.makedirs(COMPARISON_DIR)

print("Loading dataset for comparison...")
abide = datasets.fetch_abide_pcp(data_dir=RAW_DIR, pipeline='cpac', quality_checked=True, n_subjects=50)

pheno_data = abide.phenotypic
sub_ids = pheno_data['SUB_ID'] if isinstance(pheno_data, pd.DataFrame) else pheno_data['SUB_ID']

results_table = []

for i, img_path in enumerate(abide.func_preproc[:10]):
    sub_id = sub_ids.iloc[i] if hasattr(sub_ids, 'iloc') else sub_ids[i]
    
    npy_path = os.path.join(PROCESSED_DIR, f'sub_{sub_id}.npy')
    if not os.path.exists(npy_path):
        continue
        
    print(f"Comparing methods for Subject {sub_id}...")

    # Load 2D Data
    video_array = np.load(npy_path)
    n_frames = video_array.shape[2]

    # Load Ground Truth (3D CoM)
    try:
        img = nib.load(img_path)
        data = img.get_fdata()
        coms = [center_of_mass(data[..., t]) for t in range(data.shape[-1])]
        gt_motion = np.sqrt(np.sum(np.diff(np.array(coms), axis=0)**2, axis=1))
        
        # Align lengths
        min_len = min(n_frames - 1, len(gt_motion))
        gt_motion = gt_motion[:min_len]
    except:
        print("  -> GT Error, skipping")
        continue

    # ==========================================
    # Method 1: Naive Pixel Difference (Baseline 1)
    # ==========================================
    start_time = time.time()
    diff_scores = []
    for t in range(min_len):
        prev = video_array[:, :, t].astype(float)
        curr = video_array[:, :, t+1].astype(float)
        # Simple Mean Absolute Difference
        diff = np.mean(np.abs(curr - prev))
        diff_scores.append(diff)
    
    time_diff = time.time() - start_time
    corr_diff = np.corrcoef(diff_scores, gt_motion)[0, 1]

    # ==========================================
    # Method 2: Farneback Optical Flow (Baseline 2)
    # ==========================================
    start_time = time.time()
    farn_scores = []
    for t in range(min_len):
        prev = video_array[:, :, t]
        curr = video_array[:, :, t+1]
        
        # OpenCV Farneback parameters (Fast)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Masking (Same as your main method)
        mask = prev > 15
        if np.sum(mask) > 0:
            farn_scores.append(np.mean(mag[mask]))
        else:
            farn_scores.append(np.mean(mag))
            
    time_farn = time.time() - start_time
    corr_farn = np.corrcoef(farn_scores, gt_motion)[0, 1]

    # ==========================================
    # Method 3: TV-L1 Optical Flow (Your Method)
    # ==========================================
    start_time = time.time()
    tvl1_scores = []
    optical_flow = cv2.optflow.createOptFlow_DualTVL1()
    optical_flow.setMedianFiltering(1)
    optical_flow.setInnerIterations(1)
    optical_flow.setOuterIterations(10) # Your settings
    
    for t in range(min_len):
        prev = video_array[:, :, t]
        curr = video_array[:, :, t+1]
        
        flow = optical_flow.calc(prev, curr, None)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        mask = prev > 15
        if np.sum(mask) > 0:
            tvl1_scores.append(np.mean(mag[mask]))
        else:
            tvl1_scores.append(np.mean(mag))
            
    time_tvl1 = time.time() - start_time
    corr_tvl1 = np.corrcoef(tvl1_scores, gt_motion)[0, 1]

    # Store results
    results_table.append({
        'Subject': sub_id,
        'Naive_Corr': corr_diff,
        'Farneback_Corr': corr_farn,
        'TVL1_Corr': corr_tvl1,
        'Naive_Time': time_diff,
        'Farneback_Time': time_farn,
        'TVL1_Time': time_tvl1
    })


df_res = pd.DataFrame(results_table)
df_avg = df_res.mean(numeric_only=True)

print("\n" + "="*40)
print("FINAL COMPARISON TABLE (Average)")
print("="*40)
print(f"Naive Diff Correlation : {df_avg['Naive_Corr']:.4f} | Time: {df_avg['Naive_Time']:.4f}s")
print(f"Farneback Correlation  : {df_avg['Farneback_Corr']:.4f} | Time: {df_avg['Farneback_Time']:.4f}s")
print(f"TV-L1 (Yours) Correlation: {df_avg['TVL1_Corr']:.4f} | Time: {df_avg['TVL1_Time']:.4f}s")


df_res.to_csv(os.path.join(COMPARISON_DIR, 'baseline_comparison.csv'), index=False)