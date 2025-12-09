import os
import cv2
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets
from scipy.ndimage import center_of_mass
from scipy.signal import find_peaks

# --- Paths ---
RAW_DIR = '/nfs/roberts/project/pi_lhs7/xl754/abide_raw'
BASE_DIR = '/nfs/roberts/project/pi_lhs7/xl754/abide'
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'detection_results')

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

print("Loading dataset for Detection Evaluation...")
# 跑前 10 个做快速验证，或者 50 个做完整版
abide = datasets.fetch_abide_pcp(data_dir=RAW_DIR, pipeline='cpac', quality_checked=True, n_subjects=10)

pheno_data = abide.phenotypic
sub_ids = pheno_data['SUB_ID'] if isinstance(pheno_data, pd.DataFrame) else pheno_data['SUB_ID']

comparison_rows = []

def get_binary_spikes(signal, n_std=2.0):
    """
    使用 Mean + N*STD 作为阈值，将信号转换为二值 (0 或 1)
    """
    threshold = np.mean(signal) + n_std * np.std(signal)
    binary = (signal > threshold).astype(int)
    return binary, threshold

def calculate_metrics(pred_signal, gt_signal):
    # 1. Binary Classification Metrics
    min_len = min(len(pred_signal), len(gt_signal))
    pred = pred_signal[:min_len]
    gt = gt_signal[:min_len]
    
    pred_bin, _ = get_binary_spikes(pred)
    gt_bin, _ = get_binary_spikes(gt)
    
    tp = np.sum((pred_bin == 1) & (gt_bin == 1))
    tn = np.sum((pred_bin == 0) & (gt_bin == 0))
    fp = np.sum((pred_bin == 1) & (gt_bin == 0))
    fn = np.sum((pred_bin == 0) & (gt_bin == 1))
    
    sensitivity = tp / (tp + fn + 1e-6) # Recall
    specificity = tn / (tn + fp + 1e-6)
    
    # 2. Peak Time Difference (Lag)
    gt_peaks, _ = find_peaks(gt, height=np.mean(gt) + 2*np.std(gt), distance=5)
    pred_peaks, _ = find_peaks(pred, height=np.mean(pred) + 2*np.std(pred), distance=5)
    
    lags = []
    for gt_idx in gt_peaks:
        if len(pred_peaks) > 0:
            diffs = np.abs(pred_peaks - gt_idx)
            min_diff = np.min(diffs)
            if min_diff <= 5: 
                lags.append(min_diff)
    
    avg_lag = np.mean(lags) if len(lags) > 0 else np.nan
    
    return sensitivity, specificity, avg_lag

# --- Main Loop ---
for i, img_path in enumerate(abide.func_preproc[:10]): # 这里控制被试数量
    sub_id = sub_ids.iloc[i] if hasattr(sub_ids, 'iloc') else sub_ids[i]
    npy_path = os.path.join(PROCESSED_DIR, f'sub_{sub_id}.npy')
    
    if not os.path.exists(npy_path): continue
    print(f"Evaluating Subject {sub_id}...")

    # Load Data
    video_array = np.load(npy_path)
    
    # --- Generate Signals ---
    # 1. Ground Truth (CoM)
    try:
        img = nib.load(img_path)
        data = img.get_fdata()
        coms = [center_of_mass(data[..., t]) for t in range(data.shape[-1])]
        gt_motion = np.sqrt(np.sum(np.diff(np.array(coms), axis=0)**2, axis=1))
    except: continue
        
    min_len = min(video_array.shape[2]-1, len(gt_motion))
    gt_motion = gt_motion[:min_len]

    # 2. Naive Diff (Baseline)
    naive_sig = []
    for t in range(min_len):
        diff = np.mean(np.abs(video_array[:,:,t+1].astype(float) - video_array[:,:,t].astype(float)))
        naive_sig.append(diff)
        
    # 3. Farneback (Baseline)
    farn_sig = []
    for t in range(min_len):
        flow = cv2.calcOpticalFlowFarneback(video_array[:,:,t], video_array[:,:,t+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask = video_array[:,:,t] > 15
        val = np.mean(mag[mask]) if np.sum(mask) > 0 else np.mean(mag)
        farn_sig.append(val)

    tvl1_sig = []
    optical_flow = cv2.optflow.createOptFlow_DualTVL1()
    optical_flow.setMedianFiltering(1); optical_flow.setInnerIterations(1); optical_flow.setOuterIterations(10)
    for t in range(min_len):
        flow = optical_flow.calc(video_array[:,:,t], video_array[:,:,t+1], None)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask = video_array[:,:,t] > 15
        val = np.mean(mag[mask]) if np.sum(mask) > 0 else np.mean(mag)
        tvl1_sig.append(val)

    # --- Calculate Metrics ---
    # Naive
    sens_n, spec_n, lag_n = calculate_metrics(np.array(naive_sig), gt_motion)
    # Farneback
    sens_f, spec_f, lag_f = calculate_metrics(np.array(farn_sig), gt_motion)
    # TV-L1
    sens_t, spec_t, lag_t = calculate_metrics(np.array(tvl1_sig), gt_motion)

    comparison_rows.append({
        'Subject': sub_id,
        'Naive_Sens': sens_n, 'Naive_Spec': spec_n, 'Naive_Lag': lag_n,
        'Farn_Sens': sens_f, 'Farn_Spec': spec_f, 'Farn_Lag': lag_f,
        'TVL1_Sens': sens_t, 'TVL1_Spec': spec_t, 'TVL1_Lag': lag_t
    })

# --- Summary ---
df = pd.DataFrame(comparison_rows)
df_avg = df.mean(numeric_only=True)

print("\n" + "="*50)
print("FINAL DETECTION COMPARISON (Average)")
print("="*50)
print(f"Metric        | Naive  | Farneback | TV-L1 (Ours)")
print(f"--------------|--------|-----------|-------------")
print(f"Sensitivity   | {df_avg['Naive_Sens']:.3f}  | {df_avg['Farn_Sens']:.3f}     | {df_avg['TVL1_Sens']:.3f}")
print(f"Specificity   | {df_avg['Naive_Spec']:.3f}  | {df_avg['Farn_Spec']:.3f}     | {df_avg['TVL1_Spec']:.3f}")
print(f"Time Lag (Fr) | {df_avg['Naive_Lag']:.2f}   | {df_avg['Farn_Lag']:.2f}      | {df_avg['TVL1_Lag']:.2f}")

df.to_csv(os.path.join(RESULTS_DIR, 'detection_comparison.csv'), index=False)