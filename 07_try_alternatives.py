import os
import cv2
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets
from scipy.ndimage import center_of_mass
from skimage.registration import phase_cross_correlation
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

# --- Config ---
RAW_DIR = '/nfs/roberts/project/pi_lhs7/xl754/abide_raw'
BASE_DIR = '/nfs/roberts/project/pi_lhs7/xl754/abide'
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'alternative_results')

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- Load RAFT ---
print("Loading RAFT model...")
device = "cpu"
weights = Raft_Small_Weights.DEFAULT
model = raft_small(weights=weights, progress=False).to(device)
model.eval()

def preprocess_for_raft(img1, img2):
    # 1. Convert to Tensor (C, H, W)
    t1 = torch.from_numpy(img1).float().unsqueeze(0)
    t2 = torch.from_numpy(img2).float().unsqueeze(0)
    
    # 2. Replicate channels (1 -> 3) for RGB model
    t1 = t1.repeat(3, 1, 1)
    t2 = t2.repeat(3, 1, 1)
    
    # 3. Add Batch Dimension (B, C, H, W)
    t1 = t1.unsqueeze(0)
    t2 = t2.unsqueeze(0)
    
    # 4. Normalize [-1, 1]
    t1 = (t1 / 127.5) - 1.0
    t2 = (t2 / 127.5) - 1.0
    
    # 5. RESIZE to 256x256 (Fix for small fMRI size)
    # We force the image to be large enough for RAFT's pyramid
    t1 = F.interpolate(t1, size=(256, 256), mode='bilinear', align_corners=False)
    t2 = F.interpolate(t2, size=(256, 256), mode='bilinear', align_corners=False)
        
    return t1.to(device), t2.to(device)

print("Loading dataset...")
abide = datasets.fetch_abide_pcp(data_dir=RAW_DIR, pipeline='cpac', quality_checked=True, n_subjects=10)
sub_ids = abide.phenotypic['SUB_ID'] if hasattr(abide.phenotypic, 'SUB_ID') else abide.phenotypic['SUB_ID']

comparison_rows = []

# Loop through subjects
for i, img_path in enumerate(abide.func_preproc[:10]):
    sub_id = sub_ids.iloc[i] if hasattr(sub_ids, 'iloc') else sub_ids[i]
    npy_path = os.path.join(PROCESSED_DIR, f'sub_{sub_id}.npy')
    
    if not os.path.exists(npy_path): continue
    print(f"Testing alternatives on Subject {sub_id}...")
    
    # Load Video
    video_array = np.load(npy_path)
    
    # Load Ground Truth
    try:
        img = nib.load(img_path)
        data = img.get_fdata()
        coms = [center_of_mass(data[..., t]) for t in range(data.shape[-1])]
        gt_motion = np.sqrt(np.sum(np.diff(np.array(coms), axis=0)**2, axis=1))
    except: continue
        
    min_len = min(video_array.shape[2]-1, len(gt_motion))
    gt_motion = gt_motion[:min_len]
    
    raft_scores = []
    phase_scores = []
    
    # Frame Loop
    for t in range(min_len):
        prev = video_array[:, :, t]
        curr = video_array[:, :, t+1]
        
        # === Method A: RAFT (Deep Learning) ===
        # Upsampling allows RAFT to see the brain as a large object
        with torch.no_grad():
            inp1, inp2 = preprocess_for_raft(prev, curr)
            
            # Forward pass
            list_of_flows = model(inp1, inp2)
            predicted_flow = list_of_flows[-1] # (B, 2, 256, 256)
            
            # To Numpy
            flow_np = predicted_flow[0].cpu().numpy().transpose(1, 2, 0)
            
            # Magnitude
            mag, _ = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
            
            # We calculate mean directly on the upsampled flow
            # Correlation is scale-invariant, so we don't need to resize back
            raft_scores.append(np.mean(mag))
            
        # === Method B: Phase Correlation (Fourier) ===
        try:
            # Phase correlation works in frequency domain, very robust to noise
            shift, error, diffphase = phase_cross_correlation(prev, curr, upsample_factor=10)
            shift_mag = np.sqrt(shift[0]**2 + shift[1]**2)
            phase_scores.append(shift_mag)
        except:
            phase_scores.append(0.0)

    # Calculate Correlations
    corr_raft = np.corrcoef(raft_scores, gt_motion)[0, 1]
    corr_phase = np.corrcoef(phase_scores, gt_motion)[0, 1]
    
    comparison_rows.append({
        'Subject': sub_id,
        'RAFT_Corr': corr_raft,
        'Phase_Corr': corr_phase
    })
    print(f"  -> RAFT: {corr_raft:.4f} | Phase: {corr_phase:.4f}")

# Summary
df = pd.DataFrame(comparison_rows)
print("\n" + "="*40)
print("ALTERNATIVE METHODS COMPARISON (Avg)")
print("="*40)
print(f"RAFT (Deep Learning) : {df['RAFT_Corr'].mean():.4f}")
print(f"Phase Correlation    : {df['Phase_Corr'].mean():.4f}")

df.to_csv(os.path.join(RESULTS_DIR, 'alternatives_comparison.csv'), index=False)