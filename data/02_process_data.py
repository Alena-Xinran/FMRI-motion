import os
import numpy as np
import nibabel as nib
from nilearn import datasets
import pandas as pd
import cv2

RAW_DIR = '/nfs/roberts/project/pi_lhs7/xl754/abide/abide_raw'
BASE_DIR = '/nfs/roberts/project/pi_lhs7/xl754/abide'
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')

if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

print("Loading dataset...")
abide = datasets.fetch_abide_pcp(data_dir=RAW_DIR, pipeline='cpac', quality_checked=True, n_subjects=50)

pheno_data = abide.phenotypic
sub_ids = pheno_data['SUB_ID'] if isinstance(pheno_data, pd.DataFrame) else pheno_data['SUB_ID']

for i, img_path in enumerate(abide.func_preproc):
    sub_id = sub_ids.iloc[i] if hasattr(sub_ids, 'iloc') else sub_ids[i]
    
    try:
        img = nib.load(img_path)
        data = img.get_fdata()
        
        z_center = data.shape[2] // 2
      
        slice_idx = z_center + 5
        if slice_idx >= data.shape[2]:
            slice_idx = z_center
            
        slice_time_series = data[:, :, slice_idx, :]
        
        p2, p98 = np.percentile(slice_time_series, (2, 98))
        slice_clipped = np.clip(slice_time_series, p2, p98)
        
        data_min = slice_clipped.min()
        data_max = slice_clipped.max()
        
        if data_max - data_min == 0: 
            continue
            
        normalized_data = ((slice_clipped - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        
        normalized_data = np.rot90(normalized_data, k=1)
        

        blurred_video = np.zeros_like(normalized_data)
        for t in range(normalized_data.shape[2]):
            blurred_video[:, :, t] = cv2.GaussianBlur(normalized_data[:, :, t], (5, 5), 0)
            
        save_path = os.path.join(PROCESSED_DIR, f'sub_{sub_id}.npy')
        np.save(save_path, blurred_video)
        print(f"Processed Subject {sub_id} (Axial Slice)")
        
    except Exception as e:
        print(f"Error processing Subject {sub_id}: {e}")

print("Processing complete.")