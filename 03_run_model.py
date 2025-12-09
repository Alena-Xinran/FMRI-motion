import os
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

BASE_DIR = '/nfs/roberts/project/pi_lhs7/xl754/abide'
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def process_subject(file_name):
    try:
        sub_id = file_name.split('_')[1].split('.')[0]
        data_path = os.path.join(PROCESSED_DIR, file_name)
        out_csv_path = os.path.join(RESULTS_DIR, f'sub_{sub_id}_motion.csv')
        
        optical_flow = cv2.optflow.createOptFlow_DualTVL1()
        optical_flow.setMedianFiltering(1)
        optical_flow.setInnerIterations(1)
        optical_flow.setOuterIterations(2) 
        
        video_array = np.load(data_path)
        n_frames = video_array.shape[2]
        motion_magnitudes = []
        
        for t in range(n_frames - 1):
            prev = video_array[:, :, t]
            curr = video_array[:, :, t + 1]
            
            flow = optical_flow.calc(prev, curr, None)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_magnitudes.append(np.mean(mag))
            
        df = pd.DataFrame({'frame': range(len(motion_magnitudes)), 'optical_flow_mag': motion_magnitudes})
        df.to_csv(out_csv_path, index=False)
        
        return f"Finished Subject {sub_id}"
        
    except Exception as e:
        return f"Error processing {file_name}: {str(e)}"

if __name__ == '__main__':
    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.npy')]
    print(f"Found {len(files)} subjects. Starting parallel processing...")
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = executor.map(process_subject, files)
        for res in results:
            print(res)