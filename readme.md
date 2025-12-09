# Zero-Shot 2D Head Motion Estimation on fMRI

This project investigates **zero-shot motion detection** in fMRI by treating 4D volumes as 2D video streams. We benchmark computer vision methods ranging from statistical baselines to state-of-the-art Deep Learning models.

![Workflow](figures/pipeline_diagram.png)

## 📉 Key Findings: Occam's Razor
Our experiments on the ABIDE dataset ($N=50$) reveal an inverse relationship between complexity and performance:

1.  **Simple Wins:** The **Naive Pixel Difference** baseline (analogous to **DVARS**) achieved the highest sensitivity and correlation.
2.  **Domain Gap:** **RAFT** (SOTA Deep Learning) failed to generalize to fMRI ($r \approx 0$) due to the lack of rich textures in medical images.
3.  **Complexity Trap:** **TV-L1 Optical Flow** provided high temporal precision (low lag) but is computationally expensive and tends to over-smooth motion spikes.

## 📊 Performance Benchmark

| Method | Type | Correlation ($r$) | Sensitivity | Runtime (s/vol) |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Pixel Diff** | Statistical | **0.060** | **0.062** | **0.003** |
| **TV-L1 Flow** | Classical CV | 0.047 | 0.017 | 5.156 |
| **Phase Correlation** | Frequency | 0.047 | 0.045 | 0.081 |
| **RAFT** | Deep Learning | 0.003 | 0.000 | 0.520 |

## 🚀 Quick Start

### 1. Requirements
```bash
pip install numpy pandas matplotlib seaborn opencv-python nibabel nilearn scipy torch torchvision
```

## 2. Execution Pipeline

Run the scripts in the following order to reproduce the experiments:

### 1) Data Prep
```bash
python 02_process_data.py
```
- Downloads ABIDE data  
- Extracts axial slices  
- Applies Gaussian Blur  

### 2) Main Model
```bash
python 03_run_model.py
```
- Runs TV-L1 Optical Flow (multiprocessing).

### 3) Baselines
```bash
python 05_run_baselines.py
```
- Runs Naive Difference and Farneback Flow.

### 4) Deep Learning
```bash
python 07_try_alternatives.py
```
- Runs RAFT (Zero-Shot) and Phase Correlation.

### 5) Analysis & Plotting
```bash
python 04_validate.py
```
- Individual validation

```bash
python 06_run_detection_metrics.py
```
- Detection stats

```bash
python 08_generate_final_plots.py
```
- Final figures

```bash
python 09_subgroup_analysis.py
```
- High vs. Low motion split

---

## 📂 Project Structure
- `processed/`: Intermediate 2D video arrays (`.npy`).
- `results/`: Raw CSV outputs from motion estimators.
- `figures/`: Final plots (Correlation distributions, Efficiency curves, Trace alignments).
- `comparison_results/`: Evaluation metrics.

---

## 👥 Author
Alena-Xinran
