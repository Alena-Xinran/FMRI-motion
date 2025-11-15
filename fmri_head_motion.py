#!/usr/bin/env python3

import sys
sys.path.append("/home/xl754/fmri_motion_project/RAFT")
sys.path.append("/home/xl754/fmri_motion_project/RAFT/core")  # <—— 新增这一行
from core.utils.utils import InputPadder
from core.raft import RAFT
from argparse import Namespace
import argparse
import json
from pathlib import Path

import numpy as np
import nibabel as nib
from skimage.registration import phase_cross_correlation
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def load_fmri_slice(
    fmri_path: str,
    z_index: int | None = None,
    normalize: bool = True
) -> np.ndarray:

    fmri_path = Path(fmri_path)
    img = nib.load(str(fmri_path))
    data = img.get_fdata()  

    if data.shape[-1] < 10:

        data_txyz = data
    else:
        # shape: (X, Y, Z, T) -> (T, X, Y, Z)
        data_txyz = np.moveaxis(data, -1, 0)

    T, X, Y, Z = data_txyz.shape

    if z_index is None:
        z_index = Z // 2 

    video = data_txyz[:, :, :, z_index].astype(np.float32)  # (T, H, W)

    if normalize:
        vmin = np.percentile(video, 1)
        vmax = np.percentile(video, 99)
        video = np.clip(video, vmin, vmax)
        video = video - video.min()
        video = video / (video.max() + 1e-8)  # 现在在 [0, 1]

    return video  # (T, H, W)

def estimate_motion_registration(video: np.ndarray):

    assert video.ndim == 3
    T = video.shape[0]

    ref = video[0]
    dx_list = [0.0]
    dy_list = [0.0]

    for t in range(1, T):
        shift, error, _ = phase_cross_correlation(ref, video[t])
        # shift: (shift_y, shift_x)
        dy, dx = shift
        dx_list.append(float(dx))
        dy_list.append(float(dy))

    return np.array(dx_list, dtype=np.float32), np.array(dy_list, dtype=np.float32)




class RAFTWrapper(nn.Module):
    def __init__(self, ckpt_path: str, device: str):
        super().__init__()
        from argparse import Namespace
        args = Namespace(
            small=False,            # 如果你用的是 raft-sintel / things / kitti / chairs
            mixed_precision=False,
            alternate_corr=False
        )
        self.model = RAFT(args)
        self.model.to(device)

        state_dict = torch.load(ckpt_path, map_location=device)

        # 去掉 DataParallel 的 "module." 前缀
        if any(k.startswith("module.") for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace("module.", "", 1)
                new_state_dict[new_k] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        img1, img2: (1, 1, H, W)，单通道 slice
        返回 flow: (1, 2, H, W)
        """
        # 1) 复制成 3 通道，因为 RAFT 期望 3-channel
        img1_3c = img1.repeat(1, 3, 1, 1)
        img2_3c = img2.repeat(1, 3, 1, 1)

        # 2) 用 InputPadder 把尺寸 pad 到 8 的倍数
        padder = InputPadder(img1_3c.shape)
        img1_pad, img2_pad = padder.pad(img1_3c, img2_3c)

        # 3) RAFT 前向
        flow_low, flow_up = self.model(img1_pad, img2_pad, iters=20, test_mode=True)

        # 4) 把 flow unpad 回原始尺寸
        flow_up = padder.unpad(flow_up)  # (1, 2, H, W)

        return flow_up

def load_pretrained_flow_model(device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = "/home/xl754/fmri_motion_project/RAFT/models/raft-sintel.pth"
    model = RAFTWrapper(ckpt_path, device)
    return model, device


def estimate_motion_optical_flow(video: np.ndarray):

    assert video.ndim == 3
    T, H, W = video.shape

    model, device = load_pretrained_flow_model()

    dx_list = [0.0]
    dy_list = [0.0]

    cum_dx, cum_dy = 0.0, 0.0

    for t in range(T - 1):
        I1 = video[t]
        I2 = video[t + 1]

        ten1 = torch.from_numpy(I1).float().unsqueeze(0).unsqueeze(0).to(device)
        ten2 = torch.from_numpy(I2).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            flow = model(ten1, ten2)  

        flow_np = flow[0].cpu().numpy()  
        u = flow_np[0]  # dx field
        v = flow_np[1]  # dy field

        mean_dx = float(u.mean())
        mean_dy = float(v.mean())

        cum_dx += mean_dx
        cum_dy += mean_dy

        dx_list.append(cum_dx)
        dy_list.append(cum_dy)

    return np.array(dx_list, dtype=np.float32), np.array(dy_list, dtype=np.float32)



def compare_trajectories(dx_reg, dy_reg, dx_flow, dy_flow):
    """
    计算两条 motion 曲线的相关和 RMSE，
    会自动忽略包含 NaN / Inf 的帧。
    """
    dx_reg = np.asarray(dx_reg, dtype=np.float32)
    dy_reg = np.asarray(dy_reg, dtype=np.float32)
    dx_flow = np.asarray(dx_flow, dtype=np.float32)
    dy_flow = np.asarray(dy_flow, dtype=np.float32)

    assert dx_reg.shape == dx_flow.shape
    assert dy_reg.shape == dy_flow.shape

    # --------------------------
    # 1) 构造有效帧的 mask（四条曲线都要是有限数）
    # --------------------------
    valid_mask_x = np.isfinite(dx_reg) & np.isfinite(dx_flow)
    valid_mask_y = np.isfinite(dy_reg) & np.isfinite(dy_flow)

    dx_reg_valid = dx_reg[valid_mask_x]
    dx_flow_valid = dx_flow[valid_mask_x]
    dy_reg_valid = dy_reg[valid_mask_y]
    dy_flow_valid = dy_flow[valid_mask_y]

    # --------------------------
    # 2) 安全求 Pearson 相关
    #    - 至少要 3 个点
    #    - 而且不能全是常数（否则方差为 0）
    # --------------------------
    def safe_pearson(a, b):
        if len(a) < 3 or len(b) < 3:
            return np.nan
        if np.allclose(a, a[0]) or np.allclose(b, b[0]):
            return np.nan
        r, _ = pearsonr(a, b)
        return float(r)

    corr_x = safe_pearson(dx_reg_valid, dx_flow_valid)
    corr_y = safe_pearson(dy_reg_valid, dy_flow_valid)

    # --------------------------
    # 3) RMSE 也只在有效帧上算
    # --------------------------
    if len(dx_reg_valid) > 0:
        rmse_x = float(np.sqrt(((dx_reg_valid - dx_flow_valid) ** 2).mean()))
    else:
        rmse_x = np.nan

    if len(dy_reg_valid) > 0:
        rmse_y = float(np.sqrt(((dy_reg_valid - dy_flow_valid) ** 2).mean()))
    else:
        rmse_y = np.nan

    return {
        "corr_x": corr_x,
        "corr_y": corr_y,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
    }

def plot_motion_curves(
    dx_reg, dy_reg, dx_flow, dy_flow, out_path: str | Path, title: str = "Head Motion"
):

    out_path = Path(out_path)

    T = len(dx_reg)
    frames = np.arange(T)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(frames, dx_reg, label="Reg dx", linestyle="-")
    plt.plot(frames, dx_flow, label="Flow dx", linestyle="--")
    plt.ylabel("dx")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(frames, dy_reg, label="Reg dy", linestyle="-")
    plt.plot(frames, dy_flow, label="Flow dy", linestyle="--")
    plt.xlabel("Frame")
    plt.ylabel("dy")
    plt.legend()
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()




def main():
    parser = argparse.ArgumentParser(
        description="fMRI 2D head motion experiment (single subject)."
    )
    parser.add_argument("--fmri_path", type=str, required=True, help="Path to fMRI NIfTI file.")
    parser.add_argument("--z_index", type=int, default=None, help="Slice index (default: middle slice).")
    parser.add_argument("--out_dir", type=str, required=True, help="Output root directory.")
    parser.add_argument("--subject_id", type=str, default="subject01", help="Subject ID for naming.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading fMRI and extracting 2D slice video...")
    video = load_fmri_slice(args.fmri_path, z_index=args.z_index, normalize=True)
    np.save(out_dir / "video.npy", video)

    print("Estimating motion via registration (baseline)...")
    dx_reg, dy_reg = estimate_motion_registration(video)
    np.save(out_dir / "motion_registration_dx.npy", dx_reg)
    np.save(out_dir / "motion_registration_dy.npy", dy_reg)

    print("Estimating motion via optical flow (video model)...")
    dx_flow, dy_flow = estimate_motion_optical_flow(video)
    np.save(out_dir / "motion_flow_dx.npy", dx_flow)
    np.save(out_dir / "motion_flow_dy.npy", dy_flow)

    print("Comparing trajectories...")
    stats = compare_trajectories(dx_reg, dy_reg, dx_flow, dy_flow)
    with open(out_dir / "comparison_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("Plotting motion curves...")
    plot_motion_curves(
        dx_reg,
        dy_reg,
        dx_flow,
        dy_flow,
        out_path=out_dir / "motion_curves.png",
        title=f"Head Motion - {args.subject_id}",
    )

    print("Done.")
    print("Results saved in:", out_dir)
    print("Stats:", stats)


if __name__ == "__main__":
    main()