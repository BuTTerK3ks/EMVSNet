import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from matplotlib import cm

def masked_imshow(ax, arr, mask, title, cmap="viridis", vmin=None, vmax=None, cbar_label=None):
    arr_masked = np.ma.masked_where(~mask, arr)
    cmap_obj = cm.get_cmap(cmap)
    cmap_obj.set_bad(color='black')
    im = ax.imshow(arr_masked, cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.axis("off")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    cbar.ax.yaxis.label.set_fontsize(16)
    cbar.ax.yaxis.label.set_fontweight('bold')


def visualize_result_file(filepath, to_mm=1.0):
    d = torch.load(filepath)
    imgs = d["imgs"].squeeze().numpy()
    depth_pred = d["depth_pred"].squeeze().numpy()
    depth_gt = d["depth_gt"].squeeze().numpy()
    mask = d["mask"].squeeze().bool().numpy()
    err_map = np.abs(depth_pred - depth_gt)

    # Always get first image as (3, H, W)
    if imgs.ndim == 4:
        if imgs.shape[1] == 3:
            img0 = imgs[0]
        elif imgs.shape[-1] == 3:
            img0 = np.transpose(imgs[0], (2, 0, 1))
        else:
            raise ValueError("Could not interpret image shape: {}".format(imgs.shape))
    elif imgs.ndim == 3:
        img0 = imgs
    else:
        raise ValueError("Could not interpret image shape: {}".format(imgs.shape))

    if img0.shape[0] == 3:
        img0 = np.transpose(img0, (1, 2, 0))
    if img0.min() < 0 or img0.max() > 1.1:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img0 = (img0 * std) + mean
    img0 = np.clip(img0, 0, 1)

    depth_pred_vis = depth_pred * to_mm
    depth_gt_vis = depth_gt * to_mm
    err_map_vis = err_map * to_mm

    valid = mask & np.isfinite(depth_gt_vis) & np.isfinite(depth_pred_vis)
    if np.any(valid):
        combined_min = min(np.nanmin(depth_gt_vis[valid]), np.nanmin(depth_pred_vis[valid]))
        combined_max = max(np.nanmax(depth_gt_vis[valid]), np.nanmax(depth_pred_vis[valid]))
    else:
        combined_min, combined_max = 0, 1

    # --- Get uncertainties in requested order: amini, meinert ---
    aleatoric_amini    = d.get('aleatoric_amini', None)
    aleatoric_meinert  = d.get('aleatoric_meinert', None)
    epistemic_amini    = d.get('epistemic_amini', None)
    epistemic_meinert  = d.get('epistemic_meinert', None)

    unc_maps = [
        (aleatoric_amini,   "Aleatoric (Amini) [mm]"),
        (aleatoric_meinert, "Aleatoric (Meinert) [mm]"),
        (epistemic_amini,   "Epistemic (Amini) [mm]"),
        (epistemic_meinert, "Epistemic (Meinert) [mm]"),
    ]

    # ----- Figure 1: input, GT, pred, error -----
    fig, axs = plt.subplots(1, 4, figsize=(22, 7))
    axs[0].imshow(np.clip(img0, 0, 1))
    axs[0].set_title("Input Image", fontsize=18, fontweight='bold')
    axs[0].axis("off")
    masked_imshow(axs[1], depth_gt_vis, mask, "Ground Truth [mm]", cmap="jet", vmin=combined_min, vmax=combined_max)
    masked_imshow(axs[2], depth_pred_vis, mask, "Predicted Depth [mm]", cmap="jet", vmin=combined_min, vmax=combined_max)
    masked_imshow(axs[3], err_map_vis, mask, "Abs Error [mm]", cmap="jet")
    plt.tight_layout()
    plt.show()

    # ----- Figure 2: 1x4 uncertainties -----
    fig2, axs2 = plt.subplots(1, 4, figsize=(24, 7))
    for idx, (arr, title) in enumerate(unc_maps):
        if arr is not None:
            arr = np.array(arr).squeeze()
            valid_unc = mask & np.isfinite(arr)
            if np.any(valid_unc):
                vmin = np.nanmin(arr[valid_unc])
                vmax = np.nanmax(arr[valid_unc])
            else:
                vmin, vmax = 0, 1
            masked_imshow(axs2[idx], arr, mask, title, cmap="jet", vmin=vmin, vmax=vmax)
        else:
            axs2[idx].axis("off")
    plt.tight_layout()
    plt.show()

    # ----- Additional Figure: Input, Depth Error, Aleatoric (Meinert), Epistemic (Meinert), all with capping -----
    fig3, axs3 = plt.subplots(1, 4, figsize=(24, 7))

    # Cap utility function
    def cap_percentile(arr, mask, lower=2, upper=98):
        arr_valid = arr[mask & np.isfinite(arr)]
        if arr_valid.size == 0:
            return arr
        low = np.percentile(arr_valid, lower)
        high = np.percentile(arr_valid, upper)
        arr_capped = np.clip(arr, low, high)
        return arr_capped

    # 1. Input Image
    axs3[0].imshow(np.clip(img0, 0, 1))
    axs3[0].set_title("Referenzbild", fontsize=18, fontweight='bold')
    axs3[0].axis("off")

    # 2. Depth Error (capped)
    err_map_capped = cap_percentile(err_map_vis, mask)
    masked_imshow(axs3[1], err_map_capped, mask, "Fehler Tiefenschätzung [mm]", cmap="jet")

    # 3. Aleatoric (Meinert) (capped)
    if aleatoric_meinert is not None:
        aleatoric_meinert_arr = np.array(aleatoric_meinert).squeeze()
        aleatoric_meinert_capped = cap_percentile(aleatoric_meinert_arr, mask)
        masked_imshow(axs3[2], aleatoric_meinert_capped, mask, "Aleatorische Unsicherheit [mm]", cmap="jet")
    else:
        axs3[2].axis("off")

    # 4. Epistemic (Meinert) (capped)
    if epistemic_meinert is not None:
        epistemic_meinert_arr = np.array(epistemic_meinert).squeeze()
        epistemic_meinert_capped = cap_percentile(epistemic_meinert_arr, mask)
        masked_imshow(axs3[3], epistemic_meinert_capped, mask, "Epistemische Unsicherheit [mm]", cmap="jet")
    else:
        axs3[3].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results_dir = "/home/grannemann/PycharmProjects/EMVSNet/data/results/amini"
    files = sorted(glob.glob(os.path.join(results_dir, "result_*.pt")))
    print(f"Found {len(files)} result files.")

    for fpath in files:
        print(f"Viewing: {fpath}")
        visualize_result_file(fpath, to_mm=1.0)
        break  # Show only one result by default
