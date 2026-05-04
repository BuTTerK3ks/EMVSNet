import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from matplotlib import cm
from skimage.transform import resize
from realesrgan import RealESRGAN
from PIL import Image

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
    masked_imshow(axs[2], depth_pred_vis, mask, "Predicted Depth [mm]", cmap="jet", vmin=combined_min,
                  vmax=combined_max)

    valid_err = mask & np.isfinite(err_map_vis)
    if np.any(valid_err):
        vmin_err = np.nanpercentile(err_map_vis[valid_err], 5)
        vmax_err = np.nanpercentile(err_map_vis[valid_err], 95)
    else:
        vmin_err, vmax_err = 0, 1
    masked_imshow(axs[3], err_map_vis, mask, "Absolute Error [mm]", cmap="jet", vmin=vmin_err, vmax=vmax_err)

    plt.tight_layout()
    plt.show()

    # ----- Figure 2: 1x4 uncertainties -----
    fig2, axs2 = plt.subplots(1, 4, figsize=(24, 7))
    for idx, (arr, title) in enumerate(unc_maps):
        if arr is not None:
            arr = np.array(arr).squeeze()
            valid_unc = mask & np.isfinite(arr)
            if np.any(valid_unc):
                vmin = np.nanpercentile(arr[valid_unc], 5)
                vmax = np.nanpercentile(arr[valid_unc], 95)
            else:
                vmin, vmax = 0, 1
            masked_imshow(axs2[idx], arr, mask, title, cmap="jet", vmin=vmin, vmax=vmax)
        else:
            axs2[idx].axis("off")
    plt.tight_layout()
    plt.show()

    # ----- Extra Figure: mask visualization -----
    fig_mask, ax_mask = plt.subplots(figsize=(7, 7))
    ax_mask.imshow(mask, cmap="gray")
    ax_mask.set_title("Mask", fontsize=18, fontweight='bold')
    ax_mask.axis("off")
    plt.tight_layout()
    plt.show()

    # ----- Export: original image with highest 10% aleatoric_meinert overlayed (keep colormap), super-resolved -----
    if aleatoric_meinert is not None:
        from PIL import Image
        from realesrgan import RealESRGAN
        import torch

        arr = np.array(aleatoric_meinert).squeeze()
        valid_unc = mask & np.isfinite(arr)
        if np.any(valid_unc):
            threshold = np.nanpercentile(arr[valid_unc], 90)
            high_unc_mask = (arr >= threshold) & valid_unc
        else:
            high_unc_mask = np.zeros_like(arr, dtype=bool)

        img_vis = np.clip(img0.copy(), 0, 1)
        H, W, _ = img_vis.shape

        # --- Super-resolve the image using Real-ESRGAN ---
        img_pil = Image.fromarray((img_vis * 255).astype(np.uint8))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sr_model = RealESRGAN(device, scale=4)
        try:
            sr_model.load_weights('RealESRGAN_x4.pth')
        except FileNotFoundError:
            # Auto-download weights if missing
            import urllib.request
            url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRGAN_x4.pth'
            urllib.request.urlretrieve(url, 'RealESRGAN_x4.pth')
            sr_model.load_weights('RealESRGAN_x4.pth')

        img_sr = sr_model.predict(img_pil)
        img_up = np.array(img_sr) / 255.0
        out_H, out_W = img_up.shape[0], img_up.shape[1]

        # Upscale mask and uncertainty maps to super-resolved shape
        high_unc_mask_up = resize(high_unc_mask.astype(float), (out_H, out_W), order=0, preserve_range=True) > 0.5
        arr_up = resize(arr, (out_H, out_W), order=1, preserve_range=True, anti_aliasing=True)
        mask_up = resize(mask.astype(float), (out_H, out_W), order=0, preserve_range=True) > 0.5

        # Apply colormap only where high uncertainty, else show original image
        from matplotlib import cm
        cmap = cm.get_cmap('jet')
        arr_valid = arr[valid_unc]
        arr_norm = (arr_up - np.nanmin(arr_valid)) / (np.nanmax(arr_valid) - np.nanmin(arr_valid) + 1e-6)
        color_overlay = cmap(arr_norm)[..., :3]

        # Blend
        alpha = 0.6
        blend = img_up.copy()
        mask_to_paint = high_unc_mask_up & mask_up
        blend[mask_to_paint] = (
                alpha * color_overlay[mask_to_paint] +
                (1 - alpha) * img_up[mask_to_paint]
        )

        fig_overlay, ax_overlay = plt.subplots(figsize=(out_W / 100, out_H / 100), dpi=600)
        ax_overlay.imshow(blend)
        ax_overlay.set_title("Input with Aleatoric (Meinert) High Uncertainty Overlay", fontsize=18, fontweight='bold')
        ax_overlay.axis("off")
        plt.tight_layout()
        export_path = filepath.replace('.pt', '_aleatoric_meinert_high_unc_colormap_SR_overlay.png')
        plt.savefig(export_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig_overlay)
        print(f"Super-resolved overlay image exported to: {export_path}")

        # For Figure 1 (individuals)
        # Input Image
        fig_in, ax_in = plt.subplots(figsize=(7, 7))
        ax_in.imshow(np.clip(img0, 0, 1))
        ax_in.set_title("Input Image", fontsize=18, fontweight='bold')
        ax_in.axis("off")
        fig_in.savefig(filepath.replace('.pt', '_input_image.png'), dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig_in)

        # Ground Truth
        fig_gt, ax_gt = plt.subplots(figsize=(7, 7))
        masked_imshow(ax_gt, depth_gt_vis, mask, "Ground Truth [mm]", cmap="jet", vmin=combined_min, vmax=combined_max)
        fig_gt.savefig(filepath.replace('.pt', '_ground_truth.png'), dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig_gt)

        # Predicted Depth
        fig_pred, ax_pred = plt.subplots(figsize=(7, 7))
        masked_imshow(ax_pred, depth_pred_vis, mask, "Predicted Depth [mm]", cmap="jet", vmin=combined_min,
                      vmax=combined_max)
        fig_pred.savefig(filepath.replace('.pt', '_predicted_depth.png'), dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig_pred)

        # Error Map
        fig_err, ax_err = plt.subplots(figsize=(7, 7))
        masked_imshow(ax_err, err_map_vis, mask, "Absolute Error [mm]", cmap="jet", vmin=vmin_err, vmax=vmax_err)
        fig_err.savefig(filepath.replace('.pt', '_error_map.png'), dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig_err)

        # Uncertainty Maps
        for idx, (arr, title) in enumerate(unc_maps):
            if arr is not None:
                arr = np.array(arr).squeeze()
                valid_unc = mask & np.isfinite(arr)
                if np.any(valid_unc):
                    vmin = np.nanpercentile(arr[valid_unc], 5)
                    vmax = np.nanpercentile(arr[valid_unc], 95)
                else:
                    vmin, vmax = 0, 1
                fig_u, ax_u = plt.subplots(figsize=(7, 7))
                masked_imshow(ax_u, arr, mask, title, cmap="jet", vmin=vmin, vmax=vmax)
                fname = title.lower().replace(' ', '_').replace('[', '').replace(']', '').replace('(', '').replace(')',
                                                                                                                   '')
                fig_u.savefig(filepath.replace('.pt', f'_{fname}.png'), dpi=300, bbox_inches='tight', pad_inches=0.05)
                plt.close(fig_u)


if __name__ == "__main__":
    results_dir = "/home/grannemann/PycharmProjects/EMVSNet/data/results/amini"
    files = sorted(glob.glob(os.path.join(results_dir, "result_*.pt")))
    print(f"Found {len(files)} result files.")

    for fpath in files:
        print(f"Viewing: {fpath}")
        visualize_result_file(fpath, to_mm=1.0)
        break  # Show only one result by default
