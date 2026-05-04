import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.metrics import auc, roc_curve

def get_arrays_from_file(filepath, uncertainty_key):
    d = torch.load(filepath, map_location='cpu')
    pred = np.asarray(d["depth_pred"]).squeeze().astype(np.float32)
    gt = np.asarray(d["depth_gt"]).squeeze().astype(np.float32)
    mask = np.asarray(d["mask"]).squeeze().astype(bool)
    unc = np.asarray(d[uncertainty_key]).squeeze().astype(np.float32)
    err_map = np.abs(pred - gt)
    valid = mask & np.isfinite(gt) & np.isfinite(pred) & np.isfinite(unc)
    return err_map[valid], unc[valid]

def collect_all_errors_uncertainties(result_files, uncertainty_key):
    all_errs, all_uncs = [], []
    for f in result_files:
        err, unc = get_arrays_from_file(f, uncertainty_key)
        all_errs.append(err)
        all_uncs.append(unc)
    return np.concatenate(all_errs), np.concatenate(all_uncs)

def plot_roc_curves(all_data, error_threshold=4.0):
    plt.figure(figsize=(8, 6))
    for key, (error, unc) in all_data.items():
        error = np.array(error, dtype=np.float32).ravel()
        unc = np.array(unc, dtype=np.float32).ravel()
        high_error = (error > error_threshold).astype(np.uint8)
        uniq = np.unique(high_error)
        # Robust exclusion: Only accept exactly [0, 1]
        if not np.array_equal(uniq, [0, 1]):
            print(f"Skipping '{key}': high_error unique values are {uniq}, shape={high_error.shape}")
            continue
        # Extra check: skip if any value is not 0 or 1
        if np.any((high_error != 0) & (high_error != 1)):
            print(f"Skipping '{key}': non-binary values found in high_error")
            continue
        try:
            fpr, tpr, thresholds = roc_curve(high_error, unc)
            auc_roc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{key} (AUC={auc_roc:.2f})")
        except Exception as e:
            print(f"Skipping '{key}' due to exception: {e}")
            continue
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate", fontsize=14, fontweight='bold')
    plt.ylabel("True Positive Rate", fontsize=14, fontweight='bold')
    plt.title(f"ROC Curve for high-error pixels (error > {error_threshold} mm)", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc="lower right", frameon=True, prop={'weight': 'bold'})
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results_dir = "/home/grannemann/PycharmProjects/EMVSNet/data/results/amini"
    result_files = sorted(glob.glob(os.path.join(results_dir, "result_*.pt")))
    print(f"Found {len(result_files)} result files.")

    unc_keys = [
        ("aleatoric_amini", "Aleatoric (Amini)"),
        ("aleatoric_meinert", "Aleatoric (Meinert)"),
        ("epistemic_amini", "Epistemic (Amini)"),
        ("epistemic_meinert", "Epistemic (Meinert)")
    ]
    all_data = {}
    for key, label in unc_keys:
        sample = torch.load(result_files[0], map_location='cpu')
        if key in sample:
            err, unc = collect_all_errors_uncertainties(result_files, key)
            all_data[label] = (err, unc)
        else:
            print(f"Key '{key}' not found in sample file. Skipping.")

    plot_roc_curves(all_data, error_threshold=4.0)
