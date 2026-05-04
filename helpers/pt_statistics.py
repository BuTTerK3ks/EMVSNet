import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from sklearn.metrics import auc, roc_curve

def get_arrays_from_file(filepath, uncertainty_key):
    d = torch.load(filepath)
    pred = np.asarray(d["depth_pred"].squeeze().cpu().numpy(), dtype=np.float32)
    gt = np.asarray(d["depth_gt"].squeeze().cpu().numpy(), dtype=np.float32)
    mask = np.asarray(d["mask"].squeeze().cpu().numpy(), dtype=bool)
    unc = np.asarray(d[uncertainty_key].squeeze().cpu().numpy(), dtype=np.float32)
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

def compute_precision_recall(error, uncertainty, error_threshold=4.0, n_thresh=50):
    # --- fix start
    error = np.asarray(error, dtype=np.float32).ravel()
    uncertainty = np.asarray(uncertainty, dtype=np.float32).ravel()
    high_error = (error > error_threshold).astype(np.uint8)
    # --- fix end
    thresholds = np.linspace(np.nanmin(uncertainty), np.nanmax(uncertainty), n_thresh)
    precisions, recalls = [], []
    for t in thresholds:
        predicted_high_error = (uncertainty > t).astype(np.uint8)
        tp = np.sum((predicted_high_error == 1) & (high_error == 1))
        fp = np.sum((predicted_high_error == 1) & (high_error == 0))
        fn = np.sum((predicted_high_error == 0) & (high_error == 1))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        precisions.append(precision)
        recalls.append(recall)
    return np.array(precisions), np.array(recalls), thresholds

def plot_pr_curves(all_data, error_threshold=4.0):
    plt.figure(figsize=(8, 6))
    for key, (error, unc) in all_data.items():
        precision, recall, thresholds = compute_precision_recall(error, unc, error_threshold)
        sorted_idx = np.argsort(recall)
        recall_sorted = recall[sorted_idx]
        precision_sorted = precision[sorted_idx]
        auc_pr = auc(recall_sorted, precision_sorted)
        plt.plot(recall_sorted, precision_sorted, label=f"{key} (AUC={auc_pr:.2f})")
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title(f"Precision-Recall for high-error pixels (error > {error_threshold} mm)", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()


def plot_roc_curves(all_data, error_threshold=4.0):
    plt.figure(figsize=(8, 6))
    for key, (error, unc) in all_data.items():
        error = np.array(error, dtype=np.float32).copy()
        unc = np.array(unc, dtype=np.float32).copy()
        # --- definitive fix for binary mask ---
        high_error = np.zeros_like(error, dtype=np.uint8)
        high_error[error > error_threshold] = 1
        uniq = np.unique(high_error)
        if uniq.shape[0] != 2:
            print(f"Warning: '{key}' has only one class in high_error (unique={uniq}), skipping.")
            continue
        # --- end fix ---
        fpr, tpr, thresholds = roc_curve(high_error, unc)
        auc_roc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{key} (AUC={auc_roc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title(f"ROC Curve for high-error pixels (error > {error_threshold} mm)", fontsize=16)
    plt.legend(fontsize=12, loc="lower right")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results_dir = "/home/grannemann/PycharmProjects/EMVSNet/data/results/meinert"
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
        if key in torch.load(result_files[0]):
            err, unc = collect_all_errors_uncertainties(result_files, key)
            all_data[label] = (err, unc)

    # plot_pr_curves(all_data, error_threshold=4.0)
    plot_roc_curves(all_data, error_threshold=4.0)
