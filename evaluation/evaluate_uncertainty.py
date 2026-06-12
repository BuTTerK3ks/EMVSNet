import argparse
import os
import sys
import json
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score
from tqdm import tqdm
from collections import OrderedDict
import ast

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import *
from evidential.models import loss_der

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Evaluate uncertainty thresholds: calibrate on validation set, evaluate ROC/precision/recall/F1/Acc on test set')
parser.add_argument('--inverse_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)
parser.add_argument('--origin_size', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)

parser.add_argument('--max_h', type=int, default=512, help='Maximum image height')
parser.add_argument('--max_w', type=int, default=640, help='Maximum image width')
parser.add_argument('--light_idx', type=int, default=3, help='select while in test')
parser.add_argument('--view_num', type=int, default=5, help='number of views')
parser.add_argument('--image_scale', type=float, default=0.25, help='pred depth map scale')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train/val datapath (for ground truth data)')
parser.add_argument('--vallist', help='validation list file')
parser.add_argument('--testlist', help='test list file')

parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')

parser.add_argument('--loadckpt', required=True, help='load checkpoint (required)')
parser.add_argument('--outdir', default='./evaluation/results', help='output directory for results')
parser.add_argument('--evidential_method', type=str, default='der', choices=['der', 'sder'],
                    help='Evidential method: der (full NIG loss) or sder (simplified, alpha = nu + 1)')
parser.add_argument('--n_thresh', type=int, default=2000, help='Max number of thresholds to evaluate when subsampling (uses unique values when fewer)')
parser.add_argument('--error_thresholds', type=int, nargs='+', default=[2, 4, 8, 16, 32],
                    help='Error thresholds in mm to use for defining faulty pixels (default: 2 4 8 16 32)')
parser.add_argument('--auprc_only', action='store_true',
                    help='Only compute AUPRC (skip threshold calibration and F1/precision/recall/accuracy at threshold). Faster; validation set not needed.')

args = parser.parse_args()

if args.trainpath is None:
    raise ValueError("--trainpath argument is required. Please provide path to training data directory (needed for ground truth).")
if args.vallist is None and not args.auprc_only:
    raise ValueError("--vallist argument is required (unless --auprc_only). Please provide path to validation list file.")
if args.testlist is None:
    raise ValueError("--testlist argument is required. Please provide path to test list file.")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.makedirs(args.outdir, exist_ok=True)

print_args(args)


def _json_serializable(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serializable(v) for v in obj]
    return obj


@make_nograd_func
def collect_errors_uncertainties(model, dataloader, evidential_method):
    """
    Collect errors and uncertainties from model predictions, restricted to valid regions.
    
    Uses the dataset mask (depth in valid range) plus finite-depth checks. Only pixels
    in valid masked regions are included for threshold calibration and metric evaluation.
    
    Returns:
        errors: numpy array of absolute depth errors (valid pixels only)
        aleatoric: numpy array of aleatoric uncertainty values (valid pixels only)
        epistemic: numpy array of epistemic uncertainty values (valid pixels only)
    """
    model.eval()
    all_errors = []
    all_aleatoric = []
    all_epistemic = []
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(dataloader, desc="Processing batches", unit="batch")):
            sample_cuda = tocuda(sample, non_blocking=True)
            depth_gt = sample_cuda["depth"]
            mask = sample_cuda["mask"]
            
            probability_volume, evidential, probabilities = model(
                sample_cuda["imgs"], 
                sample_cuda["proj_matrices"], 
                sample_cuda["depth_values"]
            )
            
            # Ensure evidential outputs and depth_gt/mask have matching spatial dimensions
            if evidential.shape[-2:] != depth_gt.shape[-2:]:
                evidential = F.interpolate(
                    evidential,
                    size=depth_gt.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )
                probability_volume = F.interpolate(
                    probability_volume,
                    size=depth_gt.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )
            
            # Ensure mask matches depth_gt spatial dimensions
            if mask.shape[-2:] != depth_gt.shape[-2:]:
                mask = F.interpolate(
                    mask.unsqueeze(1).float(),
                    size=depth_gt.shape[-2:],
                    mode='nearest',
                    align_corners=False
                ).squeeze(1).bool()
            
            outputs = {
                "probability_volume": probability_volume,
                'evidential_prediction': evidential
            }
            
            depth_value = sample_cuda["depth_values"]
            _, depth_est, evidential_outputs = loss_der(
                outputs, depth_gt, mask, depth_value,
                method=evidential_method, weight_reg=1.0
            )
            
            # Compute errors (only on valid masked regions)
            error_map = (depth_est - depth_gt).abs()
            error_np = error_map.cpu().numpy()
            mask_np = mask.cpu().numpy()
            depth_gt_np = depth_gt.cpu().numpy()
            depth_est_np = depth_est.cpu().numpy()
            
            # Extract uncertainties
            if evidential_method == 'sder':
                aleatoric = evidential_outputs["aleatoric_sder"].cpu().numpy()
                epistemic = evidential_outputs["epistemic_sder"].cpu().numpy()
            else:  # der
                aleatoric = evidential_outputs["aleatoric_der"].cpu().numpy()
                epistemic = evidential_outputs["epistemic_der"].cpu().numpy()
            
            # Build valid region mask: dataset mask (depth in range) + finite depth
            for b in range(error_np.shape[0]):
                if mask_np.ndim == 4:  # (B, 1, H, W)
                    mask_b = (mask_np[b, 0] > 0.5).squeeze()
                else:  # (B, H, W)
                    mask_b = (mask_np[b] > 0.5).squeeze()
                # Exclude invalid depths
                depth_gt_b = depth_gt_np[b, 0] if depth_gt_np.ndim == 4 else depth_gt_np[b]
                depth_est_b = depth_est_np[b, 0] if depth_est_np.ndim == 4 else depth_est_np[b]
                valid_depth = np.isfinite(depth_gt_b) & np.isfinite(depth_est_b) & (depth_gt_b > 0)
                mask_b = mask_b & valid_depth
                if np.any(mask_b):
                    err_b = error_np[b, 0] if error_np.ndim == 4 else error_np[b]
                    ale_b = aleatoric[b, 0] if aleatoric.ndim == 4 else aleatoric[b]
                    epi_b = epistemic[b, 0] if epistemic.ndim == 4 else epistemic[b]
                    all_errors.append(err_b[mask_b])
                    all_aleatoric.append(ale_b[mask_b])
                    all_epistemic.append(epi_b[mask_b])
    
    # Concatenate all arrays (only valid masked pixels)
    errors = np.concatenate(all_errors) if all_errors else np.array([])
    aleatoric = np.concatenate(all_aleatoric) if all_aleatoric else np.array([])
    epistemic = np.concatenate(all_epistemic) if all_epistemic else np.array([])
    
    return errors, aleatoric, epistemic


def compute_combined_uncertainty(aleatoric, epistemic, evidential_method):
    """
    Compute combined uncertainty via direct variance addition (statistically sound).
    
    DER: Both aleatoric and epistemic are variances. Total predictive variance =
         aleatoric + epistemic. Combined uncertainty = sqrt(aleatoric + epistemic).
    
    SDER: Aleatoric is std, epistemic is 1/sqrt(nu). Combine in quadrature:
          sqrt(aleatoric**2 + epistemic**2).
    
    Returns:
        combined: array of shape same as aleatoric (total uncertainty as std)
    """
    eps = 1e-8
    if evidential_method == 'der':
        # Both are variances: total_var = aleatoric + epistemic
        return np.sqrt(np.maximum(aleatoric + epistemic, eps))
    else:  # sder
        # Aleatoric is std, epistemic is 1/sqrt(nu): combine in quadrature
        return np.sqrt(np.maximum(aleatoric**2 + epistemic**2, eps))


def find_best_uncertainty_threshold(errors, uncertainties, error_threshold, n_thresh=2000, uncertainty_type=''):
    """
    Find the uncertainty threshold that maximizes F1 score for detecting high-error pixels.
    
    Method: The prediction (uncertainty > t) only changes when t crosses an actual uncertainty
    value. We use unique sorted uncertainty values as candidate thresholds to find the true
    optimum. If too many unique values exist, we subsample via percentiles.
    
    Args:
        errors: numpy array of absolute depth errors
        uncertainties: numpy array of uncertainty values (aleatoric or epistemic)
        error_threshold: threshold for classifying high-error pixels (e.g., 2mm or 4mm)
        n_thresh: max number of thresholds to evaluate (for subsampling when many unique values)
        uncertainty_type: string identifier for debugging ('aleatoric' or 'epistemic')
    
    Returns:
        best_threshold: uncertainty threshold with maximum F1 score on validation set
        best_f1: maximum F1 score
        diagnostics: dict with precision, recall, and other diagnostic info
    """
    errors = np.asarray(errors, dtype=np.float32).ravel()
    uncertainties = np.asarray(uncertainties, dtype=np.float32).ravel()
    
    # Remove any NaN or Inf values
    valid_mask = np.isfinite(errors) & np.isfinite(uncertainties)
    errors = errors[valid_mask]
    uncertainties = uncertainties[valid_mask]
    
    if len(errors) == 0:
        print(f"Warning: No valid data points for {uncertainty_type} uncertainty at {error_threshold}mm.")
        return None, 0.0, {}
    
    # Ground truth: high_error = 1 if error > error_threshold (pixel is faulty)
    high_error = (errors > error_threshold).astype(np.uint8)
    
    # Check if we have both classes
    if len(np.unique(high_error)) < 2:
        print(f"Warning: Only one class present for error threshold {error_threshold}mm. Cannot compute metrics.")
        return None, 0.0, {}
    
    unc_min = np.nanmin(uncertainties)
    unc_max = np.nanmax(uncertainties)
    
    # Candidate thresholds: unique uncertainty values (prediction changes only at these points)
    unique_unc = np.unique(uncertainties)
    if len(unique_unc) > n_thresh:
        percentiles = np.linspace(0, 100, n_thresh)
        candidate_thresholds = np.unique(np.percentile(uncertainties, percentiles))
    else:
        candidate_thresholds = unique_unc
    
    # Add edge cases
    candidate_thresholds = np.unique(np.concatenate([
        [unc_min - 1e-6], candidate_thresholds, [unc_max + 1e-6]
    ]))
    thresholds = candidate_thresholds
    
    best_f1 = -1
    best_threshold = None
    best_precision = 0
    best_recall = 0
    best_tp = 0
    best_fp = 0
    best_fn = 0
    best_tn = 0
    
    for t in thresholds:
        predicted_high_error = (uncertainties > t).astype(np.uint8)
        
        tp = np.sum((predicted_high_error == 1) & (high_error == 1))
        fp = np.sum((predicted_high_error == 1) & (high_error == 0))
        fn = np.sum((predicted_high_error == 0) & (high_error == 1))
        tn = np.sum((predicted_high_error == 0) & (high_error == 0))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)
            best_precision = precision
            best_recall = recall
            best_tp = tp
            best_fp = fp
            best_fn = fn
            best_tn = tn
    
    diagnostics = {
        'precision': float(best_precision),
        'recall': float(best_recall),
        'tp': int(best_tp),
        'fp': int(best_fp),
        'fn': int(best_fn),
        'tn': int(best_tn),
        'uncertainty_range': [float(unc_min), float(unc_max)],
        'uncertainty_percentiles': {
            'p1': float(np.percentile(uncertainties, 1)),
            'p50': float(np.percentile(uncertainties, 50)),
            'p99': float(np.percentile(uncertainties, 99)),
            'min': float(unc_min),
            'max': float(unc_max)
        },
        'n_candidates_evaluated': len(candidate_thresholds),
        'high_error_rate': float(np.mean(high_error))
    }
    
    return best_threshold, best_f1, diagnostics


def compute_roc_auc(errors, uncertainties, error_threshold):
    """
    Compute ROC AUC score for uncertainty-based error detection.
    
    Args:
        errors: numpy array of absolute depth errors
        uncertainties: numpy array of uncertainty values
        error_threshold: threshold for classifying high-error pixels
    
    Returns:
        roc_auc: ROC AUC score
    """
    errors = np.asarray(errors, dtype=np.float32).ravel()
    uncertainties = np.asarray(uncertainties, dtype=np.float32).ravel()
    
    # Create binary labels: high_error = 1 if error > error_threshold
    high_error = (errors > error_threshold).astype(np.uint8)
    
    # Check if we have both classes
    if len(np.unique(high_error)) < 2:
        return None
    
    try:
        fpr, tpr, _ = roc_curve(high_error, uncertainties)
        roc_auc_score = auc(fpr, tpr)
        return float(roc_auc_score)
    except Exception as e:
        print(f"Warning: Could not compute ROC AUC: {e}")
        return None


def compute_pr_auc(errors, uncertainties, error_threshold):
    """
    Compute Area under Precision-Recall curve (AUPRC) for uncertainty-based error detection.
    
    Threshold-free metric: higher uncertainty should predict higher error.
    
    Args:
        errors: numpy array of absolute depth errors
        uncertainties: numpy array of uncertainty values
        error_threshold: threshold for classifying high-error pixels
    
    Returns:
        pr_auc: AUPRC score (average precision)
    """
    errors = np.asarray(errors, dtype=np.float32).ravel()
    uncertainties = np.asarray(uncertainties, dtype=np.float32).ravel()
    
    valid_mask = np.isfinite(errors) & np.isfinite(uncertainties)
    errors = errors[valid_mask]
    uncertainties = uncertainties[valid_mask]
    
    high_error = (errors > error_threshold).astype(np.uint8)
    
    if len(np.unique(high_error)) < 2:
        return None
    
    try:
        return float(average_precision_score(high_error, uncertainties))
    except Exception as e:
        print(f"Warning: Could not compute AUPRC: {e}")
        return None


def evaluate_with_threshold(errors, uncertainties, uncertainty_threshold, error_threshold):
    """
    Evaluate performance using a specific uncertainty threshold.
    
    Args:
        errors: numpy array of absolute depth errors
        uncertainties: numpy array of uncertainty values
        uncertainty_threshold: threshold for classifying high-uncertainty pixels
        error_threshold: threshold for classifying high-error pixels
    
    Returns:
        metrics: dict with precision, recall, F1, accuracy, and ROC AUC
    """
    errors = np.asarray(errors, dtype=np.float32).ravel()
    uncertainties = np.asarray(uncertainties, dtype=np.float32).ravel()
    
    high_error = (errors > error_threshold).astype(np.uint8)
    predicted_high_error = (uncertainties > uncertainty_threshold).astype(np.uint8)
    
    tp = np.sum((predicted_high_error == 1) & (high_error == 1))
    fp = np.sum((predicted_high_error == 1) & (high_error == 0))
    fn = np.sum((predicted_high_error == 0) & (high_error == 1))
    tn = np.sum((predicted_high_error == 0) & (high_error == 0))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    
    # Compute ROC AUC and AUPRC (threshold-free metrics)
    roc_auc_score = compute_roc_auc(errors, uncertainties, error_threshold)
    pr_auc_score = compute_pr_auc(errors, uncertainties, error_threshold)
    
    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
    }
    
    if roc_auc_score is not None:
        metrics['roc_auc'] = roc_auc_score
    if pr_auc_score is not None:
        metrics['auprc'] = pr_auc_score
    
    return metrics


def main():
    # Load model
    print(f'Loading model from {args.loadckpt}')
    model = EMVSNet(
        disparity_level=args.numdepth,
        image_scale=args.image_scale,
        max_h=args.max_h,
        max_w=args.max_w,
        return_depth=False,  # Use training mode: return (probability_volume, evidential, probabilities)
        evidential_method=args.evidential_method
    )
    
    # Load checkpoint
    state_dict = torch.load(args.loadckpt, map_location=device)
    
    # Handle both old and new checkpoint formats
    if 'model' in state_dict:
        model_state_dict = state_dict['model']
    else:
        model_state_dict = state_dict
    
    # Remove 'module.' prefix if present (from DataParallel/DDP)
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    print(f'Model loaded successfully.')
    if 'best_val_mae' in state_dict:
        print(f'Checkpoint info: Best validation MAE: {state_dict["best_val_mae"]:.3f} at epoch {state_dict.get("best_val_epoch", "N/A")}')
    
    # Create datasets
    # Use trainpath for both val and test datasets to load data with ground truth (training data structure)
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(
        args.trainpath, args.testlist, "test", args.view_num, args.numdepth,
        args.interval_scale, args.inverse_depth, args.origin_size,
        args.light_idx, args.image_scale
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, drop_last=False, pin_memory=True
    )
    print(f'Test dataset: {len(test_dataset)} samples')

    if not args.auprc_only:
        val_dataset = MVSDataset(
            args.trainpath, args.vallist, "val", args.view_num, args.numdepth,
            args.interval_scale, args.inverse_depth, args.origin_size,
            args.light_idx, args.image_scale
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, drop_last=False, pin_memory=True
        )
        print(f'Validation dataset: {len(val_dataset)} samples')
        print('\nCollecting errors and uncertainties from validation set...')
        val_errors, val_aleatoric, val_epistemic = collect_errors_uncertainties(
            model, val_loader, args.evidential_method
        )
        print(f'Validation set: {len(val_errors)} pixels')

    print('\nCollecting errors and uncertainties from test set...')
    test_errors, test_aleatoric, test_epistemic = collect_errors_uncertainties(
        model, test_loader, args.evidential_method
    )
    print(f'Test set: {len(test_errors)} pixels')
    
    # Initialize results structure
    results = {
        'evidential_method': args.evidential_method,
        'checkpoint_path': args.loadckpt,
        'dataset_path': args.trainpath,
        'test_list': args.testlist,
        'auprc_only': args.auprc_only,
        'error_thresholds': {}
    }
    if not args.auprc_only:
        results['val_list'] = args.vallist

    if args.auprc_only:
        # AUPRC-only mode: compute AUPRC on test set for each error threshold (no validation, no threshold calibration)
        test_combined = compute_combined_uncertainty(
            test_aleatoric, test_epistemic, args.evidential_method
        )

        for error_threshold in args.error_thresholds:
            print(f'\n=== Error Threshold: {error_threshold}mm ===')
            auprc_aleatoric = compute_pr_auc(test_errors, test_aleatoric, error_threshold)
            auprc_epistemic = compute_pr_auc(test_errors, test_epistemic, error_threshold)
            auprc_combined = compute_pr_auc(test_errors, test_combined, error_threshold)

            result_entry = {
                'aleatoric': {'auprc': float(auprc_aleatoric) if auprc_aleatoric is not None else None},
                'epistemic': {'auprc': float(auprc_epistemic) if auprc_epistemic is not None else None},
                'combined': {'auprc': float(auprc_combined) if auprc_combined is not None else None}
            }
            results['error_thresholds'][f'{error_threshold}mm'] = result_entry

            a_str = f'{auprc_aleatoric:.4f}' if auprc_aleatoric is not None else 'N/A'
            e_str = f'{auprc_epistemic:.4f}' if auprc_epistemic is not None else 'N/A'
            c_str = f'{auprc_combined:.4f}' if auprc_combined is not None else 'N/A'
            print(f'Aleatoric AUPRC: {a_str}, Epistemic AUPRC: {e_str}, Combined AUPRC: {c_str}')

    else:
        # Full mode: calibrate thresholds on validation, evaluate all metrics on test
        for error_threshold in args.error_thresholds:
            print(f'\n=== Error Threshold: {error_threshold}mm ===')
            
            # Find best thresholds on validation set
            print(f'Calibrating aleatoric threshold on validation set...')
            best_aleatoric_thresh, best_aleatoric_f1, aleatoric_diagnostics = find_best_uncertainty_threshold(
                val_errors, val_aleatoric, error_threshold, n_thresh=args.n_thresh, uncertainty_type='aleatoric'
            )
            
            print(f'Calibrating epistemic threshold on validation set...')
            best_epistemic_thresh, best_epistemic_f1, epistemic_diagnostics = find_best_uncertainty_threshold(
                val_errors, val_epistemic, error_threshold, n_thresh=args.n_thresh, uncertainty_type='epistemic'
            )
            
            if best_aleatoric_thresh is None or best_epistemic_thresh is None:
                print(f'Warning: Could not calibrate thresholds for {error_threshold}mm error threshold. Skipping.')
                continue
            
            print(f'Best aleatoric threshold: {best_aleatoric_thresh:.6f} (validation P: {aleatoric_diagnostics["precision"]:.4f}, R: {aleatoric_diagnostics["recall"]:.4f}, F1: {best_aleatoric_f1:.4f})')
            print(f'Best epistemic threshold: {best_epistemic_thresh:.6f} (validation P: {epistemic_diagnostics["precision"]:.4f}, R: {epistemic_diagnostics["recall"]:.4f}, F1: {best_epistemic_f1:.4f})')
            
            # Combined uncertainty: direct variance addition (sqrt(ale+epi) for DER, quadrature for SDER)
            val_combined = compute_combined_uncertainty(
                val_aleatoric, val_epistemic, args.evidential_method
            )
            print(f'Calibrating combined (direct variance) threshold on validation set...')
            best_combined_thresh, best_combined_f1, combined_diagnostics = find_best_uncertainty_threshold(
                val_errors, val_combined, error_threshold, n_thresh=args.n_thresh, uncertainty_type='combined'
            )
            if best_combined_thresh is not None:
                print(f'Best combined threshold: {best_combined_thresh:.6f} (validation P: {combined_diagnostics["precision"]:.4f}, R: {combined_diagnostics["recall"]:.4f}, F1: {best_combined_f1:.4f})')
            
            # Print uncertainty statistics
            if 'uncertainty_percentiles' in aleatoric_diagnostics:
                ale_perc = aleatoric_diagnostics['uncertainty_percentiles']
                print(f'  Aleatoric uncertainty stats: min={ale_perc["min"]:.6f}, '
                      f'p50={ale_perc["p50"]:.6f}, max={ale_perc["max"]:.6f}')
            if 'uncertainty_percentiles' in epistemic_diagnostics:
                epi_perc = epistemic_diagnostics['uncertainty_percentiles']
                print(f'  Epistemic uncertainty stats: min={epi_perc["min"]:.6f}, '
                      f'p50={epi_perc["p50"]:.6f}, max={epi_perc["max"]:.6f}')
            
            # Evaluate on test set with best thresholds
            print(f'Evaluating on test set with calibrated thresholds...')
            aleatoric_test_metrics = evaluate_with_threshold(
                test_errors, test_aleatoric, best_aleatoric_thresh, error_threshold
            )
            epistemic_test_metrics = evaluate_with_threshold(
                test_errors, test_epistemic, best_epistemic_thresh, error_threshold
            )
            
            test_combined = compute_combined_uncertainty(
                test_aleatoric, test_epistemic, args.evidential_method
            )
            combined_test_metrics = None
            if best_combined_thresh is not None:
                combined_test_metrics = evaluate_with_threshold(
                    test_errors, test_combined, best_combined_thresh, error_threshold
                )
            
            result_entry = {
                'aleatoric': {
                    'best_threshold': float(best_aleatoric_thresh),
                    'validation_f1': float(best_aleatoric_f1),
                    'validation_diagnostics': aleatoric_diagnostics,
                    'test_metrics': aleatoric_test_metrics
                },
                'epistemic': {
                    'best_threshold': float(best_epistemic_thresh),
                    'validation_f1': float(best_epistemic_f1),
                    'validation_diagnostics': epistemic_diagnostics,
                    'test_metrics': epistemic_test_metrics
                }
            }
            if best_combined_thresh is not None and combined_test_metrics is not None:
                result_entry['combined'] = {
                    'best_threshold': float(best_combined_thresh),
                    'validation_f1': float(best_combined_f1),
                    'validation_diagnostics': combined_diagnostics,
                    'test_metrics': combined_test_metrics,
                    'combination': 'direct_variance'
                }
            results['error_thresholds'][f'{error_threshold}mm'] = result_entry
            
            roc_a = aleatoric_test_metrics.get('roc_auc')
            roc_e = epistemic_test_metrics.get('roc_auc')
            auprc_a = aleatoric_test_metrics.get('auprc')
            auprc_e = epistemic_test_metrics.get('auprc')
            roc_a_str = f'{roc_a:.4f}' if roc_a is not None else 'N/A'
            roc_e_str = f'{roc_e:.4f}' if roc_e is not None else 'N/A'
            auprc_a_str = f'{auprc_a:.4f}' if auprc_a is not None else 'N/A'
            auprc_e_str = f'{auprc_e:.4f}' if auprc_e is not None else 'N/A'
            print(f'Aleatoric test metrics: ROC={roc_a_str}, AUPRC={auprc_a_str}, Precision={aleatoric_test_metrics["precision"]:.4f}, Recall={aleatoric_test_metrics["recall"]:.4f}, F1={aleatoric_test_metrics["f1"]:.4f}, Acc={aleatoric_test_metrics["accuracy"]:.4f}')
            print(f'Epistemic test metrics: ROC={roc_e_str}, AUPRC={auprc_e_str}, Precision={epistemic_test_metrics["precision"]:.4f}, Recall={epistemic_test_metrics["recall"]:.4f}, F1={epistemic_test_metrics["f1"]:.4f}, Acc={epistemic_test_metrics["accuracy"]:.4f}')
            if combined_test_metrics is not None:
                roc_c = combined_test_metrics.get('roc_auc')
                auprc_c = combined_test_metrics.get('auprc')
                roc_c_str = f'{roc_c:.4f}' if roc_c is not None else 'N/A'
                auprc_c_str = f'{auprc_c:.4f}' if auprc_c is not None else 'N/A'
                print(f'Combined (direct variance) test metrics: ROC={roc_c_str}, AUPRC={auprc_c_str}, Precision={combined_test_metrics["precision"]:.4f}, Recall={combined_test_metrics["recall"]:.4f}, F1={combined_test_metrics["f1"]:.4f}, Acc={combined_test_metrics["accuracy"]:.4f}')
    
    # Save results
    checkpoint_name = os.path.basename(args.loadckpt).replace('.ckpt', '')
    suffix = '_auprc_only' if args.auprc_only else ''
    output_file = os.path.join(args.outdir, f'uncertainty_evaluation_{checkpoint_name}{suffix}.json')
    
    with open(output_file, 'w') as f:
        json.dump(_json_serializable(results), f, indent=2)
    
    print(f'\nResults saved to {output_file}')
    
    # Print AUPRC to command line (compact summary)
    print('\n--- AUPRC ---')
    for error_thresh in args.error_thresholds:
        key = f'{error_thresh}mm'
        if key in results['error_thresholds']:
            ale = results['error_thresholds'][key]['aleatoric']
            epi = results['error_thresholds'][key]['epistemic']
            a = ale.get('auprc') if args.auprc_only else ale.get('test_metrics', {}).get('auprc')
            e = epi.get('auprc') if args.auprc_only else epi.get('test_metrics', {}).get('auprc')
            c = None
            if 'combined' in results['error_thresholds'][key]:
                comb = results['error_thresholds'][key]['combined']
                c = comb.get('auprc') if args.auprc_only else comb.get('test_metrics', {}).get('auprc')
            a_s = f'{a:.4f}' if a is not None else 'N/A'
            e_s = f'{e:.4f}' if e is not None else 'N/A'
            c_s = f'{c:.4f}' if c is not None else 'N/A'
            print(f'  {error_thresh}mm: aleatoric={a_s}, epistemic={e_s}, combined={c_s}')
    
    # Print summary
    print('\n=== Summary ===')
    for error_thresh in args.error_thresholds:
        key = f'{error_thresh}mm'
        if key in results['error_thresholds']:
            print(f'\nError Threshold: {error_thresh}mm')
            aleatoric = results['error_thresholds'][key]['aleatoric']
            epistemic = results['error_thresholds'][key]['epistemic']
            if args.auprc_only:
                auprc_a = aleatoric.get("auprc")
                auprc_e = epistemic.get("auprc")
                print(f'  Aleatoric AUPRC: {f"{auprc_a:.4f}" if auprc_a is not None else "N/A"}')
                print(f'  Epistemic AUPRC: {f"{auprc_e:.4f}" if auprc_e is not None else "N/A"}')
                if 'combined' in results['error_thresholds'][key]:
                    auprc_c = results['error_thresholds'][key]['combined'].get("auprc")
                    print(f'  Combined AUPRC: {f"{auprc_c:.4f}" if auprc_c is not None else "N/A"}')
            else:
                roc_a = aleatoric["test_metrics"].get("roc_auc")
                roc_e = epistemic["test_metrics"].get("roc_auc")
                auprc_a = aleatoric["test_metrics"].get("auprc")
                auprc_e = epistemic["test_metrics"].get("auprc")
                print(f'  Aleatoric - Threshold: {aleatoric["best_threshold"]:.6f}')
                print(f'    ROC: {f"{roc_a:.4f}" if roc_a is not None else "N/A"}, AUPRC: {f"{auprc_a:.4f}" if auprc_a is not None else "N/A"}, '
                      f'Precision: {aleatoric["test_metrics"]["precision"]:.4f}, Recall: {aleatoric["test_metrics"]["recall"]:.4f}, F1: {aleatoric["test_metrics"]["f1"]:.4f}, Acc: {aleatoric["test_metrics"]["accuracy"]:.4f}')
                print(f'  Epistemic - Threshold: {epistemic["best_threshold"]:.6f}')
                print(f'    ROC: {f"{roc_e:.4f}" if roc_e is not None else "N/A"}, AUPRC: {f"{auprc_e:.4f}" if auprc_e is not None else "N/A"}, '
                      f'Precision: {epistemic["test_metrics"]["precision"]:.4f}, Recall: {epistemic["test_metrics"]["recall"]:.4f}, F1: {epistemic["test_metrics"]["f1"]:.4f}, Acc: {epistemic["test_metrics"]["accuracy"]:.4f}')
                if 'combined' in results['error_thresholds'][key]:
                    combined = results['error_thresholds'][key]['combined']
                    roc_c = combined["test_metrics"].get("roc_auc")
                    auprc_c = combined["test_metrics"].get("auprc")
                    print(f'  Combined (direct variance) - Threshold: {combined["best_threshold"]:.6f}')
                    print(f'    ROC: {f"{roc_c:.4f}" if roc_c is not None else "N/A"}, AUPRC: {f"{auprc_c:.4f}" if auprc_c is not None else "N/A"}, '
                          f'Precision: {combined["test_metrics"]["precision"]:.4f}, Recall: {combined["test_metrics"]["recall"]:.4f}, F1: {combined["test_metrics"]["f1"]:.4f}, Acc: {combined["test_metrics"]["accuracy"]:.4f}')


if __name__ == '__main__':
    main()
