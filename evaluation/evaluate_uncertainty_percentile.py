import argparse
import os
import sys
import json
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_curve, auc
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

parser = argparse.ArgumentParser(description='Evaluate uncertainty thresholds using percentile-based calibration: find best percentile on validation, apply per-sample on test')
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
parser.add_argument('--percentile_min', type=float, default=50.0, help='Minimum percentile to test (default: 50)')
parser.add_argument('--percentile_max', type=float, default=99.5, help='Maximum percentile to test (default: 99.5)')
parser.add_argument('--percentile_step', type=float, default=2.0, help='Step size for percentile search (default: 2.0)')
parser.add_argument('--error_thresholds', type=int, nargs='+', default=[2, 4, 16, 32],
                    help='Error thresholds in mm to use for defining faulty pixels (default: 2 4 16 32)')

args = parser.parse_args()

if args.trainpath is None:
    raise ValueError("--trainpath argument is required. Please provide path to training data directory (needed for ground truth).")
if args.vallist is None:
    raise ValueError("--vallist argument is required. Please provide path to validation list file.")
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
def collect_errors_uncertainties_per_sample(model, dataloader, evidential_method):
    """
    Collect errors and uncertainties from model predictions per sample, restricted to valid regions.
    
    Returns per-sample arrays preserving sample boundaries (for percentile-based thresholds).
    
    Returns:
        errors_list: list of numpy arrays, one per sample (valid pixels only)
        aleatoric_list: list of numpy arrays, one per sample (valid pixels only)
        epistemic_list: list of numpy arrays, one per sample (valid pixels only)
    """
    model.eval()
    errors_list = []
    aleatoric_list = []
    epistemic_list = []
    
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
                    errors_list.append(err_b[mask_b])
                    aleatoric_list.append(ale_b[mask_b])
                    epistemic_list.append(epi_b[mask_b])
    
    return errors_list, aleatoric_list, epistemic_list


def compute_combined_uncertainty(aleatoric, epistemic, ale_min, ale_max, epi_min, epi_max):
    """
    Compute combined uncertainty: aleatoric and epistemic each normalized to [0, 1], then summed.
    Uses min-max normalization with the provided bounds (e.g. from validation set).
    
    Returns:
        combined: array of shape same as aleatoric, values in [0, 2]
    """
    eps = 1e-8
    ale_norm = (aleatoric - ale_min) / (ale_max - ale_min + eps)
    epi_norm = (epistemic - epi_min) / (epi_max - epi_min + eps)
    ale_norm = np.clip(ale_norm, 0, 1)
    epi_norm = np.clip(epi_norm, 0, 1)
    return ale_norm + epi_norm


def find_best_percentile(errors_list, uncertainties_list, error_threshold, percentile_min=50.0, percentile_max=99.5, percentile_step=2.0, uncertainty_type=''):
    """
    Find the percentile that maximizes F1 score for detecting high-error pixels.
    
    For each candidate percentile, applies that percentile per-sample as threshold,
    then aggregates TP/FP/FN/TN across all samples.
    
    Args:
        errors_list: list of numpy arrays (one per sample)
        uncertainties_list: list of numpy arrays (one per sample)
        error_threshold: threshold for classifying high-error pixels
        percentile_min: minimum percentile to test
        percentile_max: maximum percentile to test
        percentile_step: step size for percentile search
        uncertainty_type: string identifier for debugging
    
    Returns:
        best_percentile: percentile with maximum F1 score
        best_f1: maximum F1 score
        diagnostics: dict with precision, recall, and other info
    """
    # Generate candidate percentiles
    percentiles = np.arange(percentile_min, percentile_max + percentile_step/2, percentile_step)
    
    best_f1 = -1
    best_percentile = None
    best_precision = 0
    best_recall = 0
    best_tp = 0
    best_fp = 0
    best_fn = 0
    best_tn = 0
    
    for p in percentiles:
        # Aggregate TP/FP/FN/TN across all samples
        tp_total = 0
        fp_total = 0
        fn_total = 0
        tn_total = 0
        
        for errors, uncertainties in zip(errors_list, uncertainties_list):
            errors = np.asarray(errors, dtype=np.float32)
            uncertainties = np.asarray(uncertainties, dtype=np.float32)
            
            # Remove NaN/Inf
            valid_mask = np.isfinite(errors) & np.isfinite(uncertainties)
            errors = errors[valid_mask]
            uncertainties = uncertainties[valid_mask]
            
            if len(errors) == 0:
                continue
            
            # Ground truth: high_error = 1 if error > error_threshold
            high_error = (errors > error_threshold).astype(np.uint8)
            
            # Per-sample threshold: percentile of this sample's uncertainties
            if len(uncertainties) > 0:
                threshold = np.percentile(uncertainties, p)
                predicted_high_error = (uncertainties > threshold).astype(np.uint8)
                
                tp = np.sum((predicted_high_error == 1) & (high_error == 1))
                fp = np.sum((predicted_high_error == 1) & (high_error == 0))
                fn = np.sum((predicted_high_error == 0) & (high_error == 1))
                tn = np.sum((predicted_high_error == 0) & (high_error == 0))
                
                tp_total += tp
                fp_total += fp
                fn_total += fn
                tn_total += tn
        
        if tp_total + fp_total + fn_total + tn_total == 0:
            continue
        
        precision = tp_total / (tp_total + fp_total + 1e-8)
        recall = tp_total / (tp_total + fn_total + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_percentile = float(p)
            best_precision = precision
            best_recall = recall
            best_tp = tp_total
            best_fp = fp_total
            best_fn = fn_total
            best_tn = tn_total
    
    if best_percentile is None:
        print(f"Warning: Could not find valid percentile for {uncertainty_type} uncertainty at {error_threshold}mm.")
        return None, 0.0, {}
    
    diagnostics = {
        'precision': float(best_precision),
        'recall': float(best_recall),
        'tp': int(best_tp),
        'fp': int(best_fp),
        'fn': int(best_fn),
        'tn': int(best_tn),
    }
    
    return best_percentile, best_f1, diagnostics


def compute_roc_auc(errors_list, uncertainties_list, error_threshold):
    """
    Compute ROC AUC score by concatenating all samples.
    
    Args:
        errors_list: list of numpy arrays (one per sample)
        uncertainties_list: list of numpy arrays (one per sample)
        error_threshold: threshold for classifying high-error pixels
    
    Returns:
        roc_auc: ROC AUC score
    """
    # Concatenate all samples for ROC computation
    all_errors = np.concatenate([np.asarray(e, dtype=np.float32) for e in errors_list if len(e) > 0])
    all_uncertainties = np.concatenate([np.asarray(u, dtype=np.float32) for u in uncertainties_list if len(u) > 0])
    
    if len(all_errors) == 0:
        return None
    
    # Remove NaN/Inf
    valid_mask = np.isfinite(all_errors) & np.isfinite(all_uncertainties)
    all_errors = all_errors[valid_mask]
    all_uncertainties = all_uncertainties[valid_mask]
    
    if len(all_errors) == 0:
        return None
    
    # Create binary labels
    high_error = (all_errors > error_threshold).astype(np.uint8)
    
    if len(np.unique(high_error)) < 2:
        return None
    
    try:
        fpr, tpr, _ = roc_curve(high_error, all_uncertainties)
        roc_auc_score = auc(fpr, tpr)
        return float(roc_auc_score)
    except Exception as e:
        print(f"Warning: Could not compute ROC AUC: {e}")
        return None


def evaluate_with_percentile(errors_list, uncertainties_list, percentile, error_threshold):
    """
    Evaluate performance using a specific percentile applied per-sample.
    
    Args:
        errors_list: list of numpy arrays (one per sample)
        uncertainties_list: list of numpy arrays (one per sample)
        percentile: percentile to use as threshold per sample
        error_threshold: threshold for classifying high-error pixels
    
    Returns:
        metrics: dict with precision, recall, F1, accuracy, and ROC AUC
    """
    tp_total = 0
    fp_total = 0
    fn_total = 0
    tn_total = 0
    
    for errors, uncertainties in zip(errors_list, uncertainties_list):
        errors = np.asarray(errors, dtype=np.float32)
        uncertainties = np.asarray(uncertainties, dtype=np.float32)
        
        # Remove NaN/Inf
        valid_mask = np.isfinite(errors) & np.isfinite(uncertainties)
        errors = errors[valid_mask]
        uncertainties = uncertainties[valid_mask]
        
        if len(errors) == 0:
            continue
        
        high_error = (errors > error_threshold).astype(np.uint8)
        
        # Per-sample threshold
        if len(uncertainties) > 0:
            threshold = np.percentile(uncertainties, percentile)
            predicted_high_error = (uncertainties > threshold).astype(np.uint8)
            
            tp = np.sum((predicted_high_error == 1) & (high_error == 1))
            fp = np.sum((predicted_high_error == 1) & (high_error == 0))
            fn = np.sum((predicted_high_error == 0) & (high_error == 1))
            tn = np.sum((predicted_high_error == 0) & (high_error == 0))
            
            tp_total += tp
            fp_total += fp
            fn_total += fn
            tn_total += tn
    
    if tp_total + fp_total + fn_total + tn_total == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
        }
    
    precision = tp_total / (tp_total + fp_total + 1e-8)
    recall = tp_total / (tp_total + fn_total + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (tp_total + tn_total) / (tp_total + fp_total + fn_total + tn_total + 1e-8)
    
    # Compute ROC AUC (concatenated)
    roc_auc_score = compute_roc_auc(errors_list, uncertainties_list, error_threshold)
    
    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
    }
    
    if roc_auc_score is not None:
        metrics['roc_auc'] = roc_auc_score
    
    return metrics


def main():
    # Load model
    print(f'Loading model from {args.loadckpt}')
    model = EMVSNet(
        disparity_level=args.numdepth,
        image_scale=args.image_scale,
        max_h=args.max_h,
        max_w=args.max_w,
        return_depth=False,
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
    MVSDataset = find_dataset_def(args.dataset)
    val_dataset = MVSDataset(
        args.trainpath, args.vallist, "val", args.view_num, args.numdepth,
        args.interval_scale, args.inverse_depth, args.origin_size,
        args.light_idx, args.image_scale
    )
    test_dataset = MVSDataset(
        args.trainpath, args.testlist, "test", args.view_num, args.numdepth,
        args.interval_scale, args.inverse_depth, args.origin_size,
        args.light_idx, args.image_scale
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, drop_last=False, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, drop_last=False, pin_memory=True
    )
    
    print(f'Validation dataset: {len(val_dataset)} samples')
    print(f'Test dataset: {len(test_dataset)} samples')
    
    # Collect errors and uncertainties per sample from validation set
    print('\nCollecting errors and uncertainties from validation set (per-sample)...')
    val_errors_list, val_aleatoric_list, val_epistemic_list = collect_errors_uncertainties_per_sample(
        model, val_loader, args.evidential_method
    )
    total_val_pixels = sum(len(e) for e in val_errors_list)
    print(f'Validation set: {len(val_errors_list)} samples, {total_val_pixels} pixels')
    
    # Collect errors and uncertainties per sample from test set
    print('\nCollecting errors and uncertainties from test set (per-sample)...')
    test_errors_list, test_aleatoric_list, test_epistemic_list = collect_errors_uncertainties_per_sample(
        model, test_loader, args.evidential_method
    )
    total_test_pixels = sum(len(e) for e in test_errors_list)
    print(f'Test set: {len(test_errors_list)} samples, {total_test_pixels} pixels')
    
    # Initialize results structure
    results = {
        'evidential_method': args.evidential_method,
        'checkpoint_path': args.loadckpt,
        'dataset_path': args.trainpath,
        'val_list': args.vallist,
        'test_list': args.testlist,
        'percentile_search': {
            'min': args.percentile_min,
            'max': args.percentile_max,
            'step': args.percentile_step
        },
        'error_thresholds': {}
    }
    
    # Calibrate percentiles and evaluate for each error threshold
    for error_threshold in args.error_thresholds:
        print(f'\n=== Error Threshold: {error_threshold}mm ===')
        
        # Find best percentiles on validation set
        print(f'Calibrating aleatoric percentile on validation set...')
        best_aleatoric_percentile, best_aleatoric_f1, aleatoric_diagnostics = find_best_percentile(
            val_errors_list, val_aleatoric_list, error_threshold,
            args.percentile_min, args.percentile_max, args.percentile_step, uncertainty_type='aleatoric'
        )
        
        print(f'Calibrating epistemic percentile on validation set...')
        best_epistemic_percentile, best_epistemic_f1, epistemic_diagnostics = find_best_percentile(
            val_errors_list, val_epistemic_list, error_threshold,
            args.percentile_min, args.percentile_max, args.percentile_step, uncertainty_type='epistemic'
        )
        
        if best_aleatoric_percentile is None or best_epistemic_percentile is None:
            print(f'Warning: Could not calibrate percentiles for {error_threshold}mm error threshold. Skipping.')
            continue
        
        print(f'Best aleatoric percentile: {best_aleatoric_percentile:.2f} (validation P: {aleatoric_diagnostics["precision"]:.4f}, R: {aleatoric_diagnostics["recall"]:.4f}, F1: {best_aleatoric_f1:.4f})')
        print(f'Best epistemic percentile: {best_epistemic_percentile:.2f} (validation P: {epistemic_diagnostics["precision"]:.4f}, R: {epistemic_diagnostics["recall"]:.4f}, F1: {best_epistemic_f1:.4f})')
        
        # Combined uncertainty: normalize aleatoric and epistemic to [0,1] each, then add
        # Use pooled validation min/max for normalization
        val_aleatoric_pooled = np.concatenate(val_aleatoric_list)
        val_epistemic_pooled = np.concatenate(val_epistemic_list)
        ale_min, ale_max = np.nanmin(val_aleatoric_pooled), np.nanmax(val_aleatoric_pooled)
        epi_min, epi_max = np.nanmin(val_epistemic_pooled), np.nanmax(val_epistemic_pooled)
        
        val_combined_list = [
            compute_combined_uncertainty(ale, epi, ale_min, ale_max, epi_min, epi_max)
            for ale, epi in zip(val_aleatoric_list, val_epistemic_list)
        ]
        print(f'Calibrating combined (normalized ale+epi) percentile on validation set...')
        best_combined_percentile, best_combined_f1, combined_diagnostics = find_best_percentile(
            val_errors_list, val_combined_list, error_threshold,
            args.percentile_min, args.percentile_max, args.percentile_step, uncertainty_type='combined'
        )
        if best_combined_percentile is not None:
            print(f'Best combined percentile: {best_combined_percentile:.2f} (validation P: {combined_diagnostics["precision"]:.4f}, R: {combined_diagnostics["recall"]:.4f}, F1: {best_combined_f1:.4f})')
        
        # Evaluate on test set with best percentiles
        print(f'Evaluating on test set with calibrated percentiles (applied per-sample)...')
        aleatoric_test_metrics = evaluate_with_percentile(
            test_errors_list, test_aleatoric_list, best_aleatoric_percentile, error_threshold
        )
        epistemic_test_metrics = evaluate_with_percentile(
            test_errors_list, test_epistemic_list, best_epistemic_percentile, error_threshold
        )
        
        test_combined_list = [
            compute_combined_uncertainty(ale, epi, ale_min, ale_max, epi_min, epi_max)
            for ale, epi in zip(test_aleatoric_list, test_epistemic_list)
        ]
        combined_test_metrics = None
        if best_combined_percentile is not None:
            combined_test_metrics = evaluate_with_percentile(
                test_errors_list, test_combined_list, best_combined_percentile, error_threshold
            )
        
        result_entry = {
            'aleatoric': {
                'best_percentile': float(best_aleatoric_percentile),
                'validation_f1': float(best_aleatoric_f1),
                'validation_diagnostics': aleatoric_diagnostics,
                'test_metrics': aleatoric_test_metrics
            },
            'epistemic': {
                'best_percentile': float(best_epistemic_percentile),
                'validation_f1': float(best_epistemic_f1),
                'validation_diagnostics': epistemic_diagnostics,
                'test_metrics': epistemic_test_metrics
            }
        }
        if best_combined_percentile is not None and combined_test_metrics is not None:
            result_entry['combined'] = {
                'best_percentile': float(best_combined_percentile),
                'validation_f1': float(best_combined_f1),
                'validation_diagnostics': combined_diagnostics,
                'test_metrics': combined_test_metrics,
                'normalization': {'ale_min': float(ale_min), 'ale_max': float(ale_max),
                                 'epi_min': float(epi_min), 'epi_max': float(epi_max)}
            }
        results['error_thresholds'][f'{error_threshold}mm'] = result_entry
        
        roc_a = aleatoric_test_metrics.get('roc_auc')
        roc_e = epistemic_test_metrics.get('roc_auc')
        roc_a_str = f'{roc_a:.4f}' if roc_a is not None else 'N/A'
        roc_e_str = f'{roc_e:.4f}' if roc_e is not None else 'N/A'
        print(f'Aleatoric test metrics: ROC={roc_a_str}, Precision={aleatoric_test_metrics["precision"]:.4f}, Recall={aleatoric_test_metrics["recall"]:.4f}, F1={aleatoric_test_metrics["f1"]:.4f}, Acc={aleatoric_test_metrics["accuracy"]:.4f}')
        print(f'Epistemic test metrics: ROC={roc_e_str}, Precision={epistemic_test_metrics["precision"]:.4f}, Recall={epistemic_test_metrics["recall"]:.4f}, F1={epistemic_test_metrics["f1"]:.4f}, Acc={epistemic_test_metrics["accuracy"]:.4f}')
        if combined_test_metrics is not None:
            roc_c = combined_test_metrics.get('roc_auc')
            roc_c_str = f'{roc_c:.4f}' if roc_c is not None else 'N/A'
            print(f'Combined (norm ale+epi) test metrics: ROC={roc_c_str}, Precision={combined_test_metrics["precision"]:.4f}, Recall={combined_test_metrics["recall"]:.4f}, F1={combined_test_metrics["f1"]:.4f}, Acc={combined_test_metrics["accuracy"]:.4f}')
    
    # Save results
    checkpoint_name = os.path.basename(args.loadckpt).replace('.ckpt', '')
    output_file = os.path.join(args.outdir, f'uncertainty_evaluation_percentile_{checkpoint_name}.json')
    
    with open(output_file, 'w') as f:
        json.dump(_json_serializable(results), f, indent=2)
    
    print(f'\nResults saved to {output_file}')
    
    # Print summary
    print('\n=== Summary ===')
    for error_thresh in args.error_thresholds:
        key = f'{error_thresh}mm'
        if key in results['error_thresholds']:
            print(f'\nError Threshold: {error_thresh}mm')
            aleatoric = results['error_thresholds'][key]['aleatoric']
            epistemic = results['error_thresholds'][key]['epistemic']
            roc_a = aleatoric["test_metrics"].get("roc_auc")
            roc_e = epistemic["test_metrics"].get("roc_auc")
            print(f'  Aleatoric - Percentile: {aleatoric["best_percentile"]:.2f}')
            print(f'    ROC: {f"{roc_a:.4f}" if roc_a is not None else "N/A"}, '
                  f'Precision: {aleatoric["test_metrics"]["precision"]:.4f}, Recall: {aleatoric["test_metrics"]["recall"]:.4f}, F1: {aleatoric["test_metrics"]["f1"]:.4f}, Acc: {aleatoric["test_metrics"]["accuracy"]:.4f}')
            print(f'  Epistemic - Percentile: {epistemic["best_percentile"]:.2f}')
            print(f'    ROC: {f"{roc_e:.4f}" if roc_e is not None else "N/A"}, '
                  f'Precision: {epistemic["test_metrics"]["precision"]:.4f}, Recall: {epistemic["test_metrics"]["recall"]:.4f}, F1: {epistemic["test_metrics"]["f1"]:.4f}, Acc: {epistemic["test_metrics"]["accuracy"]:.4f}')
            if 'combined' in results['error_thresholds'][key]:
                combined = results['error_thresholds'][key]['combined']
                roc_c = combined["test_metrics"].get("roc_auc")
                print(f'  Combined (norm ale+epi) - Percentile: {combined["best_percentile"]:.2f}')
                print(f'    ROC: {f"{roc_c:.4f}" if roc_c is not None else "N/A"}, '
                      f'Precision: {combined["test_metrics"]["precision"]:.4f}, Recall: {combined["test_metrics"]["recall"]:.4f}, F1: {combined["test_metrics"]["f1"]:.4f}, Acc: {combined["test_metrics"]["accuracy"]:.4f}')


if __name__ == '__main__':
    main()
