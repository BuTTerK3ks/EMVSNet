import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import *
from evidential.models import uncertainty_der, uncertainty_sder
import ast
from collections import OrderedDict
from tqdm import tqdm

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Evaluate uncertainty thresholds')
parser.add_argument('--inverse_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)
parser.add_argument('--origin_size', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)

parser.add_argument('--max_h', type=int, default=512, help='Maximum image height')
parser.add_argument('--max_w', type=int, default=640, help='Maximum image width')
parser.add_argument('--light_idx', type=int, default=3, help='select while in test')
parser.add_argument('--view_num', type=int, default=5, help='validation view num setting')
parser.add_argument('--image_scale', type=float, default=0.25, help='pred depth map scale')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train/val datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--vallist', help='val list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')

parser.add_argument('--loadckpt', default=None, help='load checkpoint (best_model.ckpt)')
parser.add_argument('--outdir', default='./uncertainty_eval', help='output dir')
parser.add_argument('--evidential_method', type=str, default='der', choices=['der', 'sder'],
                    help='Evidential method: der (full NIG loss) or sder (simplified, alpha = nu + 1)')
parser.add_argument('--n_thresh', type=int, default=100, help='Number of uncertainty thresholds to test')

args = parser.parse_args()

if args.loadckpt is None:
    raise ValueError("--loadckpt argument is required. Please provide path to best_model.ckpt")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.makedirs(args.outdir, exist_ok=True)

print_args(args)

def find_best_uncertainty_threshold(errors, uncertainties, error_threshold, n_thresh=100):
    """
    Find the uncertainty threshold that maximizes F1 score for detecting high-error pixels.
    
    Args:
        errors: numpy array of absolute depth errors
        uncertainties: numpy array of uncertainty values (aleatoric or epistemic)
        error_threshold: threshold for classifying high-error pixels (4mm or 8mm)
        n_thresh: number of uncertainty thresholds to test
    
    Returns:
        best_threshold: uncertainty threshold with maximum F1 score
        best_f1: maximum F1 score
        metrics: dict with precision, recall, F1 at best threshold
    """
    errors = np.asarray(errors, dtype=np.float32).ravel()
    uncertainties = np.asarray(uncertainties, dtype=np.float32).ravel()
    
    # Create binary labels: high_error = 1 if error > error_threshold
    high_error = (errors > error_threshold).astype(np.uint8)
    
    # Test range of uncertainty thresholds
    unc_min = np.nanmin(uncertainties)
    unc_max = np.nanmax(uncertainties)
    thresholds = np.linspace(unc_min, unc_max, n_thresh)
    
    best_f1 = -1
    best_threshold = None
    best_precision = 0
    best_recall = 0
    
    for t in thresholds:
        predicted_high_error = (uncertainties > t).astype(np.uint8)
        
        tp = np.sum((predicted_high_error == 1) & (high_error == 1))
        fp = np.sum((predicted_high_error == 1) & (high_error == 0))
        fn = np.sum((predicted_high_error == 0) & (high_error == 1))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_precision = precision
            best_recall = recall
    
    metrics = {
        'threshold': float(best_threshold),
        'precision': float(best_precision),
        'recall': float(best_recall),
        'f1': float(best_f1)
    }
    
    return best_threshold, best_f1, metrics

def evaluate_with_threshold(errors, uncertainties, uncertainty_threshold, error_threshold):
    """
    Evaluate performance using a specific uncertainty threshold.
    
    Args:
        errors: numpy array of absolute depth errors
        uncertainties: numpy array of uncertainty values
        uncertainty_threshold: threshold for classifying high-uncertainty pixels
        error_threshold: threshold for classifying high-error pixels (4mm or 8mm)
    
    Returns:
        metrics: dict with precision, recall, F1, accuracy
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
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }

@make_nograd_func
def collect_errors_uncertainties(model, dataloader, evidential_method):
    """
    Collect errors and uncertainties from model predictions.
    
    Returns:
        errors: list of error arrays
        aleatoric: list of aleatoric uncertainty arrays
        epistemic: list of epistemic uncertainty arrays
    """
    model.eval()
    all_errors = []
    all_aleatoric = []
    all_epistemic = []
    
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
        # Model output might have different resolution due to image_scale and network architecture
        if evidential.shape[-2:] != depth_gt.shape[-2:]:
            # Resize evidential prediction to match depth_gt spatial dimensions
            evidential = F.interpolate(
                evidential,
                size=depth_gt.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
            # Resize probability_volume to match depth_gt spatial dimensions
            probability_volume = F.interpolate(
                probability_volume,
                size=depth_gt.shape[-2:],
                mode='bilinear',
                align_corners=True
            )
        
        # Ensure mask matches depth_gt spatial dimensions (should already match, but check anyway)
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
        
        # Compute errors
        error_map = (depth_est - depth_gt).abs() * mask
        error_np = error_map.cpu().numpy()
        mask_np = mask.cpu().numpy()
        
        # Extract uncertainties
        if evidential_method == 'sder':
            aleatoric = evidential_outputs["aleatoric_sder"].cpu().numpy()
            epistemic = evidential_outputs["epistemic_sder"].cpu().numpy()
        else:  # der
            aleatoric = evidential_outputs["aleatoric_der"].cpu().numpy()
            epistemic = evidential_outputs["epistemic_der"].cpu().numpy()
        
        # Flatten and filter by mask
        for b in range(error_np.shape[0]):
            mask_b = mask_np[b, 0] > 0.5
            if np.any(mask_b):
                all_errors.append(error_np[b, 0][mask_b])
                all_aleatoric.append(aleatoric[b, 0][mask_b])
                all_epistemic.append(epistemic[b, 0][mask_b])
    
    # Concatenate all arrays
    errors = np.concatenate(all_errors) if all_errors else np.array([])
    aleatoric = np.concatenate(all_aleatoric) if all_aleatoric else np.array([])
    epistemic = np.concatenate(all_epistemic) if all_epistemic else np.array([])
    
    return errors, aleatoric, epistemic

def main():
    # Load model
    print(f'Loading model from {args.loadckpt}')
    model = EMVSNet(
        disparity_level=args.numdepth,
        image_scale=args.image_scale,
        max_h=args.max_h,
        max_w=args.max_w,
        evidential_method=args.evidential_method
    )
    
    # Load checkpoint
    state_dict = torch.load(args.loadckpt, map_location=device)
    model_state_dict = {}
    for k, v in state_dict['model'].items():
        key = k[7:] if k.startswith('module.') else k
        model_state_dict[key] = v
    
    model.load_state_dict(model_state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    print(f'Model loaded. Best validation MAE: {state_dict.get("best_val_mae", "N/A")} at epoch {state_dict.get("best_val_epoch", "N/A")}')
    
    # Create datasets
    # Use trainpath for both val and test datasets to load data with ground truth (training data structure)
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
    
    # Collect errors and uncertainties from validation set
    print('Collecting errors and uncertainties from validation set...')
    val_errors, val_aleatoric, val_epistemic = collect_errors_uncertainties(
        model, val_loader, args.evidential_method
    )
    print(f'Validation set: {len(val_errors)} pixels')
    if len(val_errors) > 0:
        print(f'  Error distribution:')
        print(f'    Min: {np.min(val_errors):.3f} mm, Max: {np.max(val_errors):.3f} mm')
        print(f'    Mean: {np.mean(val_errors):.3f} mm, Median: {np.median(val_errors):.3f} mm')
        print(f'    Std: {np.std(val_errors):.3f} mm')
        print(f'    Percentiles: 25th={np.percentile(val_errors, 25):.3f} mm, 75th={np.percentile(val_errors, 75):.3f} mm, 95th={np.percentile(val_errors, 95):.3f} mm')
        print(f'  Error thresholds:')
        print(f'    Errors > 2mm: {np.sum(val_errors > 2) / len(val_errors) * 100:.1f}% ({np.sum(val_errors > 2)} pixels)')
        print(f'    Errors > 4mm: {np.sum(val_errors > 4) / len(val_errors) * 100:.1f}% ({np.sum(val_errors > 4)} pixels)')
        print(f'    Errors > 8mm: {np.sum(val_errors > 8) / len(val_errors) * 100:.1f}% ({np.sum(val_errors > 8)} pixels)')
        print(f'    Errors > 16mm: {np.sum(val_errors > 16) / len(val_errors) * 100:.1f}% ({np.sum(val_errors > 16)} pixels)')
        print(f'  Aleatoric uncertainty:')
        print(f'    Min: {np.min(val_aleatoric):.6f}, Max: {np.max(val_aleatoric):.6f}')
        print(f'    Mean: {np.mean(val_aleatoric):.6f}, Median: {np.median(val_aleatoric):.6f}')
        print(f'  Epistemic uncertainty:')
        print(f'    Min: {np.min(val_epistemic):.6f}, Max: {np.max(val_epistemic):.6f}')
        print(f'    Mean: {np.mean(val_epistemic):.6f}, Median: {np.median(val_epistemic):.6f}')
    
    # Collect errors and uncertainties from test set
    print('Collecting errors and uncertainties from test set...')
    test_errors, test_aleatoric, test_epistemic = collect_errors_uncertainties(
        model, test_loader, args.evidential_method
    )
    print(f'Test set: {len(test_errors)} pixels')
    if len(test_errors) > 0:
        print(f'  Error distribution:')
        print(f'    Min: {np.min(test_errors):.3f} mm, Max: {np.max(test_errors):.3f} mm')
        print(f'    Mean: {np.mean(test_errors):.3f} mm, Median: {np.median(test_errors):.3f} mm')
        print(f'    Std: {np.std(test_errors):.3f} mm')
        print(f'    Percentiles: 25th={np.percentile(test_errors, 25):.3f} mm, 75th={np.percentile(test_errors, 75):.3f} mm, 95th={np.percentile(test_errors, 95):.3f} mm')
        print(f'  Error thresholds:')
        print(f'    Errors > 2mm: {np.sum(test_errors > 2) / len(test_errors) * 100:.1f}% ({np.sum(test_errors > 2)} pixels)')
        print(f'    Errors > 4mm: {np.sum(test_errors > 4) / len(test_errors) * 100:.1f}% ({np.sum(test_errors > 4)} pixels)')
        print(f'    Errors > 8mm: {np.sum(test_errors > 8) / len(test_errors) * 100:.1f}% ({np.sum(test_errors > 8)} pixels)')
        print(f'    Errors > 16mm: {np.sum(test_errors > 16) / len(test_errors) * 100:.1f}% ({np.sum(test_errors > 16)} pixels)')
        print(f'  Aleatoric uncertainty:')
        print(f'    Min: {np.min(test_aleatoric):.6f}, Max: {np.max(test_aleatoric):.6f}')
        print(f'    Mean: {np.mean(test_aleatoric):.6f}, Median: {np.median(test_aleatoric):.6f}')
        print(f'  Epistemic uncertainty:')
        print(f'    Min: {np.min(test_epistemic):.6f}, Max: {np.max(test_epistemic):.6f}')
        print(f'    Mean: {np.mean(test_epistemic):.6f}, Median: {np.median(test_epistemic):.6f}')
    
    # Find best thresholds and evaluate
    results = {
        'evidential_method': args.evidential_method,
        'error_thresholds': {}
    }
    
    for error_threshold in [4, 8]:
        print(f'\n=== Error Threshold: {error_threshold}mm ===')
        
        # Find best thresholds on validation set
        print(f'Finding best aleatoric threshold for {error_threshold}mm error...')
        best_aleatoric_thresh, best_aleatoric_f1, aleatoric_val_metrics = find_best_uncertainty_threshold(
            val_errors, val_aleatoric, error_threshold, n_thresh=args.n_thresh
        )
        print(f'Best aleatoric threshold: {best_aleatoric_thresh:.6f}, F1: {best_aleatoric_f1:.4f}')
        
        print(f'Finding best epistemic threshold for {error_threshold}mm error...')
        best_epistemic_thresh, best_epistemic_f1, epistemic_val_metrics = find_best_uncertainty_threshold(
            val_errors, val_epistemic, error_threshold, n_thresh=args.n_thresh
        )
        print(f'Best epistemic threshold: {best_epistemic_thresh:.6f}, F1: {best_epistemic_f1:.4f}')
        
        # Evaluate on test set with best thresholds
        print(f'Evaluating on test set with best thresholds...')
        aleatoric_test_metrics = evaluate_with_threshold(
            test_errors, test_aleatoric, best_aleatoric_thresh, error_threshold
        )
        epistemic_test_metrics = evaluate_with_threshold(
            test_errors, test_epistemic, best_epistemic_thresh, error_threshold
        )
        
        results['error_thresholds'][f'{error_threshold}mm'] = {
            'aleatoric': {
                'best_threshold': float(best_aleatoric_thresh),
                'validation_metrics': aleatoric_val_metrics,
                'test_metrics': aleatoric_test_metrics
            },
            'epistemic': {
                'best_threshold': float(best_epistemic_thresh),
                'validation_metrics': epistemic_val_metrics,
                'test_metrics': epistemic_test_metrics
            }
        }
        
        print(f'Aleatoric test metrics: Precision={aleatoric_test_metrics["precision"]:.4f}, '
              f'Recall={aleatoric_test_metrics["recall"]:.4f}, F1={aleatoric_test_metrics["f1"]:.4f}')
        print(f'Epistemic test metrics: Precision={epistemic_test_metrics["precision"]:.4f}, '
              f'Recall={epistemic_test_metrics["recall"]:.4f}, F1={epistemic_test_metrics["f1"]:.4f}')
    
    # Save results
    output_file = os.path.join(args.outdir, 'uncertainty_threshold_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nResults saved to {output_file}')
    
    # Print summary
    print('\n=== Summary ===')
    for error_thresh in [4, 8]:
        key = f'{error_thresh}mm'
        print(f'\nError Threshold: {error_thresh}mm')
        print(f'  Aleatoric - Threshold: {results["error_thresholds"][key]["aleatoric"]["best_threshold"]:.6f}, '
              f'Test F1: {results["error_thresholds"][key]["aleatoric"]["test_metrics"]["f1"]:.4f}')
        print(f'  Epistemic - Threshold: {results["error_thresholds"][key]["epistemic"]["best_threshold"]:.6f}, '
              f'Test F1: {results["error_thresholds"][key]["epistemic"]["test_metrics"]["f1"]:.4f}')

if __name__ == '__main__':
    main()
