import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import *
from evidential.models import disparity_regression
import ast
from collections import OrderedDict

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Evaluate checkpoint depth prediction on test set')
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
parser.add_argument('--testpath', help='test data path')
parser.add_argument('--testlist', help='test list file')

parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')

parser.add_argument('--loadckpt', default=None, help='load checkpoint (required)')
parser.add_argument('--outdir', default='./test_eval_results', help='output directory for results')
parser.add_argument('--evidential_method', type=str, default='der', choices=['der', 'sder'],
                    help='Evidential method: der (full NIG loss) or sder (simplified, alpha = nu + 1)')

args = parser.parse_args()

if args.loadckpt is None:
    raise ValueError("--loadckpt argument is required. Please provide path to checkpoint file.")

if args.trainpath is None:
    raise ValueError("--trainpath argument is required. Please provide path to training data directory (needed for ground truth).")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.makedirs(args.outdir, exist_ok=True)

print_args(args)

@make_nograd_func
def evaluate_test_set(model, dataloader, evidential_method):
    """
    Evaluate model on test set and collect metrics.
    
    Returns:
        results: dict with aggregated metrics
    """
    model.eval()
    
    # Statistics collection
    avg_statistics = DictAverageMeter()
    inference_times = []
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(dataloader, desc="Evaluating", unit="sample")):
            
            sample_cuda = tocuda(sample, non_blocking=True)
            
            # Check if ground truth is available
            if "depth" not in sample or "mask" not in sample:
                print(f"Warning: Sample {batch_idx} does not have ground truth, skipping metrics computation")
                continue
            
            depth_gt_tensor = tocuda(sample["depth"], non_blocking=True)
            mask_tensor = tocuda(sample["mask"], non_blocking=True)
            
            # Measure inference time
            time_start = time.time()
            outputs_tensor = model(
                sample_cuda["imgs"],
                sample_cuda["proj_matrices"],
                sample_cuda["depth_values"]
            )
            inference_time = time.time() - time_start
            inference_times.append(inference_time)
            
            # Extract depth estimate from model outputs
            # Model returns tuple: (probability_volume, evidential, probabilities) when return_depth=False
            # Compute depth using disparity_regression (same as training)
            probability_volume, evidential, probabilities = outputs_tensor
            depth_est_tensor = disparity_regression(probability_volume, sample_cuda["depth_values"])
            
            # Ensure depth_est and depth_gt/mask have matching spatial dimensions
            # Model output might have different resolution due to image_scale and network architecture
            if depth_est_tensor.shape[-2:] != depth_gt_tensor.shape[-2:]:
                # Resize depth_est to match depth_gt spatial dimensions
                depth_est_tensor = F.interpolate(
                    depth_est_tensor.unsqueeze(1), 
                    size=depth_gt_tensor.shape[-2:], 
                    mode='bilinear', 
                    align_corners=True
                ).squeeze(1)
            
            # Ensure mask matches depth_gt spatial dimensions (should already match, but check anyway)
            if mask_tensor.shape[-2:] != depth_gt_tensor.shape[-2:]:
                mask_tensor = F.interpolate(
                    mask_tensor.unsqueeze(1).float(), 
                    size=depth_gt_tensor.shape[-2:], 
                    mode='nearest', 
                    align_corners=False
                ).squeeze(1).bool()
            
            # Compute metrics
            scalar_outputs = {}
            scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(
                depth_est_tensor, depth_gt_tensor, mask_tensor > 0.5
            )
            scalar_outputs["thres2mm_error"] = Thres_metrics(
                depth_est_tensor, depth_gt_tensor, mask_tensor > 0.5, 2
            )
            scalar_outputs["thres4mm_error"] = Thres_metrics(
                depth_est_tensor, depth_gt_tensor, mask_tensor > 0.5, 4
            )
            scalar_outputs["thres8mm_error"] = Thres_metrics(
                depth_est_tensor, depth_gt_tensor, mask_tensor > 0.5, 8
            )
            scalar_outputs["thres16mm_error"] = Thres_metrics(
                depth_est_tensor, depth_gt_tensor, mask_tensor > 0.5, 16
            )
            scalar_outputs["thres32mm_error"] = Thres_metrics(
                depth_est_tensor, depth_gt_tensor, mask_tensor > 0.5, 32
            )
            
            # Convert to float and update statistics
            scalar_outputs = tensor2float(scalar_outputs)
            avg_statistics.update(scalar_outputs)
            total_samples += 1

            '''    
            except Exception as e:
                print(f"Error processing sample {batch_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
            '''

    # Compute mean inference time
    mean_time_per_sample = sum(inference_times) / len(inference_times) if inference_times else 0.0
    
    # Get aggregated metrics
    mean_stats = avg_statistics.mean() if total_samples > 0 else {}
    
    results = {
        'num_views': args.view_num,
        'num_depth_hypotheses': args.numdepth,
        'mean_time_per_sample': mean_time_per_sample,
        'mae': mean_stats.get('abs_depth_error', 0.0),
        'error_thresholds': {
            '2mm': mean_stats.get('thres2mm_error', 0.0) * 100,  # Convert to percentage
            '4mm': mean_stats.get('thres4mm_error', 0.0) * 100,
            '8mm': mean_stats.get('thres8mm_error', 0.0) * 100,
            '16mm': mean_stats.get('thres16mm_error', 0.0) * 100,
            '32mm': mean_stats.get('thres32mm_error', 0.0) * 100,
        },
        'total_samples': total_samples,
        'evidential_method': evidential_method
    }
    
    return results

def print_results(results, checkpoint_path):
    """Print results in formatted table."""
    print('\n' + '='*80)
    print('=== Test Set Depth Evaluation ===')
    print('='*80)
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Dataset: {args.dataset}')
    print(f'Number of views: {results["num_views"]}')
    print(f'Number of depth hypotheses: {results["num_depth_hypotheses"]}')
    print(f'Total samples: {results["total_samples"]}')
    print(f'Evidential method: {results["evidential_method"]}')
    print('\nResults:')
    print(f'  Mean Absolute Error (MAE): {results["mae"]:.3f} mm')
    print(f'  Mean time per sample: {results["mean_time_per_sample"]:.3f} s')
    print('\n  Error Thresholds (% pixels exceeding):')
    print(f'    2mm:  {results["error_thresholds"]["2mm"]:.2f}%')
    print(f'    4mm:  {results["error_thresholds"]["4mm"]:.2f}%')
    print(f'    8mm:  {results["error_thresholds"]["8mm"]:.2f}%')
    print(f'    16mm: {results["error_thresholds"]["16mm"]:.2f}%')
    print(f'    32mm: {results["error_thresholds"]["32mm"]:.2f}%')
    print('='*80 + '\n')

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
    
    # Create test dataset and dataloader
    # Use trainpath to load data with ground truth (training data structure)
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(
        args.trainpath, args.testlist, "test", args.view_num, args.numdepth,
        args.interval_scale, args.inverse_depth, args.origin_size,
        args.light_idx, args.image_scale
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )
    
    print(f'Test dataset created: {len(test_dataset)} samples')
    
    # Evaluate test set
    results = evaluate_test_set(model, test_loader, args.evidential_method)
    
    # Print results
    print_results(results, args.loadckpt)
    
    # Save results to JSON
    checkpoint_name = os.path.basename(args.loadckpt).replace('.ckpt', '')
    output_file = os.path.join(args.outdir, f'test_evaluation_{checkpoint_name}.json')
    
    # Add checkpoint path to results for reference
    results['checkpoint_path'] = args.loadckpt
    results['dataset_path'] = args.trainpath
    results['test_list'] = args.testlist
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'Results saved to {output_file}')

if __name__ == '__main__':
    main()
