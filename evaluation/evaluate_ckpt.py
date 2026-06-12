"""
Evaluate EMVSNet checkpoint on test set: depth prediction metrics (MAE, threshold errors),
timing, and peak VRAM. Uses the same depth logic as training: disparity_regression(probability_volume, depth_values).
"""
import argparse
import os
import sys
import json
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from collections import OrderedDict
import ast

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import *
from evidential.models import disparity_regression

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Evaluate EMVSNet checkpoint on test set with depth statistics')
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
parser.add_argument('--testpath', help='test data path (defaults to trainpath if not provided)')
parser.add_argument('--testlist', help='test list file')

parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')

parser.add_argument('--loadckpt', required=True, help='load checkpoint (required)')
parser.add_argument('--outdir', default='./evaluation/results', help='output directory for results')
parser.add_argument('--evidential_method', type=str, default='der', choices=['der', 'sder'],
                    help='Evidential method: der (full NIG loss) or sder (simplified, alpha = nu + 1)')

args = parser.parse_args()

if args.trainpath is None:
    raise ValueError("--trainpath argument is required. Please provide path to training data directory (needed for ground truth).")

if args.testlist is None:
    raise ValueError("--testlist argument is required. Please provide path to test list file.")

if args.testpath is None:
    args.testpath = args.trainpath

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.makedirs(args.outdir, exist_ok=True)

print_args(args)


@make_nograd_func
def evaluate_test_set(model, dataloader, evidential_method):
    """
    Evaluate model on test set and collect metrics.
    Depth prediction uses disparity_regression(probability_volume, depth_values), same as training (loss_der).
    """
    model.eval()

    avg_statistics = DictAverageMeter()
    inference_times = []
    total_samples = 0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(dataloader, desc="Evaluating", unit="sample")):
            sample_cuda = tocuda(sample, non_blocking=True)

            if "depth" not in sample or "mask" not in sample:
                print(f"Warning: Sample {batch_idx} does not have ground truth, skipping metrics computation")
                continue

            depth_gt_tensor = tocuda(sample["depth"], non_blocking=True)
            mask_tensor = tocuda(sample["mask"], non_blocking=True)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            time_start = time.perf_counter()

            probability_volume, evidential, probabilities = model(
                sample_cuda["imgs"],
                sample_cuda["proj_matrices"],
                sample_cuda["depth_values"]
            )
            # Same depth logic as training: loss_der uses disparity_regression(probability_volume, depth_value)
            depth_est_tensor = disparity_regression(probability_volume, sample_cuda["depth_values"])

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_time = time.perf_counter() - time_start
            inference_times.append(inference_time)

            if depth_est_tensor.shape[-2:] != depth_gt_tensor.shape[-2:]:
                depth_est_tensor = F.interpolate(
                    depth_est_tensor.unsqueeze(1),
                    size=depth_gt_tensor.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                ).squeeze(1)

            if mask_tensor.shape[-2:] != depth_gt_tensor.shape[-2:]:
                mask_tensor = F.interpolate(
                    mask_tensor.unsqueeze(1).float(),
                    size=depth_gt_tensor.shape[-2:],
                    mode='nearest',
                    align_corners=False
                ).squeeze(1).bool()

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

            scalar_outputs = tensor2float(scalar_outputs)
            avg_statistics.update(scalar_outputs)
            total_samples += 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()
        peak_vram_mb = peak_bytes / (1024 ** 2)
    else:
        peak_vram_mb = None

    mean_time_per_sample = sum(inference_times) / len(inference_times) if inference_times else 0.0
    mean_stats = avg_statistics.mean() if total_samples > 0 else {}

    results = {
        'num_views': args.view_num,
        'num_depth_hypotheses': args.numdepth,
        'mean_time_per_sample': mean_time_per_sample,
        'peak_vram_mb': peak_vram_mb,
        'mae': mean_stats.get('abs_depth_error', 0.0),
        'error_thresholds': {
            '2mm': mean_stats.get('thres2mm_error', 0.0) * 100,
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
    print('\n' + '=' * 80)
    print('=== Test Set Evaluation Statistics ===')
    print('=' * 80)
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Dataset: {args.dataset}')
    print(f'Number of views: {results["num_views"]}')
    print(f'Number of depth hypotheses: {results["num_depth_hypotheses"]}')
    print(f'Total samples: {results["total_samples"]}')
    print(f'Evidential method: {results["evidential_method"]}')
    print('\nStatistics:')
    print(f'  Mean Absolute Error (MAE): {results["mae"]:.3f} mm')
    print(f'  Average time per sample: {results["mean_time_per_sample"]:.3f} s')
    if results["peak_vram_mb"] is not None:
        print(f'  Peak VRAM usage: {results["peak_vram_mb"]:.2f} MB')
    else:
        print(f'  Peak VRAM usage: N/A (CPU mode)')
    print('\n  Error Thresholds (% pixels exceeding):')
    print(f'    2mm:  {results["error_thresholds"]["2mm"]:.2f}%')
    print(f'    4mm:  {results["error_thresholds"]["4mm"]:.2f}%')
    print(f'    8mm:  {results["error_thresholds"]["8mm"]:.2f}%')
    print(f'    16mm: {results["error_thresholds"]["16mm"]:.2f}%')
    print(f'    32mm: {results["error_thresholds"]["32mm"]:.2f}%')
    print('=' * 80 + '\n')


def main():
    print(f'Loading model from {args.loadckpt}')
    model = EMVSNet(
        disparity_level=args.numdepth,
        image_scale=args.image_scale,
        max_h=args.max_h,
        max_w=args.max_w,
        return_depth=False,
        evidential_method=args.evidential_method
    )

    state_dict = torch.load(args.loadckpt, map_location=device)
    if 'model' in state_dict:
        model_state_dict = state_dict['model']
    else:
        model_state_dict = state_dict

    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model = model.to(device)
    load_result = model.load_state_dict(new_state_dict, strict=False)
    missing_keys = load_result.missing_keys
    unexpected_keys = load_result.unexpected_keys
    if missing_keys:
        print('Warning: Checkpoint is missing the following model weights:')
        for key in missing_keys:
            print(f'  - {key}')
    if unexpected_keys:
        print('Warning: Checkpoint contains unexpected weights not used by the model:')
        for key in unexpected_keys:
            print(f'  - {key}')
    if not missing_keys and not unexpected_keys:
        print('Model loaded successfully (all checkpoint keys matched).')
    else:
        print('Model loaded with warnings (see above). Evaluation may be incorrect if required weights are missing.')
    model.eval()
    if 'best_val_mae' in state_dict:
        print(f'Checkpoint info: Best validation MAE: {state_dict["best_val_mae"]:.3f} at epoch {state_dict.get("best_val_epoch", "N/A")}')

    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(
        args.testpath, args.testlist, "test", args.view_num, args.numdepth,
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

    results = evaluate_test_set(model, test_loader, args.evidential_method)
    print_results(results, args.loadckpt)

    checkpoint_name = os.path.basename(args.loadckpt).replace('.ckpt', '')
    output_file = os.path.join(args.outdir, f'test_evaluation_{checkpoint_name}.json')
    results['checkpoint_path'] = args.loadckpt
    results['dataset_path'] = args.trainpath
    results['test_list'] = args.testlist

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'Results saved to {output_file}')


if __name__ == '__main__':
    main()
