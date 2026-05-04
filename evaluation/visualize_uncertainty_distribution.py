import argparse
import os
import random
import sys
import json
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from collections import OrderedDict
import ast

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import *
from evidential.models import loss_der, uncertainty_der, uncertainty_sder, disparity_regression

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Visualize uncertainty distribution (aleatoric or epistemic): DTU training vs TNT')
parser.add_argument('--inverse_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)
parser.add_argument('--origin_size', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)

parser.add_argument('--max_h', type=int, default=512, help='Maximum image height')
parser.add_argument('--max_w', type=int, default=640, help='Maximum image width')
parser.add_argument('--light_idx', type=int, default=3, help='select while in test')
parser.add_argument('--view_num', type=int, default=5, help='number of views')
parser.add_argument('--image_scale', type=float, default=0.25, help='pred depth map scale')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset for DTU')
parser.add_argument('--trainpath', help='train/val datapath (for DTU ground truth data)')
parser.add_argument('--tntpath', help='TNT datapath (for Tanks and Temples data)')
parser.add_argument('--dtu_train_list', default='lists/dtu/train.txt', help='DTU training list file')
parser.add_argument('--dtu_test_list', default='lists/dtu/test.txt', help='DTU test list file')
parser.add_argument('--tnt_list', default='lists/tnt/lighthouse_train_m60.txt', help='TNT list file')

parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')

parser.add_argument('--loadckpt', required=True, help='load checkpoint (required)')
parser.add_argument('--outdir', default='./evaluation/results', help='output directory for results')
parser.add_argument('--evidential_method', type=str, default='der', choices=['der', 'sder'],
                    help='Evidential method: der (full NIG loss) or sder (simplified, alpha = nu + 1)')
parser.add_argument('--uncertainty_type', type=str, default='epistemic',
                    choices=['aleatoric', 'epistemic', 'both'],
                    help='Uncertainty to evaluate: aleatoric, epistemic, or both (dual y-axis plot)')
parser.add_argument('--n_dtu', type=int, default=50, help='Number of DTU samples to use')
parser.add_argument('--n_tnt', type=int, default=50, help='Number of TNT samples to use')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sample selection (for reproducibility)')
parser.add_argument('--fig_height', type=float, default=6.0, help='Figure height in inches')

args = parser.parse_args()

if args.trainpath is None:
    raise ValueError("--trainpath argument is required. Please provide path to DTU training data directory.")
if args.tntpath is None:
    raise ValueError("--tntpath argument is required. Please provide path to TNT data directory.")

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
def collect_mean_uncertainty_dtu(model, dataloader, evidential_method, uncertainty_type, dataset_label='DTU'):
    """
    Collect mean uncertainty (per pixel) per sample from DTU dataset (with ground truth).
    For DTU: uses the dataset mask (valid image region) intersected with valid predicted depth.
    For TNT: only pixels with valid predicted depth (finite and > 0) are used (no mask).
    
    Args:
        model: The EMVSNet model
        dataloader: DataLoader for DTU dataset
        evidential_method: 'der' or 'sder'
        uncertainty_type: 'aleatoric', 'epistemic', or 'both'
        dataset_label: Label for the dataset (e.g., 'DTU Training' or 'DTU Test')
    
    Returns:
        results: list of dicts with keys: 'sample_id', 'mean_uncertainty', 'dataset';
                 if uncertainty_type=='both' also 'mean_aleatoric', 'mean_epistemic'
    """
    model.eval()
    results = []
    collect_both = (uncertainty_type == 'both')
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(dataloader, desc="Processing DTU", unit="batch")):
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
            
            # Extract uncertainty (aleatoric, epistemic, or both)
            key_aleatoric = "aleatoric_sder" if evidential_method == 'sder' else "aleatoric_der"
            key_epistemic = "epistemic_sder" if evidential_method == 'sder' else "epistemic_der"
            aleatoric_np = evidential_outputs[key_aleatoric].cpu().numpy()
            epistemic_np = evidential_outputs[key_epistemic].cpu().numpy()
            if not collect_both:
                uncertainty = epistemic_np if uncertainty_type == 'epistemic' else aleatoric_np
            
            depth_est_np = depth_est.cpu().numpy()
            mask_np = mask.cpu().numpy()
            
            # DTU: use dataset mask (valid image region) and valid predicted depth
            for b in range(aleatoric_np.shape[0]):
                depth_est_b = depth_est_np[b, 0] if depth_est_np.ndim == 4 else depth_est_np[b]
                mask_b = mask_np[b, 0] if mask_np.ndim == 4 else mask_np[b]
                mask_b = mask_b.astype(bool) & np.isfinite(depth_est_b) & (depth_est_b > 0)
                
                if np.any(mask_b):
                    if collect_both:
                        al_b = aleatoric_np[b, 0] if aleatoric_np.ndim == 4 else aleatoric_np[b]
                        ep_b = epistemic_np[b, 0] if epistemic_np.ndim == 4 else epistemic_np[b]
                        mean_aleatoric = float(np.mean(al_b[mask_b]))
                        mean_epistemic = float(np.mean(ep_b[mask_b]))
                        mean_uncertainty = mean_epistemic  # for compatibility
                    else:
                        unc_b = uncertainty[b, 0] if uncertainty.ndim == 4 else uncertainty[b]
                        mean_uncertainty = float(np.mean(unc_b[mask_b]))
                        mean_aleatoric = mean_epistemic = None
                    
                    # Path for display (depth file path like "Depths/scan2_train/depth_map_0000.pfm")
                    if 'name' in sample:
                        path_val = sample['name'][b] if isinstance(sample['name'], list) else sample['name']
                        path = str(path_val) if path_val is not None else f"dtu_batch{batch_idx}_sample{b}"
                    else:
                        path = f"dtu_batch{batch_idx}_sample{b}"
                    
                    r = {
                        'sample_id': path,
                        'path': path,
                        'mean_uncertainty': mean_uncertainty,
                        'dataset': dataset_label
                    }
                    if collect_both:
                        r['mean_aleatoric'] = mean_aleatoric
                        r['mean_epistemic'] = mean_epistemic
                    results.append(r)
    
    return results


@make_nograd_func
def collect_mean_uncertainty_tnt(model, dataloader, evidential_method, uncertainty_type):
    """
    Collect mean uncertainty (per pixel) per sample from TNT dataset (no ground truth).
    TNT has no object mask; only pixels with valid predicted depth are used (resolution-invariant).
    
    Args:
        uncertainty_type: 'aleatoric', 'epistemic', or 'both'
    
    Returns:
        results: list of dicts with keys: 'sample_id', 'mean_uncertainty', 'dataset';
                 if uncertainty_type=='both' also 'mean_aleatoric', 'mean_epistemic'
    """
    model.eval()
    results = []
    collect_both = (uncertainty_type == 'both')
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(dataloader, desc="Processing TNT", unit="batch")):
            sample_cuda = tocuda(sample, non_blocking=True)
            
            probability_volume, evidential, probabilities = model(
                sample_cuda["imgs"], 
                sample_cuda["proj_matrices"], 
                sample_cuda["depth_values"]
            )
            
            # Compute depth estimate from probability volume
            depth_est = disparity_regression(probability_volume, sample_cuda["depth_values"])
            
            # Extract evidential parameters
            # evidential has shape [B, 4, H, W] where 4 = (gamma, nu, alpha, beta)
            gamma, nu, alpha, beta = torch.unbind(evidential, dim=1)  # Each: [B, H, W]
            
            # Compute uncertainties (aleatoric, epistemic)
            if evidential_method == 'sder':
                aleatoric, epistemic = uncertainty_sder(gamma, nu, alpha, beta)
            else:  # der
                aleatoric, epistemic = uncertainty_der(gamma, nu, alpha, beta)
            
            aleatoric_np = aleatoric.cpu().numpy()
            epistemic_np = epistemic.cpu().numpy()
            depth_est_np = depth_est.cpu().numpy()
            
            # TNT has no GT mask; use only pixels with valid predicted depth as valid region
            for b in range(aleatoric_np.shape[0]):
                depth_est_b = depth_est_np[b, 0] if depth_est_np.ndim == 4 else depth_est_np[b]
                mask_b = np.isfinite(depth_est_b) & (depth_est_b > 0)
                
                if np.any(mask_b):
                    al_b = aleatoric_np[b, 0] if aleatoric_np.ndim == 4 else aleatoric_np[b]
                    ep_b = epistemic_np[b, 0] if epistemic_np.ndim == 4 else epistemic_np[b]
                    mean_aleatoric = float(np.mean(al_b[mask_b]))
                    mean_epistemic = float(np.mean(ep_b[mask_b]))
                    mean_uncertainty = mean_epistemic if uncertainty_type == 'epistemic' else mean_aleatoric
                    
                    # Path for display (filename template like "Lighthouse/{}/00000000{}")
                    if 'filename' in sample:
                        filename_template = sample['filename'][b] if isinstance(sample['filename'], list) else sample['filename']
                        path = str(filename_template) if filename_template is not None else f"tnt_batch{batch_idx}_sample{b}"
                    else:
                        path = f"tnt_batch{batch_idx}_sample{b}"
                    
                    r = {
                        'sample_id': path,
                        'path': path,
                        'mean_uncertainty': mean_uncertainty,
                        'dataset': 'TNT'
                    }
                    if collect_both:
                        r['mean_aleatoric'] = mean_aleatoric
                        r['mean_epistemic'] = mean_epistemic
                    results.append(r)
    
    return results


def plot_uncertainty_distribution(results, output_path, uncertainty_type='epistemic', title_suffix=''):
    """
    Plot sorted bar chart with dots showing mean uncertainty (per pixel) per sample.
    Each value is the mean over valid pixels for that image; resolution-invariant.
    
    Args:
        results: list of dicts with 'sample_id', 'mean_uncertainty', 'dataset'
        output_path: path to save the figure
        uncertainty_type: 'aleatoric' or 'epistemic' (for axis label)
        title_suffix: suffix to add to title (e.g., 'DTU Training vs TNT' or 'DTU Test vs TNT')
    """
    # Sort by uncertainty
    sorted_results = sorted(results, key=lambda x: x['mean_uncertainty'])
    
    # Extract sorted values and labels
    sorted_vals = [r['mean_uncertainty'] for r in sorted_results]
    sorted_labels = np.array([r['dataset'] for r in sorted_results])
    
    # Map dataset labels to simplified names and colors
    # For DTU Training vs TNT or DTU Test vs TNT: use DTU/TnT
    # For DTU Test vs DTU Training: use DTU Test/DTU Training
    unique_labels = list(set(sorted_labels))
    
    # Determine color mapping based on datasets present
    if 'TNT' in unique_labels:
        # Map DTU variants to "DTU" for consistency
        label_map = {
            'DTU Training': 'DTU',
            'DTU Test': 'DTU',
            'TNT': 'TnT'
        }
        color_map = {
            'DTU Training': 'tab:blue',
            'DTU Test': 'tab:blue',
            'TNT': 'tab:orange'
        }
    else:
        # DTU Test vs DTU Training case
        label_map = {
            'DTU Training': 'DTU Training',
            'DTU Test': 'DTU Test'
        }
        color_map = {
            'DTU Training': 'tab:blue',
            'DTU Test': 'tab:green'
        }
    
    # Map labels
    mapped_labels = np.array([label_map.get(label, label) for label in sorted_labels])
    
    # Create figure
    plt.figure(figsize=(10, args.fig_height))
    
    # Get colors for each sample
    colors = [color_map.get(label, 'gray') for label in sorted_labels]
    
    # Plot vertical lines + dots on top
    for i, v in enumerate(sorted_vals):
        plt.vlines(i, 0, v, color=colors[i], linewidth=1)
        plt.scatter(i, v, color=colors[i], s=15)
    
    # Create legend - use unique mapped labels; put TnT first so orange count is higher
    unique_mapped = sorted(set(mapped_labels), key=lambda x: (0 if x == 'TnT' else 1))
    for label in unique_mapped:
        # Find a sample with this label to get the color
        sample_idx = np.where(mapped_labels == label)[0][0]
        color = colors[sample_idx]
        plt.plot([], [], color=color, label=label)
    
    # Red line between 50th and 51st sample (at 50.5 in 1-based terms)
    plt.axvline(49.5, color="red", linestyle="--", linewidth=1.5)
    
    # Count samples in lower half (first 50 samples)
    lower50_labels = mapped_labels[:50]
    label_counts = {}
    for label in unique_mapped:
        label_counts[label] = np.sum(lower50_labels == label)
    
    # Add dots + numbers left of red line
    x_pos = 45
    y_max = max(sorted_vals)
    # TnT (orange) closer to top; larger gap between the two indicators
    y_base = y_max * 0.96
    y_offset = 0.12 * y_max
    y_positions = [y_base - i * y_offset for i in range(len(unique_mapped))]
    
    for idx, label in enumerate(unique_mapped):
        count = label_counts.get(label, 0)
        y_pos = y_positions[idx]
        # Find color for this label
        sample_idx = np.where(mapped_labels == label)[0][0]
        color = colors[sample_idx]
        
        # Number text
        plt.text(x_pos, y_pos, f"{count}", va="center", fontsize=10)
        # Dot on the right
        plt.scatter(x_pos + 3, y_pos, color=color, s=30)
    
    n = len(sorted_results)
    tick_positions = [0, 24, 49, 74, 99]
    tick_labels = [1, 25, 50, 75, 100]
    valid_ticks = [(p, l) for p, l in zip(tick_positions, tick_labels) if p < n]
    if valid_ticks:
        plt.xticks([p for p, _ in valid_ticks], [l for _, l in valid_ticks])
    ylabel = "Mean aleatoric uncertainty" if uncertainty_type == 'aleatoric' else "Mean epistemic uncertainty"
    plt.xlabel("Sample index")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Figure saved to {output_path}')
    plt.close()


# Color scheme for dual-axis: light/dark for aleatoric/epistemic, blue/orange for DTU/TnT
# Aleatoric (light): DTU = light blue, TnT = light orange
# Epistemic (dark):  DTU = dark blue,  TnT = dark orange
COLORS = {
    'aleatoric_dtu': '#93c5fd',   # light blue
    'aleatoric_tnt': '#fdba74',   # light orange
    'epistemic_dtu': '#1d4ed8',   # dark blue
    'epistemic_tnt': '#ea580c',   # dark orange
}


def plot_uncertainty_distribution_dual_axis(results, output_path, title_suffix=''):
    """
    Plot both aleatoric and epistemic uncertainty on one graph: shared x-axis (sample rank),
    left y-axis for aleatoric, right y-axis for epistemic. Bars with dots per sample.
    Left bars sorted by aleatoric (low to high); right bars sorted by epistemic (low to high).
    Light colors = aleatoric, dark colors = epistemic. Blue = DTU, orange = TnT.
    
    Args:
        results: list of dicts with 'mean_aleatoric', 'mean_epistemic', 'dataset'
        output_path: path to save the figure
        title_suffix: suffix for the plot (e.g. 'DTU Training vs TNT')
    """
    # Sort independently: aleatoric by aleatoric, epistemic by epistemic
    sorted_by_aleatoric = sorted(results, key=lambda x: x['mean_aleatoric'])
    sorted_by_epistemic = sorted(results, key=lambda x: x['mean_epistemic'])
    n = len(results)
    x = np.arange(n)
    aleatoric_vals = np.array([r['mean_aleatoric'] for r in sorted_by_aleatoric])
    epistemic_vals = np.array([r['mean_epistemic'] for r in sorted_by_epistemic])
    aleatoric_labels = np.array([r['dataset'] for r in sorted_by_aleatoric])
    epistemic_labels = np.array([r['dataset'] for r in sorted_by_epistemic])

    # Map dataset to dtu/tnt for color choice (DTU Training, DTU Test -> dtu; TNT -> tnt)
    def is_dtu(lab):
        return lab in ('DTU Training', 'DTU Test')
    aleatoric_colors = [COLORS['aleatoric_dtu'] if is_dtu(lab) else COLORS['aleatoric_tnt'] for lab in aleatoric_labels]
    epistemic_colors = [COLORS['epistemic_dtu'] if is_dtu(lab) else COLORS['epistemic_tnt'] for lab in epistemic_labels]

    fig, ax1 = plt.subplots(figsize=(10, args.fig_height))
    ax2 = ax1.twinx()

    # Left y-axis: aleatoric bars + dots (offset left of center)
    ax1.set_xlabel("Sample rank (by aleatoric / by epistemic)")
    ax1.set_ylabel("Mean aleatoric uncertainty", color='#64748b')
    ax1.tick_params(axis='y', labelcolor='#64748b')
    ax1.set_xlim(-0.5, n - 0.5)
    x_aleat = x - 0.2
    for i in range(n):
        ax1.vlines(x_aleat[i], 0, aleatoric_vals[i], color=aleatoric_colors[i], linewidth=1)
        ax1.scatter(x_aleat[i], aleatoric_vals[i], color=aleatoric_colors[i], s=15, zorder=2)

    # Right y-axis: epistemic bars + dots (offset right of center)
    ax2.set_ylabel("Mean epistemic uncertainty", color='#64748b')
    ax2.tick_params(axis='y', labelcolor='#64748b')
    x_epist = x + 0.2
    for i in range(n):
        ax2.vlines(x_epist[i], 0, epistemic_vals[i], color=epistemic_colors[i], linewidth=1)
        ax2.scatter(x_epist[i], epistemic_vals[i], color=epistemic_colors[i], s=15, zorder=2)

    # Red separator between 50th and 51st sample (consistent with single-type plot)
    ax1.axvline(49.5, color='#b91c1c', linestyle='--', linewidth=1.5, zorder=1)

    # Legend: Aleatoric (light) DTU/TnT, Epistemic (dark) DTU/TnT
    legend_handles = [
        Line2D([0], [0], color=COLORS['aleatoric_dtu'], linewidth=3, label='Aleatoric (DTU)'),
        Line2D([0], [0], color=COLORS['aleatoric_tnt'], linewidth=3, label='Aleatoric (TnT)'),
        Line2D([0], [0], color=COLORS['epistemic_dtu'], linewidth=3, label='Epistemic (DTU)'),
        Line2D([0], [0], color=COLORS['epistemic_tnt'], linewidth=3, label='Epistemic (TnT)'),
    ]
    ax1.legend(handles=legend_handles, loc='upper left')

    tick_positions = [0, 24, 49, 74, 99]
    tick_labels = [1, 25, 50, 75, 100]
    valid_ticks = [(p, l) for p, l in zip(tick_positions, tick_labels) if p < n]
    if valid_ticks:
        ax1.set_xticks([p for p, _ in valid_ticks])
        ax1.set_xticklabels([str(l) for _, l in valid_ticks])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Figure saved to {output_path}')
    plt.close()


def main():
    # Set random seed for reproducible sample selection
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    
    # Process DTU training dataset
    print('\n=== Processing DTU Training Set ===')
    MVSDataset_DTU = find_dataset_def(args.dataset)
    dtu_train_dataset = MVSDataset_DTU(
        args.trainpath, args.dtu_train_list, "train", args.view_num, args.numdepth,
        args.interval_scale, args.inverse_depth, args.origin_size,
        args.light_idx, args.image_scale
    )
    
    # Limit to n_dtu samples (random selection)
    dtu_train_total = len(dtu_train_dataset)
    if len(dtu_train_dataset) > args.n_dtu:
        indices = random.sample(range(len(dtu_train_dataset)), args.n_dtu)
        dtu_train_dataset = Subset(dtu_train_dataset, indices)
    
    dtu_train_loader = DataLoader(
        dtu_train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, drop_last=False, pin_memory=True
    )
    
    print(f'DTU training dataset: {len(dtu_train_dataset)} samples')
    dtu_train_results = collect_mean_uncertainty_dtu(model, dtu_train_loader, args.evidential_method, args.uncertainty_type, 'DTU Training')
    print(f'Collected {len(dtu_train_results)} DTU training samples')
    
    # Process DTU test dataset
    print('\n=== Processing DTU Test Set ===')
    dtu_test_dataset = MVSDataset_DTU(
        args.trainpath, args.dtu_test_list, "test", args.view_num, args.numdepth,
        args.interval_scale, args.inverse_depth, args.origin_size,
        args.light_idx, args.image_scale
    )
    
    # Limit to n_dtu samples (random selection)
    dtu_test_total = len(dtu_test_dataset)
    if len(dtu_test_dataset) > args.n_dtu:
        indices = random.sample(range(len(dtu_test_dataset)), args.n_dtu)
        dtu_test_dataset = Subset(dtu_test_dataset, indices)
    
    dtu_test_loader = DataLoader(
        dtu_test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, drop_last=False, pin_memory=True
    )
    
    print(f'DTU test dataset: {len(dtu_test_dataset)} samples')
    dtu_test_results = collect_mean_uncertainty_dtu(model, dtu_test_loader, args.evidential_method, args.uncertainty_type, 'DTU Test')
    print(f'Collected {len(dtu_test_results)} DTU test samples')
    
    # Process TNT dataset
    print('\n=== Processing TNT Dataset ===')
    # TNT uses different dataset class and settings
    MVSDataset_TNT = find_dataset_def('data_eval_transform_padding')
    
    # TNT-specific settings (from eval_tnt.sh)
    tnt_max_h = 544
    tnt_max_w = 1024
    tnt_inverse_depth = True
    tnt_interval_scale = 1.0
    
    tnt_dataset = MVSDataset_TNT(
        args.tntpath, args.tnt_list, "test", args.view_num, args.numdepth,
        tnt_interval_scale, tnt_inverse_depth,
        adaptive_scaling=True, max_h=tnt_max_h, max_w=tnt_max_w, 
        sample_scale=1, base_image_size=8
    )
    
    # Limit to n_tnt samples (random selection)
    tnt_total = len(tnt_dataset)
    if len(tnt_dataset) > args.n_tnt:
        indices = random.sample(range(len(tnt_dataset)), args.n_tnt)
        tnt_dataset = Subset(tnt_dataset, indices)
    
    tnt_loader = DataLoader(
        tnt_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, drop_last=False, pin_memory=True
    )
    
    print(f'TNT dataset: {len(tnt_dataset)} samples')
    tnt_results = collect_mean_uncertainty_tnt(model, tnt_loader, args.evidential_method, args.uncertainty_type)
    print(f'Collected {len(tnt_results)} TNT samples')
    
    # Create plots: Training vs TNT, Test vs TNT, and Test vs Training
    checkpoint_name = os.path.basename(args.loadckpt).replace('.ckpt', '')
    ut = args.uncertainty_type  # for filenames
    
    if ut == 'both':
        # Dual y-axis: aleatoric (left) and epistemic (right), same x-axis
        train_vs_tnt_results = dtu_train_results + tnt_results
        output_fig_train = os.path.join(args.outdir, f'uncertainty_distribution_both_train_vs_tnt_{checkpoint_name}.png')
        plot_uncertainty_distribution_dual_axis(train_vs_tnt_results, output_fig_train, 'DTU Training vs TNT')
        test_vs_tnt_results = dtu_test_results + tnt_results
        output_fig_test = os.path.join(args.outdir, f'uncertainty_distribution_both_test_vs_tnt_{checkpoint_name}.png')
        plot_uncertainty_distribution_dual_axis(test_vs_tnt_results, output_fig_test, 'DTU Test vs TNT')
        test_vs_train_results = dtu_test_results + dtu_train_results
        output_fig_test_train = os.path.join(args.outdir, f'uncertainty_distribution_both_test_vs_train_{checkpoint_name}.png')
        plot_uncertainty_distribution_dual_axis(test_vs_train_results, output_fig_test_train, 'DTU Test vs DTU Training')
    else:
        # Single uncertainty type
        train_vs_tnt_results = dtu_train_results + tnt_results
        output_fig_train = os.path.join(args.outdir, f'uncertainty_distribution_{ut}_train_vs_tnt_{checkpoint_name}.png')
        plot_uncertainty_distribution(train_vs_tnt_results, output_fig_train, args.uncertainty_type, 'DTU Training vs TNT')
        test_vs_tnt_results = dtu_test_results + tnt_results
        output_fig_test = os.path.join(args.outdir, f'uncertainty_distribution_{ut}_test_vs_tnt_{checkpoint_name}.png')
        plot_uncertainty_distribution(test_vs_tnt_results, output_fig_test, args.uncertainty_type, 'DTU Test vs TNT')
        test_vs_train_results = dtu_test_results + dtu_train_results
        output_fig_test_train = os.path.join(args.outdir, f'uncertainty_distribution_{ut}_test_vs_train_{checkpoint_name}.png')
        plot_uncertainty_distribution(test_vs_train_results, output_fig_test_train, args.uncertainty_type, 'DTU Test vs DTU Training')
    
    # Save JSON for reproducibility
    all_results = {
        'train_vs_tnt': train_vs_tnt_results,
        'test_vs_tnt': test_vs_tnt_results,
        'test_vs_train': test_vs_train_results
    }
    output_json = os.path.join(args.outdir, f'uncertainty_distribution_{ut}_{checkpoint_name}.json')
    with open(output_json, 'w') as f:
        json.dump(_json_serializable(all_results), f, indent=2)
    print(f'Results saved to {output_json}')
    
    # Print summary statistics
    print('\n=== Summary Statistics ===')
    if ut == 'both':
        dtu_train_al = [r['mean_aleatoric'] for r in dtu_train_results]
        dtu_train_ep = [r['mean_epistemic'] for r in dtu_train_results]
        dtu_test_al = [r['mean_aleatoric'] for r in dtu_test_results]
        dtu_test_ep = [r['mean_epistemic'] for r in dtu_test_results]
        tnt_al = [r['mean_aleatoric'] for r in tnt_results]
        tnt_ep = [r['mean_epistemic'] for r in tnt_results]
        for name, al_vals, ep_vals in [
            ('DTU Training', dtu_train_al, dtu_train_ep),
            ('DTU Test', dtu_test_al, dtu_test_ep),
            ('TNT', tnt_al, tnt_ep),
        ]:
            print(f'{name} ({len(al_vals)} samples):')
            print(f'  Aleatoric  - Mean: {np.mean(al_vals):.6f}, Std: {np.std(al_vals):.6f}, Min: {np.min(al_vals):.6f}, Max: {np.max(al_vals):.6f}')
            print(f'  Epistemic  - Mean: {np.mean(ep_vals):.6f}, Std: {np.std(ep_vals):.6f}, Min: {np.min(ep_vals):.6f}, Max: {np.max(ep_vals):.6f}')
            print()
    else:
        dtu_train_unc = [r['mean_uncertainty'] for r in dtu_train_results]
        dtu_test_unc = [r['mean_uncertainty'] for r in dtu_test_results]
        tnt_unc = [r['mean_uncertainty'] for r in tnt_results]
        print(f'DTU Training Set ({len(dtu_train_results)} samples):')
        print(f'  Mean {ut} uncertainty: {np.mean(dtu_train_unc):.6f}, Std: {np.std(dtu_train_unc):.6f}, Min: {np.min(dtu_train_unc):.6f}, Max: {np.max(dtu_train_unc):.6f}')
        print(f'\nDTU Test Set ({len(dtu_test_results)} samples):')
        print(f'  Mean {ut} uncertainty: {np.mean(dtu_test_unc):.6f}, Std: {np.std(dtu_test_unc):.6f}, Min: {np.min(dtu_test_unc):.6f}, Max: {np.max(dtu_test_unc):.6f}')
        print(f'\nTNT Set ({len(tnt_results)} samples):')
        print(f'  Mean {ut} uncertainty: {np.mean(tnt_unc):.6f}, Std: {np.std(tnt_unc):.6f}, Min: {np.min(tnt_unc):.6f}, Max: {np.max(tnt_unc):.6f}')

    # Print all samples ordered by mean uncertainty (low to high)
    all_samples = dtu_train_results + dtu_test_results + tnt_results
    sorted_samples = sorted(all_samples, key=lambda x: x['mean_uncertainty'])
    print(f'\n=== All Samples (ordered by mean {"epistemic" if ut == "both" else ut} uncertainty, low to high) ===')
    print(f'  DTU Training: {len(dtu_train_results)} samples drawn from {dtu_train_total} total')
    print(f'  DTU Test:     {len(dtu_test_results)} samples drawn from {dtu_test_total} total')
    print(f'  TNT:          {len(tnt_results)} samples drawn from {tnt_total} total')
    if ut == 'both':
        print(f'{"#":>4} {"Dataset":<14} {"Path":<44} {"Aleatoric":>12} {"Epistemic":>12}')
        print('-' * 90)
        for i, r in enumerate(sorted_samples, 1):
            path = r.get('path', r.get('sample_id', '-'))[:44]
            print(f'{i:>4} {r["dataset"]:<14} {path:<44} {r["mean_aleatoric"]:>12.6f} {r["mean_epistemic"]:>12.6f}')
    else:
        col_name = f"Mean {ut}"
        print(f'{"#":>4} {"Dataset":<14} {"Path":<50} {col_name:>20}')
        print('-' * 90)
        for i, r in enumerate(sorted_samples, 1):
            path = r.get('path', r.get('sample_id', '-'))
            print(f'{i:>4} {r["dataset"]:<14} {path:<50} {r["mean_uncertainty"]:>20.6f}')


if __name__ == '__main__':
    main()
