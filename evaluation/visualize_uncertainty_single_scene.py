"""
Visualize epistemic uncertainty distribution for all images/views of a single DTU scene.
"""
import argparse
import os
import sys
import json
import tempfile
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
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

parser = argparse.ArgumentParser(
    description='Visualize epistemic uncertainty distribution for all views of a single DTU scene'
)
parser.add_argument('--scene', default='scan47',
    help='DTU scene name (e.g. scan47). Default from Depths/scan47_train/depth_map_0032.pfm')
parser.add_argument('--inverse_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)
parser.add_argument('--origin_size', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)

parser.add_argument('--max_h', type=int, default=512, help='Maximum image height')
parser.add_argument('--max_w', type=int, default=640, help='Maximum image width')
parser.add_argument('--light_idx', type=int, default=3, help='select light condition')
parser.add_argument('--view_num', type=int, default=5, help='number of views')
parser.add_argument('--image_scale', type=float, default=0.25, help='pred depth map scale')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset for DTU')
parser.add_argument('--trainpath', help='train/val datapath (for DTU ground truth data)')

parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')

parser.add_argument('--loadckpt', required=True, help='load checkpoint (required)')
parser.add_argument('--outdir', default='./evaluation/results', help='output directory for results')
parser.add_argument('--evidential_method', type=str, default='der', choices=['der', 'sder'],
                    help='Evidential method: der (full NIG loss) or sder (simplified, alpha = nu + 1)')

args = parser.parse_args()

if args.trainpath is None:
    raise ValueError("--trainpath argument is required. Please provide path to DTU training data directory.")

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
def collect_mean_epistemic_dtu(model, dataloader, evidential_method):
    """
    Collect mean epistemic uncertainty per sample from DTU dataset.
    Returns list of dicts with keys: 'path', 'mean_epistemic'
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(dataloader, desc="Processing", unit="batch")):
            sample_cuda = tocuda(sample, non_blocking=True)
            depth_gt = sample_cuda["depth"]
            mask = sample_cuda["mask"]

            probability_volume, evidential, probabilities = model(
                sample_cuda["imgs"],
                sample_cuda["proj_matrices"],
                sample_cuda["depth_values"]
            )

            if evidential.shape[-2:] != depth_gt.shape[-2:]:
                evidential = F.interpolate(
                    evidential, size=depth_gt.shape[-2:],
                    mode='bilinear', align_corners=True
                )
                probability_volume = F.interpolate(
                    probability_volume, size=depth_gt.shape[-2:],
                    mode='bilinear', align_corners=True
                )

            if mask.shape[-2:] != depth_gt.shape[-2:]:
                mask = F.interpolate(
                    mask.unsqueeze(1).float(),
                    size=depth_gt.shape[-2:],
                    mode='nearest', align_corners=False
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

            if evidential_method == 'sder':
                epistemic = evidential_outputs["epistemic_sder"].cpu().numpy()
            else:
                epistemic = evidential_outputs["epistemic_der"].cpu().numpy()

            mask_np = mask.cpu().numpy()
            depth_gt_np = depth_gt.cpu().numpy()
            depth_est_np = depth_est.cpu().numpy()

            for b in range(epistemic.shape[0]):
                if mask_np.ndim == 4:
                    mask_b = (mask_np[b, 0] > 0.5).squeeze()
                else:
                    mask_b = (mask_np[b] > 0.5).squeeze()
                depth_gt_b = depth_gt_np[b, 0] if depth_gt_np.ndim == 4 else depth_gt_np[b]
                depth_est_b = depth_est_np[b, 0] if depth_est_np.ndim == 4 else depth_est_np[b]
                valid_depth = np.isfinite(depth_gt_b) & np.isfinite(depth_est_b) & (depth_gt_b > 0)
                mask_b = mask_b & valid_depth

                if np.any(mask_b):
                    epi_b = epistemic[b, 0] if epistemic.ndim == 4 else epistemic[b]
                    mean_epistemic = float(np.mean(epi_b[mask_b]))

                    if 'name' in sample:
                        path_val = sample['name'][b] if isinstance(sample['name'], list) else sample['name']
                        path = str(path_val) if path_val is not None else f"batch{batch_idx}_sample{b}"
                    else:
                        path = f"batch{batch_idx}_sample{b}"

                    results.append({
                        'path': path,
                        'mean_epistemic': mean_epistemic
                    })

    return results


def plot_uncertainty_distribution(results, output_path, scene_name):
    """Plot sorted bar chart of epistemic uncertainty for all views of the scene."""
    sorted_results = sorted(results, key=lambda x: x['mean_epistemic'])
    sorted_vals = [r['mean_epistemic'] for r in sorted_results]
    paths = [os.path.basename(r.get('path', '')) for r in sorted_results]

    n = len(sorted_results)
    fig_height = 4 + max(0, n * 0.06)  # Extra space for labels
    plt.figure(figsize=(12, fig_height))
    color = 'tab:blue'
    for i, v in enumerate(sorted_vals):
        plt.vlines(i, 0, v, color=color, linewidth=1)
        plt.scatter(i, v, color=color, s=15)

    plt.xticks(range(n), paths, rotation=90, ha='right', fontsize=8)
    plt.ylabel("Mean epistemic uncertainty")
    plt.title(f"Scene {scene_name}: epistemic uncertainty per view ({len(results)} views)")
    plt.tight_layout(rect=[0, 0.02 + min(0.35, n * 0.004), 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Figure saved to {output_path}')
    plt.close()


def main():
    scene = args.scene

    # Create a temporary list file containing only this scene
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(scene + '\n')
        scene_list_path = f.name

    try:
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

        state_dict = torch.load(args.loadckpt, map_location=device)
        if 'model' in state_dict:
            model_state_dict = state_dict['model']
        else:
            model_state_dict = state_dict

        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=True)
        model = model.to(device)
        model.eval()
        print('Model loaded successfully.')

        # Load dataset for this scene only
        print(f'\n=== Loading scene {scene} ===')
        MVSDataset_DTU = find_dataset_def(args.dataset)
        dataset = MVSDataset_DTU(
            args.trainpath, scene_list_path, "train", args.view_num, args.numdepth,
            args.interval_scale, args.inverse_depth, args.origin_size,
            args.light_idx, args.image_scale
        )

        total_views = len(dataset)
        if total_views == 0:
            print(f'No samples found for scene {scene}. Check that the scene exists in the dataset.')
            return

        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, drop_last=False, pin_memory=True
        )

        print(f'Scene {scene}: {total_views} views')
        results = collect_mean_epistemic_dtu(model, loader, args.evidential_method)
        print(f'Collected {len(results)} samples')

        # Plot
        checkpoint_name = os.path.basename(args.loadckpt).replace('.ckpt', '')
        output_fig = os.path.join(args.outdir, f'uncertainty_single_scene_{scene}_{checkpoint_name}.png')
        plot_uncertainty_distribution(results, output_fig, scene)

        # Save JSON
        output_json = os.path.join(args.outdir, f'uncertainty_single_scene_{scene}_{checkpoint_name}.json')
        with open(output_json, 'w') as f:
            json.dump(_json_serializable({'scene': scene, 'results': results}), f, indent=2)
        print(f'Results saved to {output_json}')

        # Summary statistics
        epistemic_vals = [r['mean_epistemic'] for r in results]
        print('\n=== Summary Statistics ===')
        print(f'Scene {scene} ({len(results)} views):')
        print(f'  Mean epistemic uncertainty: {np.mean(epistemic_vals):.6f}')
        print(f'  Std epistemic uncertainty:  {np.std(epistemic_vals):.6f}')
        print(f'  Min: {np.min(epistemic_vals):.6f}, Max: {np.max(epistemic_vals):.6f}')

        # Print all views ordered by mean epistemic uncertainty (low to high)
        sorted_results = sorted(results, key=lambda x: x['mean_epistemic'])
        print(f'\n=== All Views (ordered by mean epistemic uncertainty, low to high) ===')
        print(f'  {len(results)} views (all views of scene {scene})')
        print(f'{"#":>4} {"Path":<55} {"Mean epistemic":>16}')
        print('-' * 80)
        for i, r in enumerate(sorted_results, 1):
            path = r.get('path', '-')
            print(f'{i:>4} {path:<55} {r["mean_epistemic"]:>16.6f}')

    finally:
        os.unlink(scene_list_path)


if __name__ == '__main__':
    main()
