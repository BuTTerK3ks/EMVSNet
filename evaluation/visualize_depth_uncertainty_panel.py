"""
Single-sample depth / uncertainty exports: six PNGs (reference RGB, predicted depth, GT,
error, aleatoric, epistemic), each named from its panel headline.
"""
import argparse
import os
import sys
import tempfile
import ast
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import find_dataset_def
from models import *
from utils import print_args, tocuda
from evidential.models import loss_der

cudnn.benchmark = True

# Fixed colormap range for predicted / GT depth panels (mm), DTU-typical span
DEPTH_VIZ_MIN_MM = 425.0
DEPTH_VIZ_MAX_MM = 935.0

FIG_TEXT_KW = dict(fontweight='bold', fontsize=14)

parser = argparse.ArgumentParser(
    description='Visualize depth prediction, error, and uncertainties for one DTU reference view'
)
parser.add_argument('--scene', required=True,
                    help='DTU scan (e.g. scan47 or scan47_train; _train suffix is stripped for paths)')
parser.add_argument('--ref_view', type=int, required=True,
                    help='0-based reference view index as in Cameras/pair.txt / depth_map_XXXX.pfm')
parser.add_argument('--light_idx', type=int, default=3, help='Lighting index 0–6')

parser.add_argument('--inverse_depth', type=ast.literal_eval, default=False)
parser.add_argument('--origin_size', type=ast.literal_eval, default=False)
parser.add_argument('--max_h', type=int, default=512)
parser.add_argument('--max_w', type=int, default=640)
parser.add_argument('--view_num', type=int, default=5)
parser.add_argument('--image_scale', type=float, default=0.25)

parser.add_argument('--dataset', default='dtu_yao')
parser.add_argument('--trainpath', required=True)

parser.add_argument('--numdepth', type=int, default=192)
parser.add_argument('--interval_scale', type=float, default=1.06)

parser.add_argument('--loadckpt', required=True)
parser.add_argument('--outdir', default='./evaluation/results')
parser.add_argument('--out_name', default=None,
                    help='Subfolder under the scene output dir for this run (default: checkpoint stem)')
parser.add_argument('--evidential_method', type=str, default='der', choices=['der', 'sder'])

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def normalize_scan(name):
    return name.replace('_train', '').strip()


def find_dataset_index(dataset, scan, light_idx, ref_view):
    scan_n = normalize_scan(scan)
    for i, meta in enumerate(dataset.metas):
        s, li, rv, _, flip = meta
        if normalize_scan(s) != scan_n or li != light_idx or rv != ref_view or flip != 0:
            continue
        return i
    raise ValueError(
        f'No dataset entry for scan={scan!r} (normalized {scan_n!r}), '
        f'light_idx={light_idx}, ref_view={ref_view}, flip=0. Check scene name and indices.'
    )


def load_raw_image(datapath, scan, view_1idx, light_idx, image_scale):
    fn = os.path.join(
        datapath,
        'Rectified/{}_train/rect_{:03d}_{}_r5000.png'.format(scan, view_1idx, light_idx),
    )
    img = np.array(Image.open(fn).convert('RGB'), dtype=np.float32) / 255.0
    if image_scale != 1.0:
        from scipy import ndimage
        h, w = img.shape[:2]
        nh, nw = int(image_scale * h), int(image_scale * w)
        img = ndimage.zoom(img, (nh / h, nw / w, 1), order=1)
    return img


def sample_to_batch(sample):
    """Match DataLoader-style tensors for tocuda / model."""
    di = float(sample['depth_interval'])
    return {
        'imgs': torch.from_numpy(sample['imgs']).float().unsqueeze(0),
        'proj_matrices': torch.from_numpy(sample['proj_matrices']).float().unsqueeze(0),
        'depth_values': torch.from_numpy(sample['depth_values']).float().unsqueeze(0),
        'depth': torch.from_numpy(sample['depth']).float().unsqueeze(0),
        'mask': torch.from_numpy(sample['mask']).float().unsqueeze(0),
        'depth_interval': torch.tensor([[di]], dtype=torch.float32),
    }


def tensor_hw(t, batch_idx=0):
    """(B, H, W) or (B, 1, H, W) -> numpy (H, W)."""
    x = t[batch_idx].detach().cpu()
    if x.dim() == 3:
        x = x[0]
    return x.numpy()


def valid_mask_numpy(mask_np, depth_gt_np, depth_pred_np):
    if mask_np.ndim == 3:
        m = (mask_np[0] > 0.5).squeeze()
    else:
        m = (mask_np > 0.5).squeeze()
    if depth_gt_np.ndim == 3:
        gt = depth_gt_np[0]
    else:
        gt = depth_gt_np
    if depth_pred_np.ndim == 3:
        pr = depth_pred_np[0]
    else:
        pr = depth_pred_np
    finite = np.isfinite(gt) & np.isfinite(pr) & (gt > 0)
    return m & finite, gt, pr


def percentile_vmin_vmax(values, valid, p_lo=4, p_hi=96):
    v = values[valid]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(v, [p_lo, p_hi])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def headline_to_basename(headline):
    """Filesystem-safe basename (no extension) matching the plot headline."""
    h = headline.replace('[mm]', 'mm')
    h = h.replace(' ', '_')
    for c in '\\/:*?"<>|':
        h = h.replace(c, '')
    return h


def run_subdir_name(out_name_arg, ck_stem):
    if out_name_arg:
        s = out_name_arg.strip()
        if s.lower().endswith('.png'):
            s = s[:-4]
        return s or ck_stem
    return ck_stem


def save_rgb_panel(img_rgb, path, title, dpi=200):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(np.clip(img_rgb, 0, 1))
    ax.set_title(title, **FIG_TEXT_KW)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()


def save_scalar_panel(data, path, cmap, valid, use_percentile, fixed_clim, title, dpi=200):
    disp = np.array(data, dtype=np.float64).copy()
    disp[~valid] = np.nan
    if use_percentile:
        vmin, vmax = percentile_vmin_vmax(data, valid)
    else:
        vmin, vmax = fixed_clim
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, **FIG_TEXT_KW)
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.setp(
        cbar.ax.get_yticklabels(),
        fontweight=FIG_TEXT_KW['fontweight'],
        fontsize=FIG_TEXT_KW['fontsize'],
    )
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    print_args(args)
    scan = normalize_scan(args.scene)
    list_line = scan + '\n'

    cmap = plt.cm.jet.copy()
    cmap.set_bad(color='black')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(list_line)
        list_path = f.name

    try:
        MVSDataset = find_dataset_def(args.dataset)
        dataset = MVSDataset(
            args.trainpath, list_path, 'train', args.view_num, args.numdepth,
            args.interval_scale, args.inverse_depth, args.origin_size,
            args.light_idx, args.image_scale,
        )
        idx = find_dataset_index(dataset, scan, args.light_idx, args.ref_view)
        sample = dataset[idx]

        print(f'Loading model from {args.loadckpt}')
        model = EMVSNet(
            disparity_level=args.numdepth,
            image_scale=args.image_scale,
            max_h=args.max_h,
            max_w=args.max_w,
            return_depth=False,
            evidential_method=args.evidential_method,
        )
        state_dict = torch.load(args.loadckpt, map_location=device)
        model_state = state_dict.get('model', state_dict)
        new_sd = OrderedDict()
        for k, v in model_state.items():
            new_sd[k[7:] if k.startswith('module.') else k] = v
        model.load_state_dict(new_sd, strict=True)
        model = model.to(device)
        model.eval()

        batch = sample_to_batch(sample)
        sample_cuda = tocuda(batch, non_blocking=True)
        depth_gt = sample_cuda['depth']
        mask = sample_cuda['mask']

        with torch.no_grad():
            probability_volume, evidential, _ = model(
                sample_cuda['imgs'],
                sample_cuda['proj_matrices'],
                sample_cuda['depth_values'],
            )

        if evidential.shape[-2:] != depth_gt.shape[-2:]:
            evidential = F.interpolate(
                evidential, size=depth_gt.shape[-2:], mode='bilinear', align_corners=True,
            )
            probability_volume = F.interpolate(
                probability_volume, size=depth_gt.shape[-2:], mode='bilinear', align_corners=True,
            )
        if mask.shape[-2:] != depth_gt.shape[-2:]:
            mask = F.interpolate(
                mask.unsqueeze(1).float(),
                size=depth_gt.shape[-2:],
                mode='nearest',
                align_corners=False,
            ).squeeze(1)

        outputs = {
            'probability_volume': probability_volume,
            'evidential_prediction': evidential,
        }
        _, depth_est, evidential_outputs = loss_der(
            outputs, depth_gt, mask, sample_cuda['depth_values'],
            method=args.evidential_method, weight_reg=1.0,
        )

        if args.evidential_method == 'sder':
            ale_t = evidential_outputs['aleatoric_sder']
            epi_t = evidential_outputs['epistemic_sder']
        else:
            ale_t = evidential_outputs['aleatoric_der']
            epi_t = evidential_outputs['epistemic_der']

        depth_est_np = tensor_hw(depth_est, 0)
        depth_gt_np = tensor_hw(depth_gt, 0)
        mask_np = mask.cpu().numpy()
        err_np = np.abs(depth_est_np - depth_gt_np)
        ale_np = tensor_hw(ale_t, 0)
        epi_np = tensor_hw(epi_t, 0)

        valid, _, _ = valid_mask_numpy(mask_np, depth_gt_np, depth_est_np)

        view_1idx = args.ref_view + 1
        ref_rgb = load_raw_image(args.trainpath, scan, view_1idx, args.light_idx, args.image_scale)

        out_folder = os.path.join(
            args.outdir,
            f'depth_uncertainty_panel_{scan}_v{args.ref_view}_l{args.light_idx}',
        )
        ck_stem = os.path.basename(args.loadckpt).replace('.ckpt', '')
        sub = run_subdir_name(args.out_name, ck_stem)
        run_folder = os.path.join(out_folder, sub)
        os.makedirs(run_folder, exist_ok=True)

        panels = [
            ('rgb', ref_rgb, 'Reference Image', None, None),
            ('scalar', depth_est_np, 'Depth Prediction [mm]', False, (DEPTH_VIZ_MIN_MM, DEPTH_VIZ_MAX_MM)),
            ('scalar', depth_gt_np, 'Ground Truth Depth [mm]', False, (DEPTH_VIZ_MIN_MM, DEPTH_VIZ_MAX_MM)),
            ('scalar', err_np, 'Prediction Error [mm]', False, (0.0, 50.0)),
            ('scalar', ale_np, 'Aleatoric Uncertainty', True, None),
            ('scalar', epi_np, 'Epistemic Uncertainty', True, None),
        ]

        saved = []
        for kind, data, headline, use_percentile, fixed_clim in panels:
            base = headline_to_basename(headline)
            path = os.path.join(run_folder, f'{base}.png')
            if kind == 'rgb':
                save_rgb_panel(data, path, headline)
            else:
                save_scalar_panel(data, path, cmap, valid, use_percentile, fixed_clim, headline)
            saved.append(path)

        for p in saved:
            print(f'Saved {p}')

    finally:
        os.unlink(list_path)


if __name__ == '__main__':
    main()
