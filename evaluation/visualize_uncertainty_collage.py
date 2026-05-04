"""
Create collages for all views of a DTU scene: original image, masked image, uncertainty overlay.
Processes the entire scene and saves each view as a .png in an output folder.
"""
import argparse
import os
import sys
import tempfile
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
from tqdm import tqdm
import ast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import *
from evidential.models import loss_der, uncertainty_der, uncertainty_sder, disparity_regression

cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description='Create collages for all views of a DTU scene: original, masked, uncertainty overlay'
)
parser.add_argument('--scene', default='scan47_train',
    help='Scene name (e.g. scan47_train or scan47)')
parser.add_argument('--light_idx', type=int, default=3, help='Light condition (0-6)')
parser.add_argument('--inverse_depth', type=ast.literal_eval, default=False)
parser.add_argument('--origin_size', type=ast.literal_eval, default=False)

parser.add_argument('--max_h', type=int, default=512)
parser.add_argument('--max_w', type=int, default=640)
parser.add_argument('--view_num', type=int, default=5)
parser.add_argument('--image_scale', type=float, default=0.25)

parser.add_argument('--dataset', default='dtu_yao')
parser.add_argument('--trainpath', required=True, help='DTU data path')

parser.add_argument('--numdepth', type=int, default=192)
parser.add_argument('--interval_scale', type=float, default=1.06)

parser.add_argument('--loadckpt', required=True)
parser.add_argument('--outdir', default='./evaluation/results')
parser.add_argument('--evidential_method', type=str, default='der', choices=['der', 'sder'])

args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_scene_view_indices(dataset, scan, light_idx):
    """Return list of (idx, ref_view) for all non-flipped views of the scene."""
    indices = []
    seen_ref_views = set()
    for i in range(len(dataset)):
        meta = dataset.metas[i]
        s, li, rv, _, flip = meta
        if flip != 0 or s != scan or li != light_idx:
            continue
        if rv in seen_ref_views:
            continue
        seen_ref_views.add(rv)
        indices.append((i, rv))
    return sorted(indices, key=lambda x: x[1])


def load_raw_image(datapath, scan, view_1idx, light_idx, image_scale):
    """Load image as RGB 0-1 without normalization."""
    fn = os.path.join(datapath, 'Rectified/{}_train/rect_{:03d}_{}_r5000.png'.format(scan, view_1idx, light_idx))
    img = np.array(Image.open(fn).convert('RGB'), dtype=np.float32) / 255.0
    if image_scale != 1.0:
        from scipy import ndimage
        h, w = img.shape[:2]
        nh, nw = int(image_scale * h), int(image_scale * w)
        img = ndimage.zoom(img, (nh / h, nw / w, 1), order=1)
    return img


def create_collage(orig_img, mask, epistemic, output_path, alpha_overlay=0.5):
    """Create 3-panel collage: original, masked, uncertainty overlay."""
    if mask.dtype != bool and mask.max() <= 1:
        mask = mask > 0.5
    if orig_img.ndim != 3 or orig_img.shape[-1] != 3:
        orig_img = np.tile(orig_img[..., np.newaxis], (1, 1, 3))

    h, w = orig_img.shape[:2]
    if epistemic.shape[0] != h or epistemic.shape[1] != w:
        epistemic = np.array(
            F.interpolate(
                torch.from_numpy(epistemic).float().unsqueeze(0).unsqueeze(0),
                size=(h, w), mode='bilinear', align_corners=True
            ).squeeze().numpy()
        )
    if mask.shape[0] != h or mask.shape[1] != w:
        mask_float = np.array(mask, dtype=np.float32)
        mask = np.array(
            F.interpolate(
                torch.from_numpy(mask_float).unsqueeze(0).unsqueeze(0),
                size=(h, w), mode='nearest'
            ).squeeze().numpy() > 0.5
        )

    img_orig = np.clip(orig_img, 0, 1)
    img_masked = img_orig.copy()
    img_masked[~mask] = 0

    epi_norm = epistemic.copy()
    epi_valid = epistemic[mask]
    if len(epi_valid) > 0:
        vmin, vmax = np.percentile(epi_valid, [2, 98])  # 2% clipping for visibility
        epi_norm = np.clip((epistemic - vmin) / (vmax - vmin + 1e-8), 0, 1)
    else:
        epi_norm = np.zeros_like(epistemic)
    epi_norm[~mask] = np.nan
    cmap = plt.cm.get_cmap('jet')
    overlay_rgba = cmap(epi_norm)
    overlay_rgba[..., 3] = alpha_overlay * np.where(mask, 1, 0)
    overlay_rgba[~mask, 3] = 0
    img_overlay = img_orig.copy()
    for c in range(3):
        img_overlay[..., c] = overlay_rgba[..., c] * overlay_rgba[..., 3] + img_orig[..., c] * (1 - overlay_rgba[..., 3])
    img_overlay = np.clip(img_overlay, 0, 1)
    img_overlay[~mask] = 0  # Apply mask: black outside valid region

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_orig)
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(img_masked)
    axes[1].set_title('Masked')
    axes[1].axis('off')
    axes[2].imshow(img_overlay)
    axes[2].set_title('Epistemic')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    scan = args.scene.replace('_train', '').strip()
    light_idx = args.light_idx

    # Create output folder for this scene
    scene_folder = os.path.join(args.outdir, f'collage_{scan}')
    os.makedirs(scene_folder, exist_ok=True)
    print(f'Output folder: {scene_folder}')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(scan + '\n')
        list_path = f.name

    try:
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
        model_state = state_dict.get('model', state_dict)
        new_sd = OrderedDict()
        for k, v in model_state.items():
            new_sd[k[7:] if k.startswith('module.') else k] = v
        model.load_state_dict(new_sd, strict=True)
        model = model.to(device)
        model.eval()

        MVSDataset = find_dataset_def(args.dataset)
        dataset = MVSDataset(
            args.trainpath, list_path, "train", args.view_num, args.numdepth,
            args.interval_scale, args.inverse_depth, args.origin_size,
            light_idx, args.image_scale
        )

        view_indices = get_scene_view_indices(dataset, scan, light_idx)
        if not view_indices:
            print(f'No views found for scan={scan}, light_idx={light_idx}')
            return

        print(f'Processing {len(view_indices)} views of scene {scan}...')
        for idx, ref_view in tqdm(view_indices, desc='Views', unit='view'):
            sample = dataset[idx]
            view_1idx = ref_view + 1  # 1-indexed for filename

            model_input = {
                "imgs": torch.from_numpy(sample["imgs"]).float().unsqueeze(0),
                "proj_matrices": torch.from_numpy(sample["proj_matrices"]).float().unsqueeze(0),
                "depth_values": torch.from_numpy(sample["depth_values"]).float().unsqueeze(0),
            }
            sample_cuda = tocuda(model_input, non_blocking=True)

            with torch.no_grad():
                pv, evidential, _ = model(
                    sample_cuda["imgs"],
                    sample_cuda["proj_matrices"],
                    sample_cuda["depth_values"]
                )
                gamma, nu, alpha, beta = torch.unbind(evidential, dim=1)
                if args.evidential_method == 'sder':
                    _, epistemic = uncertainty_sder(gamma, nu, alpha, beta)
                else:
                    _, epistemic = uncertainty_der(gamma, nu, alpha, beta)
                epistemic_np = epistemic[0].cpu().numpy()

            mask_np = np.array(sample["mask"])
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            mask_bool = mask_np.astype(bool) if mask_np.max() <= 1 else mask_np > 0.5

            orig_img = load_raw_image(args.trainpath, scan, view_1idx, light_idx, args.image_scale)

            out_name = f'rect_{view_1idx:03d}_{light_idx}_r5000.png'
            out_path = os.path.join(scene_folder, out_name)
            create_collage(orig_img, mask_bool, epistemic_np, out_path, alpha_overlay=0.5)

        print(f'Saved {len(view_indices)} collages to {scene_folder}')

    finally:
        os.unlink(list_path)


if __name__ == '__main__':
    main()
