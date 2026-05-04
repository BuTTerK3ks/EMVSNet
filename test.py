import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import *
from datasets import find_dataset_def
from utils import *
from datasets.data_io import *
import ast


def parse_args():
    parser = argparse.ArgumentParser(description='AA-RMVSNet Testing Script')
    parser.add_argument('--loadckpt', required=True, help='Checkpoint to load')
    parser.add_argument('--dataset', default='dtu_yao')
    parser.add_argument('--testpath', required=True)
    parser.add_argument('--testlist', required=True)
    parser.add_argument('--save_dir', required=True, help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--view_num', type=int, default=3)
    parser.add_argument('--numdepth', type=int, default=192)
    parser.add_argument('--interval_scale', type=float, default=1.0)
    parser.add_argument('--inverse_depth', type=ast.literal_eval, default=False)
    parser.add_argument('--origin_size', type=ast.literal_eval, default=False)
    parser.add_argument('--max_h', type=int, default=512)
    parser.add_argument('--max_w', type=int, default=640)
    parser.add_argument('--light_idx', type=int, default=3)
    parser.add_argument('--image_scale', type=float, default=0.5)
    parser.add_argument('--evidential_method', type=str, default='der', choices=['der', 'sder'],
                        help='Evidential method: der (full NIG loss) or sder (simplified, alpha = nu + 1)')
    parser.add_argument('--weight_reg', type=float, default=1.0, help='Regularization weight for evidential loss')
    return parser.parse_args()

def print_param_stats(model, name=""):
    print(f"Parameter stats {name}:")
    for n, p in model.named_parameters():
        print(f"{n}: mean={p.data.mean():.4f}, std={p.data.std():.4f}, min={p.data.min():.4f}, max={p.data.max():.4f}")
        break  # print just the first parameter for brevity

def setup_model(args, device):
    print('Model: EMVSNet')
    print(f'Evidential method: {args.evidential_method}')
    model = EMVSNet(
        disparity_level=args.numdepth,
        image_scale=args.image_scale,
        max_h=args.max_h,
        max_w=args.max_w,
        evidential_method=args.evidential_method
    )
    # Before loading
    print_param_stats(model, "before loading")
    print(f"loading model {args.loadckpt}")
    state_dict = torch.load(args.loadckpt, map_location=device)
    new_state_dict = {}
    for k, v in state_dict['model'].items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # remove "module." prefix
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(device)
    #model.eval()

    # Before loading
    print_param_stats(model, "after loading")
    return model

def get_dataloader(args):
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(
        args.testpath, args.testlist, "test",
        args.view_num, args.numdepth, args.interval_scale,
        args.inverse_depth, args.origin_size, args.light_idx, args.image_scale
    )
    loader = DataLoader(
        test_dataset, args.batch_size, shuffle=False,
        num_workers=4, drop_last=False, prefetch_factor=5
    )
    return loader

@make_nograd_func
def test_sample(model, sample, device, evidential_method='der', weight_reg=1.0):
    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]
    imgs = sample_cuda["imgs"]

    probability_volume, evidential, probabilities = model(
        sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"]
    )
    outputs = {
        "probability_volume": probability_volume,
        'evidential_prediction': evidential
    }

    pred_shape = probability_volume.shape[-2:]
    if depth_gt.shape[-2:] != pred_shape:
        depth_gt = F.interpolate(depth_gt.unsqueeze(0), size=pred_shape, mode="nearest").squeeze(0)
    if mask.shape[-2:] != pred_shape:
        mask = F.interpolate(mask.unsqueeze(0), size=pred_shape, mode="nearest").squeeze(0)

    loss, depth_est, evidential_outputs = loss_der(
        outputs, depth_gt, mask, sample_cuda["depth_values"],
        method=evidential_method, weight_reg=weight_reg
    )

    # Unpack evidential_outputs directly into the result dict
    result_dict = {
        "depth_gt": depth_gt.cpu(),
        "mask": mask.cpu(),
        "depth_pred": depth_est.cpu(),
        "imgs": imgs.cpu(),
        **{k: v.cpu() for k, v in evidential_outputs.items()},
    }
    return result_dict

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Optionally move this to argparse as well

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    model = setup_model(args, device)
    dataloader = get_dataloader(args)

    print("Starting inference...")
    for batch_idx, sample in enumerate(dataloader):
        result = test_sample(model, sample, device, 
                           evidential_method=args.evidential_method, 
                           weight_reg=args.weight_reg)
        out_path = os.path.join(args.save_dir, f"result_{batch_idx:05d}.pt")
        torch.save(result, out_path)
        print(f"Saved {out_path}")
    print("All results saved.")

if __name__ == "__main__":
    main()
