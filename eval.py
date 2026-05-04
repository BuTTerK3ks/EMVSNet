import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import time
from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import *
from evidential.models import uncertainty_der, uncertainty_sder
import ast
from collections import OrderedDict
import numpy as np

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth')

parser.add_argument('--inverse_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)

parser.add_argument('--return_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=True)

parser.add_argument('--max_h', type=int, default=512, help='Maximum image height when training')
parser.add_argument('--max_w', type=int, default=960, help='Maximum image width when training.')
parser.add_argument('--image_scale', type=float, default=1.0, help='pred depth map scale (compared to input image)') 

parser.add_argument('--light_idx', type=int, default=3, help='select while in test')
parser.add_argument('--view_num', type=int, default=7, help='training view num setting')

parser.add_argument('--dataset', default='data_eval_transform', help='select dataset')
parser.add_argument('--testpath', help='testing data path')
parser.add_argument('--testlist', help='testing scan list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=256, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.0, help='the depth interval scale')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--evidential_method', type=str, default='der', choices=['der', 'sder'],
                    help='Evidential method: der (full NIG loss) or sder (simplified, alpha = nu + 1)')
parser.add_argument('--gpus', type=str, default=None, help='GPU IDs to use (e.g., "0,1" for two GPUs)')
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
parser.add_argument('--world_size', type=int, default=-1, help='Number of processes for distributed training')
parser.add_argument('--dist_url', type=str, default='env://', help='URL used to set up distributed training')

# parse arguments and check
args = parser.parse_args()

# Initialize distributed training
if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    # Running with torchrun or torch.distributed.launch
    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    use_ddp = True
elif args.local_rank != -1:
    # Running with manual DDP setup
    args.rank = args.local_rank
    args.world_size = args.world_size if args.world_size > 0 else torch.cuda.device_count()
    use_ddp = True
else:
    # Single GPU or DataParallel mode (fallback)
    args.rank = 0
    args.local_rank = 0
    args.world_size = 1
    use_ddp = False

# Set CUDA_VISIBLE_DEVICES based on --gpus argument (only if not using DDP)
if not use_ddp and args.gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    num_gpus = len(args.gpus.split(','))
else:
    num_gpus = args.world_size if use_ddp else 1

# Initialize DDP process group
if use_ddp:
    dist.init_process_group(
        backend='nccl',
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}')
    is_main_process = (args.rank == 0)
else:
    device = torch.device('cuda:0')
    is_main_process = True

if is_main_process:
    print_args(args)

# Validate that loadckpt is provided before using it
if args.loadckpt is None:
    raise ValueError("--loadckpt argument is required. Please provide a checkpoint file path.")

model_name = str.split(args.loadckpt, '/')[-2] + '_' + str.split(args.loadckpt, '/')[-1]
save_dir = os.path.join(args.outdir, model_name)
if is_main_process:
    if not os.path.exists(save_dir):
        print('save dir', save_dir)
        os.makedirs(save_dir)

# Synchronize all processes before continuing
if use_ddp:
    dist.barrier()

if is_main_process:
    print(f'Using {num_gpus} GPU(s) with DDP: {use_ddp}, evidential_method: {args.evidential_method}')

# run MVS model to save depth maps and confidence maps
def save_depth():
    
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", args.view_num, args.numdepth, args.interval_scale, args.inverse_depth,
                    adaptive_scaling=True, max_h=args.max_h, max_w=args.max_w, sample_scale=1, base_image_size=8)

    # Create distributed sampler for DDP
    if use_ddp:
        test_sampler = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
        test_shuffle = False
    else:
        test_sampler = None
        test_shuffle = False

    TestImgLoader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,  # Per-GPU batch size
        shuffle=test_shuffle,
        sampler=test_sampler,
        num_workers=4, 
        drop_last=False,
        pin_memory=True,
        prefetch_factor=2
    )

    model = EMVSNet(disparity_level=args.numdepth, image_scale=args.image_scale,
                    max_h=args.max_h, max_w=args.max_w, return_depth=args.return_depth,
                    evidential_method=args.evidential_method)


    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))

    '''

    # Allow both keys xxx & module.xxx in dict
    state_dict = torch.load(args.loadckpt)
    if "module.feature.conv0_0.0.weight" in state_dict['model']:
        print("With module in keys")
        model = nn.DataParallel(model)
        model.load_state_dict(state_dict['model'],True)
        
    else:
        print("No module in keys")
        model.load_state_dict(state_dict['model'], True)
        model = nn.DataParallel(model)
        
    '''
    # Load the checkpoint
    state_dict = torch.load(args.loadckpt)['model']  # Assuming 'model' is the key under which the state dict is saved

    # Create a new state dictionary without the 'module.' prefix

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # Remove 'module.' of each key
        new_state_dict[name] = v

    # Load the adjusted state dict into the model
    model.load_state_dict(new_state_dict, strict=True)

    # Move model to device
    model = model.to(device)

    # Wrap model with DDP or DataParallel
    if use_ddp and num_gpus > 1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
        if is_main_process:
            print(f'Wrapped model with DistributedDataParallel for {num_gpus} GPUs')
    elif num_gpus > 1:
        # Fallback to DataParallel if DDP not available
        model = nn.DataParallel(model)
        if is_main_process:
            print(f'Wrapped model with DataParallel for {num_gpus} GPUs')

    model.eval()
    
    # Statistics collection
    avg_statistics = DictAverageMeter()
    inference_times = []
    total_samples = 0
    has_gt = False  # Track if any sample has ground truth
    
    count = -1
    total_time = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            count += 1
            try:
                if is_main_process or batch_idx % 50 == 0:
                    print('[Rank {}] process {}'.format(args.rank, sample['filename']))
                sample_cuda = tocuda(sample, non_blocking=True)
                if is_main_process and batch_idx == 0:
                    print('input shape: ', sample_cuda["imgs"].shape, sample_cuda["proj_matrices"].shape, sample_cuda["depth_values"].shape)
                time_s = time.time()
                outputs_tensor = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])

                one_time = time.time() - time_s
                total_time += one_time
                if is_main_process and count % 50 == 0:
                    print('[Rank {}] avg time: {:.3f}s'.format(args.rank, total_time / 50))
                    total_time = 0

                # Store inference time
                inference_times.append(one_time)
                
                # Compute statistics if ground truth is available
                sample_has_gt = "depth" in sample and "mask" in sample
                if sample_has_gt:
                    has_gt = True  # At least one sample has GT
                    depth_gt_tensor = tocuda(sample["depth"], non_blocking=True)
                    mask_tensor = tocuda(sample["mask"], non_blocking=True)
                    
                    # Extract depth estimate from tensor outputs
                    if isinstance(outputs_tensor, dict):
                        depth_est_tensor = outputs_tensor["depth"]
                    else:
                        # If outputs is a tuple, try to extract depth
                        depth_est_tensor = outputs_tensor["depth"] if hasattr(outputs_tensor, "depth") else outputs_tensor
                    
                    # Compute metrics
                    scalar_outputs = {}
                    scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est_tensor, depth_gt_tensor, mask_tensor > 0.5)
                    scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est_tensor, depth_gt_tensor, mask_tensor > 0.5, 2)
                    scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est_tensor, depth_gt_tensor, mask_tensor > 0.5, 4)
                    scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est_tensor, depth_gt_tensor, mask_tensor > 0.5, 8)
                    scalar_outputs["thres16mm_error"] = Thres_metrics(depth_est_tensor, depth_gt_tensor, mask_tensor > 0.5, 16)
                    scalar_outputs["thres32mm_error"] = Thres_metrics(depth_est_tensor, depth_gt_tensor, mask_tensor > 0.5, 32)
                    
                    # Convert to float and update statistics
                    scalar_outputs = tensor2float(scalar_outputs)
                    avg_statistics.update(scalar_outputs)
                    total_samples += 1
                    
                    if is_main_process and batch_idx % 50 == 0:
                        print('[Rank {}] Iter {}/{}, MAE: {:.3f}mm, thres2mm: {:.3f}%, thres4mm: {:.3f}%, thres8mm: {:.3f}%'.format(
                            args.rank, batch_idx, len(TestImgLoader),
                            scalar_outputs["abs_depth_error"],
                            scalar_outputs["thres2mm_error"] * 100,
                            scalar_outputs["thres4mm_error"] * 100,
                            scalar_outputs["thres8mm_error"] * 100))
                
                outputs = tensor2numpy(outputs_tensor)
                del sample_cuda
                if is_main_process or batch_idx % 50 == 0:
                    print('[Rank {}] Iter {}/{}'.format(args.rank, batch_idx, len(TestImgLoader)))
                filenames = sample["filename"]
                outputs = [outputs]

                # save depth maps and confidence maps (only on main process)
                if is_main_process:
                    for filename, output in zip(filenames, outputs):
                        depth_filename_pfm = os.path.join(save_dir, filename.format('depth_est_{}'.format(0), '.pfm'))
                        confidence_filename_pfm = os.path.join(save_dir, filename.format('confidence_{}'.format(0), '.pfm'))
                        epistemic_filename_pfm = os.path.join(save_dir, filename.format('epistemic_{}'.format(0), '.pfm'))
                        aleatoric_filename_pfm = os.path.join(save_dir, filename.format('aleatoric_{}'.format(0), '.pfm'))

                        depth_filename_png = os.path.join(save_dir, filename.format('depth_png_{}'.format(0), '.png'))
                        aleatoric_filename_png = os.path.join(save_dir, filename.format('aleatoric_{}'.format(0), '.png'))
                        epistemic_filename_png = os.path.join(save_dir, filename.format('epistemic_{}'.format(0), '.png'))

                        os.makedirs(depth_filename_pfm.rsplit('/', 1)[0], exist_ok=True)
                        os.makedirs(confidence_filename_pfm.rsplit('/', 1)[0], exist_ok=True)
                        os.makedirs(depth_filename_png.rsplit('/', 1)[0], exist_ok=True)
                        os.makedirs(aleatoric_filename_png.rsplit('/', 1)[0], exist_ok=True)
                        os.makedirs(epistemic_filename_png.rsplit('/', 1)[0], exist_ok=True)
                        os.makedirs(epistemic_filename_pfm.rsplit('/', 1)[0], exist_ok=True)
                        os.makedirs(aleatoric_filename_pfm.rsplit('/', 1)[0], exist_ok=True)

                        depth_est = output["depth"]
                        photometric_confidence = output["photometric_confidence"]
                        evidential_prediction = output["evidential_prediction"]

                        gamma, nu, alpha, beta = evidential_prediction[0, :, :], evidential_prediction[1, :, :], evidential_prediction[2, :, :], evidential_prediction[3, :, :]

                        # Use proper uncertainty computation based on evidential_method
                        if args.evidential_method == 'sder':
                            aleatoric_1, epistemic_1 = uncertainty_sder(
                                torch.from_numpy(gamma).to(device),
                                torch.from_numpy(nu).to(device),
                                torch.from_numpy(alpha).to(device),
                                torch.from_numpy(beta).to(device)
                            )
                            aleatoric_1 = aleatoric_1.cpu().numpy()
                            epistemic_1 = epistemic_1.cpu().numpy()
                        else:  # der
                            aleatoric_1, epistemic_1 = uncertainty_der(
                                torch.from_numpy(gamma).to(device),
                                torch.from_numpy(nu).to(device),
                                torch.from_numpy(alpha).to(device),
                                torch.from_numpy(beta).to(device)
                            )
                            aleatoric_1 = aleatoric_1.cpu().numpy()
                            epistemic_1 = epistemic_1.cpu().numpy()

                        save_png(gamma, depth_filename_png, title="Estimated Depth", mode="depth")
                        save_png(aleatoric_1, aleatoric_filename_png, title="Aleatoric uncertainty ({})".format(args.evidential_method))
                        save_png(epistemic_1, epistemic_filename_png, title="Epistemic uncertainty ({})".format(args.evidential_method))

                        # save depth maps
                        save_pfm(depth_filename_pfm, gamma)
                        # save confidence maps
                        save_pfm(confidence_filename_pfm, photometric_confidence.squeeze())
                        save_pfm(epistemic_filename_pfm, epistemic_1)
                        save_pfm(aleatoric_filename_pfm, aleatoric_1)
            except Exception as e:
                if is_main_process:
                    print(f"PROBLEM!!! Error: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Synchronize all processes before final statistics
    if use_ddp:
        dist.barrier()
    
    # Aggregate statistics across all processes if using DDP
    if use_ddp and has_gt:
        # Gather statistics from all processes
        gathered_stats = [None] * args.world_size
        dist.all_gather_object(gathered_stats, {
            'statistics': avg_statistics.data,
            'count': avg_statistics.count,
            'inference_times': inference_times,
            'total_samples': total_samples
        })
        
        # Aggregate on main process
        if is_main_process:
            combined_stats = DictAverageMeter()
            all_inference_times = []
            total_samples_all = 0
            
            for stats_dict in gathered_stats:
                if stats_dict and stats_dict['count'] > 0:
                    # Update with each process's statistics
                    for _ in range(stats_dict['count']):
                        combined_stats.update(stats_dict['statistics'])
                    all_inference_times.extend(stats_dict['inference_times'])
                    total_samples_all += stats_dict['total_samples']
            
            avg_statistics = combined_stats
            inference_times = all_inference_times
            total_samples = total_samples_all
    
    # Save and print statistics (only on main process)
    if is_main_process:
        # Compute mean inference time
        mean_inference_time = np.mean(inference_times) if inference_times else 0.0
        
        # Prepare statistics dictionary
        stats_dict = {
            'evidential_method': args.evidential_method,
            'view_num': args.view_num,
            'total_samples': total_samples,
            'mean_inference_time_seconds': float(mean_inference_time),
        }
        
        if has_gt and avg_statistics.count > 0:
            mean_stats = avg_statistics.mean()
            stats_dict.update({
                'mean_absolute_depth_error_mm': float(mean_stats.get('abs_depth_error', 0.0)),
                'thres2mm_error_percent': float(mean_stats.get('thres2mm_error', 0.0) * 100),
                'thres4mm_error_percent': float(mean_stats.get('thres4mm_error', 0.0) * 100),
                'thres8mm_error_percent': float(mean_stats.get('thres8mm_error', 0.0) * 100),
                'thres16mm_error_percent': float(mean_stats.get('thres16mm_error', 0.0) * 100),
                'thres32mm_error_percent': float(mean_stats.get('thres32mm_error', 0.0) * 100),
            })
        
        # Save statistics to JSON
        stats_file = os.path.join(save_dir, 'evaluation_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        # Print statistics
        print('\n' + '='*80)
        print('Evaluation Statistics:')
        print('='*80)
        if has_gt and avg_statistics.count > 0:
            mean_stats = avg_statistics.mean()
            print(f'Mean Absolute Depth Error: {mean_stats.get("abs_depth_error", 0.0):.3f} mm')
            print(f'Threshold Errors:')
            print(f'  - 2mm:  {mean_stats.get("thres2mm_error", 0.0)*100:.2f}%')
            print(f'  - 4mm:  {mean_stats.get("thres4mm_error", 0.0)*100:.2f}%')
            print(f'  - 8mm:  {mean_stats.get("thres8mm_error", 0.0)*100:.2f}%')
            print(f'  - 16mm: {mean_stats.get("thres16mm_error", 0.0)*100:.2f}%')
            print(f'  - 32mm: {mean_stats.get("thres32mm_error", 0.0)*100:.2f}%')
        print(f'Mean Inference Time: {mean_inference_time:.3f} seconds')
        print(f'Number of Views: {args.view_num}')
        print(f'Total Samples: {total_samples}')
        print(f'Evidential Method: {args.evidential_method}')
        print(f'Statistics saved to: {stats_file}')
        print('='*80 + '\n')
    
    # Cleanup DDP
    if use_ddp:
        dist.destroy_process_group()






if __name__ == '__main__':
    # step1. save all the depth maps and the masks in outputs directory
    print('save depth *******************\n')
    save_depth()
