import argparse
from itertools import islice
import os
import queue
import sys
import threading
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import time
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from utils import *
import ast
from datasets.data_io import *

from helpers.statistics import *

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch Codebase for AA-RMVSNet')
parser.add_argument('--mode', default='train', help='train, val or test')

parser.add_argument('--inverse_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)
parser.add_argument('--origin_size', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)
parser.add_argument('--save_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)

parser.add_argument('--max_h', type=int, default=512, help='Maximum image height when training')
parser.add_argument('--max_w', type=int, default=640, help='Maximum image width when training.')

parser.add_argument('--light_idx', type=int, default=3, help='select while in test')
parser.add_argument('--view_num', type=int, default=3, help='training view num setting')

parser.add_argument('--image_scale', type=float, default=0.25, help='pred depth map scale')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--vallist', help='val list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=6, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--optimizer', type=str, required=True, choices=['adam', 'adamw'],
                    help='Optimizer to use: adam or adamw')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for AdamW')

parser.add_argument('--batch_size', type=int, default=12, help='train batch size')
parser.add_argument('--num_workers', type=int, default=4, help='DataLoader num_workers per process')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--save_dir', default=None, help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=20, help='print and summary frequency')
parser.add_argument('--save_freq_checkpoint', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

parser.add_argument('--evidential_method', type=str, required=True, choices=['der', 'sder'],
                    help='Evidential method: der (full NIG loss) or sder (simplified, alpha = nu + 1)')
parser.add_argument('--weight_reg', type=float, default=1.0, help='Regularization weight for evidential loss')
parser.add_argument('--gpus', type=str, default=None, help='GPU IDs to use (e.g., "0,1" for two GPUs)')
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
parser.add_argument('--world_size', type=int, default=-1, help='Number of processes for distributed training')
parser.add_argument('--dist_url', type=str, default='env://', help='URL used to set up distributed training')

parser.add_argument('--early_stopping', action='store_true', default=False,
                    help='Enable early stopping based on validation MAE')
parser.add_argument('--early_stopping_patience', type=int, default=None,
                    help='Number of epochs to wait without improvement before stopping (default: 2, only used if --early_stopping is enabled)')
parser.add_argument('--early_stopping_min_delta', type=float, default=None,
                    help='Minimum change in MAE to qualify as an improvement (default: 10.0, only used if --early_stopping is enabled)')

# parse arguments and check
args = parser.parse_args()

# Optional: use TRAINPATH from env when --trainpath not given (e.g. Docker default CMD override)
if args.trainpath is None and os.environ.get('TRAINPATH'):
    args.trainpath = os.environ['TRAINPATH']

# Validate early stopping arguments
if not args.early_stopping:
    if args.early_stopping_patience is not None:
        raise ValueError("--early_stopping_patience can only be used when --early_stopping is enabled")
    if args.early_stopping_min_delta is not None:
        raise ValueError("--early_stopping_min_delta can only be used when --early_stopping is enabled")
else:
    # Set defaults if early stopping is enabled but values not provided
    if args.early_stopping_patience is None:
        args.early_stopping_patience = 2
    if args.early_stopping_min_delta is None:
        args.early_stopping_min_delta = 10.0

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

if args.resume:
    assert args.mode == "train"
    assert args.loadckpt is None
if args.testpath is None:
    args.testpath = args.trainpath

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

# Automatically nest training runs inside timestamped folders unless resuming
if args.mode == "train" and not args.resume:
    if use_ddp:
        run_logdir = None
        if is_main_process:
            run_logdir = os.path.join(args.logdir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        obj_list = [run_logdir]
        dist.broadcast_object_list(obj_list, src=0)
        args.logdir = obj_list[0]
    else:
        args.logdir = os.path.join(args.logdir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# create logger (only on main process)
if is_main_process:
    os.makedirs(args.logdir, exist_ok=True)
    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)
    print("creating new summary file")
    logger = SummaryWriter(args.logdir)
else:
    logger = None

# Background summary writer (main process only): offload TensorBoard I/O from training critical path
summary_queue = None
summary_writer_thread = None
if is_main_process and logger is not None:
    summary_queue = queue.Queue()

    def _summary_writer():
        while True:
            item = summary_queue.get()
            if item is None:
                break
            mode, scalar_dict, images_dict, global_step, lr = item
            save_scalars(logger, mode, scalar_dict, global_step)
            if mode == 'train' and lr is not None:
                logger.add_scalar('train/lr', lr, global_step)
            save_images(logger, mode, images_dict, global_step)

    summary_writer_thread = threading.Thread(target=_summary_writer, daemon=False)
    summary_writer_thread.start()

# Synchronize all processes before continuing
if use_ddp:
    dist.barrier()

print(f"[Rank {args.rank}] argv:", sys.argv[1:])
if is_main_process:
    print_args(args)

# Batch size: args.batch_size is per-GPU batch size in DDP mode
# Effective batch size = per-GPU batch size * number of GPUs
effective_batch_size = args.batch_size * num_gpus
effective_lr = args.lr * num_gpus
if is_main_process:
    print(f'Using {num_gpus} GPU(s) with DDP: batch_size={effective_batch_size} (per-GPU={args.batch_size}), lr={effective_lr} (base={args.lr})')

SAVE_DEPTH = args.save_depth
if SAVE_DEPTH:
    if args.save_dir is None:
        sub_dir, ckpt_name = os.path.split(args.loadckpt)
        index = ckpt_name[6:-5]
        save_dir = os.path.join(sub_dir, index)
    else:
        save_dir = args.save_dir
    print(os.path.exists(save_dir), ' exists', save_dir)
    if not os.path.exists(save_dir):
        print('save dir', save_dir)
        os.makedirs(save_dir)


MVSDataset = find_dataset_def(args.dataset)
train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", args.view_num, args.numdepth, args.interval_scale, args.inverse_depth, args.origin_size, -1, args.image_scale) # Training with False, Test with inverse_depth
val_dataset = MVSDataset(args.trainpath, args.vallist, "val", 5, args.numdepth, args.interval_scale, args.inverse_depth, args.origin_size, args.light_idx, args.image_scale) #view_num = 5, light_idx = 3
test_dataset = MVSDataset(args.testpath, args.testlist, "test", args.view_num, args.numdepth, args.interval_scale, args.inverse_depth, args.origin_size, args.light_idx, args.image_scale) # use 3

# Create distributed samplers for DDP
if use_ddp:
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    train_shuffle = False  # Shuffle handled by sampler
else:
    train_sampler = None
    val_sampler = None
    test_sampler = None
    train_shuffle = True

TrainImgLoader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,  # Per-GPU batch size
    shuffle=train_shuffle,
    sampler=train_sampler,
    num_workers=args.num_workers,
    drop_last=True,
    pin_memory=True,  # Faster CPU-GPU transfer
    prefetch_factor=4,
    persistent_workers=(args.num_workers > 0),
)
ValImgLoader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,  # Per-GPU batch size
    shuffle=False,
    sampler=val_sampler,
    num_workers=args.num_workers,
    drop_last=False,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=(args.num_workers > 0),
)
TestImgLoader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,  # Per-GPU batch size
    shuffle=False,
    sampler=test_sampler,
    num_workers=args.num_workers,
    drop_last=False,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=(args.num_workers > 0),
)

if is_main_process:
    print('Model: EMVSNet')
    print(f'Evidential method: {args.evidential_method}')
model = EMVSNet(disparity_level=args.numdepth, image_scale=args.image_scale, max_h=args.max_h, max_w=args.max_w, evidential_method=args.evidential_method)

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if is_main_process:
    print(f'Total Parameters: {total_params:,}')
    print(f'Training Parameters: {total_trainable_params:,}')

# Move model to GPU
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

if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
elif args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=args.weight_decay)
else:
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")

print(f'Optimizer: {args.optimizer} \n')

# Load checkpoint or resume training
start_epoch = 0
best_val_mae = float('inf')
best_val_epoch = -1
early_stopping_patience_counter = 0  # Global variable for early stopping state
if args.mode == "train" and args.resume:
    # Resume training: load latest checkpoint from logdir
    if is_main_process:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming from:", loadckpt)
    else:
        loadckpt = None
    
    # Synchronize all processes
    if use_ddp:
        dist.barrier()
        if not is_main_process:
            # Other processes need to know the checkpoint path
            loadckpt = os.path.join(args.logdir, sorted([fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")], 
                                                         key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1])
    
    state_dict = torch.load(loadckpt, map_location=device)
    
    # Strip "module." prefix if present (for both DDP and DataParallel)
    model_state_dict = {}
    for k, v in state_dict['model'].items():
        key = k[7:] if k.startswith("module.") else k
        model_state_dict[key] = v
    
    # Load into underlying model (unwrap if DDP or DataParallel)
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(model_state_dict, strict=True)
    
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
    if 'best_val_mae' in state_dict:
        best_val_mae = state_dict['best_val_mae']
        best_val_epoch = state_dict.get('best_val_epoch', -1)
    if is_main_process:
        print(f"Resumed from epoch {start_epoch}")
        if best_val_mae < float('inf'):
            print(f"Best validation MAE so far: {best_val_mae:.3f} at epoch {best_val_epoch}")
    # Restore early stopping state if enabled and available (for all processes)
    if args.early_stopping and 'early_stopping_patience_counter' in state_dict:
        early_stopping_patience_counter = state_dict['early_stopping_patience_counter']
        if is_main_process:
            print(f"Resumed early stopping state: patience_counter={early_stopping_patience_counter}")

elif args.loadckpt:
    # Load checkpoint specified by --loadckpt
    if is_main_process:
        print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=device)
    
    # Strip "module." prefix if present
    model_state_dict = {}
    for k, v in state_dict['model'].items():
        key = k[7:] if k.startswith("module.") else k
        model_state_dict[key] = v
    
    # Load into underlying model (unwrap if DDP or DataParallel)
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(model_state_dict, strict=True)

if is_main_process:
    print("start at epoch {}".format(start_epoch))


# main function
def train():
    global best_val_mae, best_val_epoch, early_stopping_patience_counter
    print('run train()')
    
    # Initialize early stopping state (use restored value if resuming, otherwise 0)
    if not (args.mode == "train" and args.resume):
        early_stopping_patience_counter = 0
    if args.early_stopping:
        if is_main_process:
            print(f"Early stopping enabled: patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta}")
            if args.mode == "train" and args.resume:
                print(f"Resuming with patience_counter={early_stopping_patience_counter}")
    
    T_max = args.epochs * len(TrainImgLoader)  # total number of batches
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=2e-06)
    ## get intermediate learning rate - fix: step for all batches in previous epochs, not just epochs
    if start_epoch > 0:
        for _ in range(start_epoch * len(TrainImgLoader)):
            lr_scheduler.step()

    import warnings
    warnings.simplefilter("always")

    for epoch_idx in range(start_epoch, args.epochs):
        if is_main_process:
            print('Epoch {}/{}:'.format(epoch_idx, args.epochs))
        global_step = len(TrainImgLoader) * epoch_idx
        if is_main_process:
            print('Start Training')

        # Set epoch for distributed sampler (ensures different data each epoch)
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch_idx)

        # training
        #TODO Hier wird nur bis x trainiert
        for batch_idx, sample in enumerate(TrainImgLoader):
        #for batch_idx, sample in enumerate(islice(TrainImgLoader, 0, 10, 1)):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = (global_step % args.summary_freq == 0)
            if batch_idx == 0:
                do_summary = False
            loss, scalar_outputs, image_outputs, evidential_outputs = train_sample(sample, detailed_summary=do_summary)

            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            if do_summary and is_main_process and summary_queue is not None:
                images_numpy = tensor2numpy(image_outputs)
                summary_queue.put(('train', scalar_outputs, images_numpy, global_step, lr))
            del scalar_outputs, image_outputs
            if is_main_process or batch_idx % 50 == 0:  # Print less frequently on non-main processes
                print(
                    '[Rank {}] Epoch {}/{}, Iter {}/{}, LR {}, train loss = {:.3f}, time = {:.3f}'.format(
                        args.rank, epoch_idx, args.epochs, batch_idx,
                        len(TrainImgLoader), lr, loss,
                        time.time() - start_time))
            lr_scheduler.step()

        # Validation loop (only on main process for metrics, but all processes for DDP)
        if is_main_process:
            print('Start Validation')
        avg_val_scalars = DictAverageMeter()
        
        # Set epoch for distributed sampler
        if use_ddp and val_sampler is not None:
            val_sampler.set_epoch(epoch_idx)
        
        for batch_idx, sample in enumerate(ValImgLoader):
            start_time = time.time()
            global_step = len(ValImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs, evidential_outputs = test_sample(sample, detailed_summary=do_summary)
            if do_summary and is_main_process and summary_queue is not None:
                images_numpy = tensor2numpy(image_outputs)
                summary_queue.put(('val', scalar_outputs, images_numpy, global_step, None))
            avg_val_scalars.update(scalar_outputs)

            if is_main_process or batch_idx % 50 == 0:
                print('[Rank {}] Epoch: {}/{}, Iter: {}/{}, Views: {}, val loss = {:.3f}, time = {:3f}, mae = {:3f}, thres2mm = {:3f}, thres4mm = {:3f}, thres8mm = {:3f}, thres16mm = {:3f}, thres32mm = {:3f}'.format(
                                    args.rank, epoch_idx, args.epochs, batch_idx,
                                    len(ValImgLoader), 5, loss,
                                    time.time() - start_time,
                                    scalar_outputs["abs_depth_error"], scalar_outputs["thres2mm_error"],
                                    scalar_outputs["thres4mm_error"], scalar_outputs["thres8mm_error"],
                                    scalar_outputs["thres16mm_error"], scalar_outputs["thres32mm_error"]))

            del image_outputs

        # Aggregate validation metrics across all processes if using DDP
        if use_ddp:
            # Gather validation statistics from all processes
            gathered_stats = [None] * args.world_size
            dist.all_gather_object(gathered_stats, {
                'statistics': avg_val_scalars.data,
                'count': avg_val_scalars.count
            })
            
            if is_main_process:
                # Combine statistics from all processes
                # stats_dict['statistics'] already contains accumulated sums from each process
                # We need to sum these sums and sum the counts, not multiply
                combined_stats = DictAverageMeter()
                for stats_dict in gathered_stats:
                    if stats_dict and stats_dict['count'] > 0:
                        if len(combined_stats.data) == 0:
                            # Initialize with first process's statistics
                            combined_stats.data = stats_dict['statistics'].copy()
                            combined_stats.count = stats_dict['count']
                        else:
                            # Add subsequent processes' statistics (sums) and counts
                            for k, v in stats_dict['statistics'].items():
                                combined_stats.data[k] += v
                            combined_stats.count += stats_dict['count']
                avg_val_scalars = combined_stats

        # Early stopping flag - will be broadcast to all processes
        should_stop = False
        
        if is_main_process and logger is not None:
            val_mean = avg_val_scalars.mean()
            save_scalars(logger, 'fullval', val_mean, global_step)
            print("avg_val_scalars:", val_mean)
            
            # Track best validation MAE and save best checkpoint
            current_val_mae = val_mean.get('abs_depth_error', float('inf'))
            
            # Early stopping logic (only if enabled) - check improvement before updating best_val_mae
            improved = False
            if args.early_stopping:
                # Check if validation MAE improved compared to previous best
                improvement_delta = best_val_mae - current_val_mae
                improved = improvement_delta >= args.early_stopping_min_delta
                
                if improved:
                    early_stopping_patience_counter = 0
                    if is_main_process:
                        print(f"Validation MAE improved by {improvement_delta:.3f} (>= {args.early_stopping_min_delta:.3f}), resetting patience counter")
                else:
                    early_stopping_patience_counter += 1
                    if is_main_process:
                        print(f"No improvement in validation MAE (current: {current_val_mae:.3f}, best: {best_val_mae:.3f}, delta needed: {args.early_stopping_min_delta:.3f}). Patience: {early_stopping_patience_counter}/{args.early_stopping_patience}")
            
            # Update best validation MAE and save checkpoint if improved
            if current_val_mae < best_val_mae:
                best_val_mae = current_val_mae
                best_val_epoch = epoch_idx
                # Save best checkpoint
                model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                checkpoint_dict = {
                    'epoch': epoch_idx,
                    'model': model_state,
                    'optimizer': optimizer.state_dict(),
                    'best_val_mae': best_val_mae,
                    'best_val_epoch': best_val_epoch}
                # Add early stopping state if enabled
                if args.early_stopping:
                    checkpoint_dict['early_stopping_patience_counter'] = early_stopping_patience_counter
                torch.save(checkpoint_dict, "{}/best_model.ckpt".format(args.logdir))
                print(f"New best validation MAE: {best_val_mae:.3f} at epoch {epoch_idx}, saved best checkpoint")
            
            # Check if early stopping should be triggered (after updating best checkpoint)
            if args.early_stopping:
                if early_stopping_patience_counter >= args.early_stopping_patience:
                    print(f"\n{'='*80}")
                    print(f"Early stopping triggered!")
                    print(f"Validation MAE did not improve for {args.early_stopping_patience} consecutive epochs.")
                    print(f"Best validation MAE: {best_val_mae:.3f} at epoch {best_val_epoch}")
                    print(f"Stopping training at epoch {epoch_idx}")
                    print(f"{'='*80}\n")
                    should_stop = True
        
        # Broadcast early stopping flag to all processes if using DDP
        if use_ddp and args.early_stopping:
            should_stop_list = [should_stop]
            dist.broadcast_object_list(should_stop_list, src=0)
            should_stop = should_stop_list[0]
        
        # All processes check the flag and break together
        if should_stop:
            break

        # checkpoint (only on main process)
        if is_main_process and (epoch_idx + 1) % args.save_freq_checkpoint == 0:
            # Save model.module.state_dict() if using DDP/DataParallel to avoid "module." prefix
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            checkpoint_dict = {
                'epoch': epoch_idx,
                'model': model_state,
                'optimizer': optimizer.state_dict(),
                'best_val_mae': best_val_mae,
                'best_val_epoch': best_val_epoch}
            # Add early stopping state if enabled
            if args.early_stopping:
                checkpoint_dict['early_stopping_patience_counter'] = early_stopping_patience_counter
            torch.save(checkpoint_dict, "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
    
    # Drain summary queue and stop writer thread so no logs are lost
    if summary_writer_thread is not None:
        summary_queue.put(None)
        summary_writer_thread.join()

    # Synchronize all processes before cleanup to ensure clean exit
    if use_ddp:
        dist.barrier()
    
    # Cleanup DDP
    if use_ddp:
        dist.destroy_process_group()


def train_sample(sample, detailed_summary=False):

    model.train()
    optimizer.zero_grad()
    sample_cuda = tocuda(sample, non_blocking=True)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]
    depth_interval = sample_cuda["depth_interval"]
    depth_value = sample_cuda["depth_values"]

    probability_volume, evidential, probabilities = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])

    outputs = {
        "probability_volume": probability_volume,
        'evidential_prediction': evidential
    }

    loss, depth_est, evidential_outputs = loss_der(outputs, depth_gt, mask, depth_value, 
                                                    method=args.evidential_method, weight_reg=args.weight_reg)

    '''
    # Check for NaN/Inf in loss
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"WARNING: Loss is NaN/Inf: {loss.item()}")
        # Return safe dummy values
        return float('inf'), {}, {}, {}
    '''

    loss.backward()             # No scaling

    # Gradient clipping: stabilizes training with more depth planes (e.g. numdepth=128)
    # and prevents divergence after first epoch when gradients scale with depth dimension.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()           # Direct optimizer step

    std_dev = std_prob(probabilities)
    aleatoric_by_total, epistemic_by_total = divide_by_total(evidential_outputs, method=args.evidential_method)
    error_map = (depth_est - depth_gt).abs() * mask

    # Build image_outputs based on selected method
    image_outputs = {"depth_est": depth_est * mask,
                     "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "std_dev": std_dev,
                     "mask": sample["mask"],
                     "error_map": error_map,
                     }
    
    # Add uncertainty outputs based on selected method
    if args.evidential_method == 'sder':
        image_outputs["aleatoric_sder_"] = evidential_outputs["aleatoric_sder"]
        image_outputs["epistemic_sder_"] = evidential_outputs["epistemic_sder"]
        image_outputs["aleatoric_sder_by_total"] = aleatoric_by_total
        image_outputs["epistemic_sder_by_total"] = epistemic_by_total
    elif args.evidential_method == 'der':
        image_outputs["aleatoric_der_"] = evidential_outputs["aleatoric_der"]
        image_outputs["epistemic_der_"] = evidential_outputs["epistemic_der"]
        image_outputs["aleatoric_der_by_total"] = aleatoric_by_total
        image_outputs["epistemic_der_by_total"] = epistemic_by_total

    scalar_outputs = {"loss": loss}
    # Add uncertainty scalars based on selected method
    if args.evidential_method == 'sder':
        scalar_outputs["aleatoric_sder"] = torch.mean(evidential_outputs["aleatoric_sder"]).item()
        scalar_outputs["epistemic_sder"] = torch.mean(evidential_outputs["epistemic_sder"]).item()
    elif args.evidential_method == 'der':
        scalar_outputs["aleatoric_der"] = torch.mean(evidential_outputs["aleatoric_der"]).item()
        scalar_outputs["epistemic_der"] = torch.mean(evidential_outputs["epistemic_der"]).item()
    scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
    scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
    scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
    scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)
    scalar_outputs["thres16mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 16)
    scalar_outputs["thres32mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 32)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs, evidential_outputs


@make_nograd_func
def test_sample(sample, detailed_summary=True):
    model.eval()
    sample_cuda = tocuda(sample, non_blocking=True)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]
    depth_interval = sample_cuda["depth_interval"]
    depth_value = sample_cuda["depth_values"]
    probability_volume, evidential, probabilities = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])

    outputs = {
        "probability_volume": probability_volume,
        'evidential_prediction': evidential
    }

    prob_volume = outputs['probability_volume']
    loss, depth_est, evidential_outputs = loss_der(outputs, depth_gt, mask, depth_value,
                                                    method=args.evidential_method, weight_reg=args.weight_reg)

    std_dev = std_prob(probabilities)
    aleatoric_by_total, epistemic_by_total = divide_by_total(evidential_outputs, method=args.evidential_method)
    error_map = (depth_est - depth_gt).abs() * mask

    # Build image_outputs based on selected method
    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est": depth_est * mask,
                     "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "std_dev": std_dev,
                     "mask": sample["mask"],
                     "error_map": error_map,
                     }
    
    # Add uncertainty outputs based on selected method
    if args.evidential_method == 'sder':
        image_outputs["aleatoric_sder_"] = evidential_outputs["aleatoric_sder"]
        image_outputs["epistemic_sder_"] = evidential_outputs["epistemic_sder"]
        image_outputs["aleatoric_sder_by_total"] = aleatoric_by_total
        image_outputs["epistemic_sder_by_total"] = epistemic_by_total
    elif args.evidential_method == 'der':
        image_outputs["aleatoric_der_"] = evidential_outputs["aleatoric_der"]
        image_outputs["epistemic_der_"] = evidential_outputs["epistemic_der"]
        image_outputs["aleatoric_der_by_total"] = aleatoric_by_total
        image_outputs["epistemic_der_by_total"] = epistemic_by_total

    # Add uncertainty scalars based on selected method
    if args.evidential_method == 'sder':
        scalar_outputs["aleatoric_sder"] = torch.mean(evidential_outputs["aleatoric_sder"]).item()
        scalar_outputs["epistemic_sder"] = torch.mean(evidential_outputs["epistemic_sder"]).item()
    elif args.evidential_method == 'der':
        scalar_outputs["aleatoric_der"] = torch.mean(evidential_outputs["aleatoric_der"]).item()
        scalar_outputs["epistemic_der"] = torch.mean(evidential_outputs["epistemic_der"]).item()
    scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
    scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
    scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
    scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)
    scalar_outputs["thres16mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 16)
    scalar_outputs["thres32mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 32)

    # clear cache
    #torch.cuda.empty_cache()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs, evidential_outputs

if __name__ == '__main__':
    train()