import sys
# set package path
sys.path.append("/content/drive/MyDrive/CardiacSeg")

import argparse
import os
from functools import partial

import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter
import ray
from ray import air, tune

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from data_utils.chgh_dataset import get_loader
from data_utils.utils import get_pids_by_loader
from runners.tuner import run_training
from runners.tester import run_testing
from networks.network import network
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


parser = argparse.ArgumentParser(description="model segmentation pipeline")

# mode
parser.add_argument("--tune_mode", default="", type=str, help="tune mode")
parser.add_argument("--test_mode", action="store_true", help="test mode")
parser.add_argument("--resume_tuner", action="store_true", help="test mode")

# dir and path
parser.add_argument("--exp_name", default="", type=str, help="exp name")
parser.add_argument("--local_dir", default="", type=str, help="tune directory")
parser.add_argument("--data_dir", default="", type=str, help="dataset directory")
parser.add_argument("--model_dir", default="./models", type=str, help="directory to save the models")
parser.add_argument("--log_dir", default="./logs", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--eval_dir", default="./evals", type=str, help="directory to save the eval result")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--filename", default="best_model.pth", help="save model file name")
parser.add_argument("--ssl_pretrained", default=None, type=str, help="use self-supervised pretrained weights")

# train loop
parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
parser.add_argument("--val_every", default=20, type=int, help="validation frequency")
parser.add_argument("--max_epoch", default=2000, type=int, help="max number of training epochs")
parser.add_argument("--early_stop_count", default=0, type=int, help="early stop count")
parser.add_argument("--max_early_stop_count", default=10, type=int, help="max early stop count")
parser.add_argument("--save_checkpoint_freq", default=1, type=int, help="save final checkpoint freq, if value is 0 won't save.")

# data
parser.add_argument("--data_dicts_json", default=None, type=str, help="data dicts json")
parser.add_argument("--fold", default=1, type=int, help="index of fold")
parser.add_argument("--split_train_ratio", default=0.6, type=float, help="split train ratio")
parser.add_argument("--num_fold", default=2, type=int, help="num fold")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--pin_memory", action="store_true", help="pin memory")
parser.add_argument("--workers", default=2, type=int, help="number of workers")

# transform
parser.add_argument("--num_samples", default=2, type=int, help="number of samples")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=0.7, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=0.7, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--rand_flipd_prob", default=0.1, type=float, help="RandFlipd aug probability")
parser.add_argument("--rand_rotate90d_prob", default=0.1, type=float, help="RandRotate90d aug probability")
parser.add_argument("--rand_shift_intensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")

# model
parser.add_argument("--model_name", default=None, type=str, help="model name")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")

# optimizer
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")

# scheduler
parser.add_argument("--lrschedule", default=None, type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")

# infer
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--infer_overlap", default=0.25, type=float, help="sliding window inference overlap")


args = parser.parse_args()

def main(config):
    if args.tune_mode == 'transform_intesity':
        args.a_min = config['a_min']
        args.a_max = config['a_max']
    elif args.tune_mode == 'transform_space':
        search_space = config['space']
        args.space_x = search_space[0]
        args.space_y = search_space[1]
        args.space_z = search_space[2]
    elif args.tune_mode == 'transform_roi':
        search_space = config['roi']
        args.roi_x = search_space[0]
        args.roi_y = search_space[1]
        args.roi_z = search_space[2]
    elif args.tune_mode == 'optim_adamw':
        args.optim_lr = config['optim_lr']
        args.reg_weight = config['reg_weight']

    main_worker(args)


def main_worker(args):
    # # make dir
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # device
    if torch.cuda.is_available():
        print("cuda is available")
        args.device = torch.device("cuda")
    else:
        print("cuda is not available")
        args.device = torch.device("cpu")

    # load train and test data
    loader = get_loader(args)

    # model
    model = network(args.model_name, args)

    # loss
    dice_loss = DiceCELoss(to_onehot_y=True, softmax=True)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)

    # lrschedule
    if args.lrschedule == "warmup_cosine":
        print(f'lrschedule: {args.lrschedule}')
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epoch
        )
    else:
        scheduler = None

    # check point
    start_epoch = args.start_epoch
    early_stop_count = args.early_stop_count
    best_acc = 0
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        # load model
        model.load_state_dict(checkpoint["state_dict"])
        # load optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])
        # load lrschedule
        if args.lrschedule is not None and 'scheduler' in checkpoint:
          scheduler.load_state_dict(checkpoint['scheduler'])
        # load check point epoch and best acc
        if "epoch" in checkpoint and args.start_epoch == 0:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        if "early_stop_count" in checkpoint and args.early_stop_count == 0:
            early_stop_count = checkpoint["early_stop_count"]
        print(
          "=> loaded checkpoint '{}' (epoch {}) (bestacc {}) (early stop count {})"\
          .format(args.checkpoint, start_epoch, best_acc, early_stop_count)
        )

    # inferer
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    # writer
    writer = SummaryWriter(log_dir=args.log_dir)

    if not args.test_mode:
        tr_loader, val_loader = loader
        
        # training
        run_training(
            start_epoch=start_epoch,
            best_acc=best_acc,
            early_stop_count=early_stop_count,
            model=model,
            train_loader=tr_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_func=dice_loss,
            acc_func=dice_acc,
            model_inferer=model_inferer,
            post_label=post_label,
            post_pred=post_pred,
            writer=writer,
            args=args,
        )
        

if __name__ == "__main__":
    if args.tune_mode == 'transform_intensity':
        search_space = {
            "a_min": tune.grid_search([-42, 13, 32]),
            "a_max": tune.grid_search([423, 320, 294])
        }
    elif args.tune_mode == 'transform_space':
        search_space = {
            "space": tune.grid_search([
              [0.76,0.76,1.0],
              [1.0,1.0,1.0],
              [1.14,1.14,1.5],
              [1.5,1.5,1.5],
              [1.52,1.52,2.0]
            ]),
        }
    elif args.tune_mode == 'transform_roi':
        search_space = {
            "roi": tune.grid_search([
              [64,64,64],
              [96,96,96],
              [128,128,128],
            ]),
        }
    elif args.tune_mode == 'optim_adamw':
        search_space = {
            "optim_lr": tune.grid_search([
              4e-2,
              4e-3,
              4e-4,
            ]),
            "reg_weight": tune.grid_search([
              5e-2,
              5e-3,
              5e-4,
              5e-5,
            ]),
        }
    else:
      raise ValueError("Invalid args tune_mode")

    trainable_with_cpu_gpu = tune.with_resources(main, {"cpu": 2, "gpu": 1})

    if args.resume_tuner:
        print(f'resume tuner form {args.local_dir}')
        restored_tuner = tune.Tuner.restore(os.path.join(args.local_dir, args.exp_name))
        result = restored_tuner.fit()
    else:
        tuner = tune.Tuner(
            trainable_with_cpu_gpu,
            param_space=search_space,
            run_config=air.RunConfig(
              name=args.exp_name,
              local_dir=args.local_dir,
            )
        )
        tuner.fit()
    
