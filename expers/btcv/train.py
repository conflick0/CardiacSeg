import sys

# set package path
sys.path.append("/content/drive/MyDrive/CardiacSeg")

import argparse
import os
from functools import partial

import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from data_utils.btcv_dataset import get_loader
from data_utils.utils import get_pids_by_loader
from runners.trainer import run_training
from runners.tester import run_testing
from networks.network import network
from losses.loss import loss_func
from optimizers.optimizer import optim

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


parser = argparse.ArgumentParser(description="model segmentation pipeline")

# mode
parser.add_argument("--test_mode", action="store_true", help="test mode")

# dir and path
parser.add_argument("--data_dir", default="", type=str, help="dataset directory")
parser.add_argument("--model_dir", default="models", type=str, help="directory to save the models")
parser.add_argument("--log_dir", default="logs", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--eval_dir", default="evals", type=str, help="directory to save the eval result")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--filename", default="best_model.pth", help="save model file name")

# train loop
parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
parser.add_argument("--val_every", default=20, type=int, help="validation frequency")
parser.add_argument("--max_epoch", default=2000, type=int, help="max number of training epochs")

# data
parser.add_argument("--data_dicts_json", default=None, type=str, help="data dicts json")
parser.add_argument("--fold", default=4, type=int, help="index of fold")
parser.add_argument("--split_train_ratio", default=0.8, type=float, help="split train ratio")
parser.add_argument("--num_fold", default=5, type=float, help="num fold")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--pin_memory", action="store_true", help="pin memory")
parser.add_argument("--workers", default=2, type=int, help="number of workers")

# transform
parser.add_argument("--num_samples", default=2, type=int, help="number of samples")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--rand_flipd_prob", default=0.1, type=float, help="RandFlipd aug probability")
parser.add_argument("--rand_rotate90d_prob", default=0.1, type=float, help="RandRotate90d aug probability")
parser.add_argument("--rand_shift_intensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")

# model
parser.add_argument("--model_name", default=None, type=str, help="model name")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")

# loss
parser.add_argument("--loss_name", default="dice_ce", type=str, help="optimization algorithm")

# optimizer
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")

# scheduler
parser.add_argument("--lrschedule", default=None, type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")

# infer
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--infer_overlap", default=0.25, type=float, help="sliding window inference overlap")


def main():
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    # make dir
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)

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
    loss = loss_func(args.loss_name)

    # optimizer
    optimizer = optim(model, args)

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
    best_acc = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        # load model
        model.load_state_dict(checkpoint["state_dict"])
        # load optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])
        # load lrschedule
        if args.lrschedule is not None:
          scheduler.load_state_dict(checkpoint['scheduler'])
        # load check point epoch and best acc
        if "epoch" in checkpoint and args.start_epoch == 0:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

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
            model=model,
            train_loader=tr_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_func=loss,
            acc_func=dice_acc,
            model_inferer=model_inferer,
            post_label=post_label,
            post_pred=post_pred,
            writer=writer,
            args=args,
        )
    else:
        tt_loader = loader[0]

        # test
        dc_vals, hd95_vals = run_testing(
            model,
            tt_loader,
            model_inferer,
            post_label,
            post_pred,
        )

        pids = get_pids_by_loader(tt_loader)

        lbl_cols = [
            'spleen',
            'right_kidney',
            'left_kidney',
            'gallbladder',
            'esophagus',
            'liver',
            'stomach',
            'aorta',
            'inferior_vena_cava',
            'portal_vein_and_splenic_vein',
            'pancreas',
            'right_adrenal_gland',
            'left_adrenal_gland',
        ]

        eval_tt_dice_val_df = pd.DataFrame(
            dc_vals,
            columns=[f'dice_{lbl_col}' for lbl_col in lbl_cols]
        )
        eval_tt_hd95_val_df = pd.DataFrame(
            hd95_vals,
            columns=[f'hd95_{lbl_col}' for lbl_col in lbl_cols]
        )
        eval_tt_df = pd.DataFrame({
            'patientId': pids,
            'type': 'test',
        })

        eval_tt_df = pd.concat([eval_tt_df, eval_tt_dice_val_df, eval_tt_hd95_val_df], axis=1, join='inner')\
            .reset_index(drop=True)
        eval_tt_df.to_csv(os.path.join(args.eval_dir, f'best_model_eval.csv'), index=False)

        print("\neval result:")
        print('avg dice: ', eval_tt_dice_val_df.T.mean().mean())
        print('avg hd95:', eval_tt_hd95_val_df.T.mean().mean())
        print(eval_tt_df.to_string())


if __name__ == "__main__":
    main()
