import argparse
import os
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from data_utils.segthor import get_loader
from trainers.trainer import run_training
from networks.unetcnx import UNETCNX

parser = argparse.ArgumentParser(description="model segmentation pipeline")
# mode
parser.add_argument("--test_mode", action="store_false", help="test mode")

# dir and path
parser.add_argument("--root_dir", default="./", type=str, help="root directory")
parser.add_argument("--data_dir", default="", type=str, help="dataset directory")
parser.add_argument("--model_dir", default="test", type=str, help="directory to save the models")
parser.add_argument("--log_dir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--eval_dir", default="test", type=str, help="directory to save the eval result")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")

# train loop
parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
parser.add_argument("--val_every", default=100, type=int, help="validation frequency")
parser.add_argument("--max_epochs", default=5000, type=int, help="max number of training epochs")

# data
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--pin_memory", action="store_true", help="pin memory")
parser.add_argument("--workers", default=2, type=int, help="number of workers")

# model
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")

# optimizer
parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")

# scheduler
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")

# infer
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")


def main():
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    # device
    if torch.cuda.is_available():
        print("cuda is available")
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        print("cuda is not available")
        device = torch.device("cpu")

    # data loader
    loader = get_loader(args)

    # model
    model = UNETCNX(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=48,
        patch_size=4
    ).to(device)

    # check point
    start_epoch = 0
    best_acc = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    # loss
    dice_loss = DiceCELoss(
        to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
    )

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)

    # inferer
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[96, 96, 96],
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    # writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # training
    run_training(
        start_epoch=start_epoch,
        best_acc=best_acc,
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        model_inferer=model_inferer,
        post_label=post_label,
        post_pred=post_pred,
        writer=writer,
        args=args,
    )


if __name__ == "__main__":
    main()