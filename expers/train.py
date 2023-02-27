import sys

import os
from functools import partial

import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from expers.args import get_parser
from data_utils.dataset import DataLoader, get_label_names
from data_utils.utils import get_pids_by_loader
from runners.trainer import run_training
from runners.tester import run_testing
from networks.network import network
from optimizers.optimizer import Optimizer, LR_Scheduler


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
    loader = DataLoader(args.data_name, args)()

    # model
    model = network(args.model_name, args)

    # loss
    dice_loss = DiceCELoss(to_onehot_y=True, softmax=True)

    # optimizer
    print(f'optimzer: {args.optim}')
    optimizer = Optimizer(args.optim, model.parameters(), args)

    # lrschedule
    if args.lrschedule is not None:
        print(f'lrschedule: {args.lrschedule}')
        scheduler = LR_Scheduler(args.lrschedule, optimizer, args)
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
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        if "early_stop_count" in checkpoint:
            early_stop_count = checkpoint["early_stop_count"]
        print(
            "=> loaded checkpoint '{}' (epoch {}) (bestacc {}) (early stop count {})" \
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

    else:
        os.makedirs(args.eval_dir, exist_ok=True)

        tt_loader = loader[0]

        # test
        dc_vals, hd95_vals = run_testing(
            model,
            tt_loader,
            model_inferer,
            post_label,
            post_pred
        )

        pids = get_pids_by_loader(tt_loader)

        label_names = get_label_names(args.data_name)

        eval_tt_dice_val_df = pd.DataFrame(
            dc_vals,
            columns=[f'dice{n}' for n in label_names]
        )
        eval_tt_hd95_val_df = pd.DataFrame(
            hd95_vals,
            columns=[f'hd95{n}' for n in label_names]
        )
        eval_tt_df = pd.DataFrame({
            'patientId': pids,
            'type': 'test',
        })

        eval_tt_df = pd.concat([eval_tt_df, eval_tt_dice_val_df, eval_tt_hd95_val_df], axis=1, join='inner') \
            .reset_index(drop=True)
        eval_tt_df.to_csv(os.path.join(args.eval_dir, f'best_model_eval.csv'), index=False)

        avg_dice = eval_tt_dice_val_df.T.mean().mean()
        avg_hd95 = eval_tt_hd95_val_df.T.mean().mean()

        print("\neval result:")
        print('avg dice:', avg_dice)
        print('avg hd95:', avg_hd95)
        print(eval_tt_df.to_string())


if __name__ == "__main__":
    args = get_parser(sys.argv[1:])

    # for LinearWarmupCosineAnnealingLR
    args.max_epochs = args.max_epoch

    main_worker(args)