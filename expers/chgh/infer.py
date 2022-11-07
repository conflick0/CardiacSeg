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
from monai.transforms import (
    AsDiscrete,
    Compose,
    Orientationd,
    ToNumpyd,
    SqueezeDimd,
    AddChanneld
)
from monailabel.transform.post import BoundingBoxd, Restored

from data_utils.chgh_dataset import get_infer_data
from data_utils.io import load_json
from data_utils.utils import get_pid_by_data
from runners.inferer import run_infering
from networks.network import network


parser = argparse.ArgumentParser(description="model segmentation pipeline")

# dir and path
parser.add_argument("--data_dir", default="", type=str, help="dataset directory")
parser.add_argument("--model_dir", default="models", type=str, help="directory to save the models")
parser.add_argument("--log_dir", default="logs", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--eval_dir", default="evals", type=str, help="directory to save the eval result")
parser.add_argument("--infer_dir", default="infers", type=str, help="directory to save the infer result")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--img_pth", default=None, help="target img for infer")
parser.add_argument("--lbl_pth", default=None, help="target lbl for infer")

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

# infer
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--infer_overlap", default=0.25, type=float, help="sliding window inference overlap")


def main():
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    # make dir
    os.makedirs(args.infer_dir, exist_ok=True)

    # device
    if torch.cuda.is_available():
        print("cuda is available")
        args.device = torch.device("cuda")
    else:
        print("cuda is not available")
        args.device = torch.device("cpu")

    # model
    model = network(args.model_name, args)

    # check point
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        # load model
        model.load_state_dict(checkpoint["state_dict"])
        # load check point epoch and best acc
                
        print(
          "=> loaded checkpoint '{}')"\
          .format(args.checkpoint)
        )

    # inferer
    post_transform = Compose([
        Orientationd(keys=["pred"], axcodes="LPS"),
        ToNumpyd(keys="pred"),
        Restored(keys=["pred"], ref_image="image"),
        BoundingBoxd(keys="pred", result="result", bbox="bbox")
    ])
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    # prepare data_dict
    if args.data_dicts_json:
      data_dicts = load_json(args.data_dicts_json)
    else:
      data_dicts = [{
        'image': args.img_pth,
        'label': args.lbl_pth
      }]

    # run infer
    for data_dict in data_dicts:
      print('infer data:', data_dict)
      
      # load infer data
      data = get_infer_data(data_dict, args)

      # infer
      run_infering(
          model,
          data,
          model_inferer,
          post_transform,
          args
      )


if __name__ == "__main__":
    main()
