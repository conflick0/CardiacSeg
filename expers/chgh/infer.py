import sys

# set package path
sys.path.append("/nfs/Workspace/CardiacSeg")

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

from expers.args import get_parser


def main():
    args = get_parser(sys.argv[1:])
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
    keys = ['pred']
    post_transform = Compose([
        Orientationd(keys=keys, axcodes="LPS"),
        ToNumpyd(keys=keys),
        Restored(keys=keys, ref_image="image")
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
