import sys

# set package path
sys.path.append("/nfs/Workspace/CardiacSeg")

import argparse
import os
from pathlib import PurePath
from functools import partial

import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter

from monai.metrics import DiceMetric, HausdorffDistanceMetric

from monai.transforms import (
    AsDiscrete,
    Compose,
    Orientationd,
    ToNumpyd,
    SqueezeDimd,
    AddChanneld
)
from monailabel.transform.post import BoundingBoxd, Restored

from expers.args import get_parser
from data_utils.chgh_dataset import get_infer_data, get_multiple_label_data_dicts, multi_label_to_label_pred_data_dicts
from data_utils.io import load_json
from data_utils.utils import get_pid_by_data
from runners.inferer import eval_label_pred


def main_worker(args):
    if torch.cuda.is_available():
        print("cuda is available")
        args.device = torch.device("cuda")
    else:
        print("cuda is not available")
        args.device = torch.device("cpu")
    
    # cls num
    cls_num = args.out_channels
    
    # prepare data_dict
    if args.data_dicts_json:
        data_dicts = load_json(args.data_dicts_json)
    elif args.data_dir:
        data_dicts = get_multiple_label_data_dicts(args.data_dir)
        data_dicts = multi_label_to_label_pred_data_dicts(data_dicts)
    else:
        data_dicts = [{
            'image': args.img_pth,
            'label': args.lbl_pth,
            'pred': args.pred_pth,
        }]

    # run infer
    for data_dict in data_dicts:
        print('infer data:', data_dict)
        
      
          # load infer data
        data = get_infer_data(data_dict, args)

        # eval
        dc_vals, hd95_vals = eval_label_pred(
            data,
            args.out_channels,
            args.device
        )
        
        print('pred name:',  PurePath(data_dict['pred']).parts[-1].split('_')[-1].split('.')[0])
        print('dice:', dc_vals)
        print('hd95:', hd95_vals)
        print()


if __name__ == "__main__":
    args = get_parser(sys.argv[1:])
    main_worker(args)