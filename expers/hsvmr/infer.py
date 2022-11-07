import sys

# set package path
sys.path.append("/content/drive/MyDrive/CardiacSeg")

import argparse
import os
from functools import partial

import torch

from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    Orientationd,
    ToNumpyd,
    SqueezeDimd,
    AddChanneld
)
from monailabel.transform.post import BoundingBoxd, Restored
from data_utils import hsvmr_dataset, chgh_dataset
from runners.inferer import run_infering
from networks.network import network
from expers.hsvmr.config import get_args


def main():
    args = get_args()
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

    # load infer data
    if args.data_dir.lower().find('hsvmr'):
      loader = hsvmr_dataset.get_infer_loader(args)
    else:
      keys = ['image', 'label']
      loader = chgh_dataset.get_infer_loader(keys, args)

    # model
    model = network(args.model_name, args)
    
    # check point
    early_stop_count = args.early_stop_count
    start_epoch = args.start_epoch
    best_acc = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        # load model
        model.load_state_dict(checkpoint["state_dict"])
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
    post_transform = Compose([
        Orientationd(keys=["pred"], axcodes="LPI"),
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

    # test
    run_infering(
        model,
        loader,
        model_inferer,
        post_transform,
        args.device,
        args
    )


if __name__ == "__main__":
    main()
