import sys
# set package path
sys.path.append("/nfs/Workspace/CardiacSeg")

import os
from functools import partial

import torch

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    Orientationd,
    ToNumpyd,
)
from monailabel.transform.post import Restored

from data_utils.data_loader_utils import load_data_dict_json
from data_utils.dataset import get_infer_data
from data_utils.io import load_json
from runners.inferer import run_infering
from networks.network import network

from expers.args import get_parser


def main():
    args = get_parser(sys.argv[1:])
    main_worker(args)
    
    
def is_deep_sup(checkpoint):
    for key in list(checkpoint["state_dict"].keys()):
        if 'ds' in key:
            return True
    return False

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
        
        if is_deep_sup(checkpoint) and args.model_name != 'cotr':
            # load check point epoch and best acc
            print("Tag 'ds (deeply supervised)' found in state dict - fixing!")
            for key in list(checkpoint["state_dict"].keys()):
                if 'ds' in key:
                    checkpoint["state_dict"].pop(key) 
        
        # load model
        model.load_state_dict(checkpoint["state_dict"])
        
        print(
          "=> loaded checkpoint '{}')"\
          .format(args.checkpoint)
        )

    # inferer
    keys = ['pred']
    if args.data_name == 'mmwhs':
        axcodes = 'LAS'
    else:
        axcodes = 'LPS'
    # axcodes = 'RAS'
    post_transform = Compose([
        Orientationd(keys=keys, axcodes=axcodes),
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
    if args.data_dicts_json and args.data_name != 'mmwhs':
        data_dicts = load_data_dict_json(args.data_dir, args.data_dicts_json)
    elif args.data_dicts_json and args.data_name == 'mmwhs':
        data_dicts = load_json(args.data_dicts_json)
    else:
        if args.lbl_pth is not None:
            data_dicts = [{
                'image': args.img_pth,
                'label': args.lbl_pth
            }]
        else:
            data_dicts = [{
                'image': args.img_pth,
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