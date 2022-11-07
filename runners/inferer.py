import os
from pathlib import PurePath

import torch
from monai.data import decollate_batch

from data_utils.io import save_img
import matplotlib.pyplot as plt


def infer(model, data, model_inferer, device):
    model.eval()
    with torch.no_grad():
        output = model_inferer(data['image'].to(device))
        output = torch.argmax(output, dim=1)
    return output


def get_filename(data):
    return PurePath(data['image_meta_dict']['filename_or_obj']).parts[-1]


def run_infering(
        model,
        data,
        model_inferer,
        post_transform,
        args
    ):
    
    data['pred'] = infer(model, data, model_inferer, args.device)
    data = post_transform(data)
    
    filename = get_filename(data)
    infer_img_pth = os.path.join(args.infer_dir, filename)
        
    save_img(
      data['pred'], 
      data['pred_meta_dict'], 
      infer_img_pth
    )
