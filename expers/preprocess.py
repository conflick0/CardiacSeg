import os
import importlib
from pathlib import PurePath
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

from data_utils.data_loader_utils import split_data_dicts
from data_utils.io import save_json


def build_2d_dataset(data_dicts, dst_data_dir, transform):
    '''
    convert 3d dataset to 2d slices dataset
    '''
    # make dst data dir
    os.makedirs(dst_data_dir, exist_ok=True)
    
    for data_dict in data_dicts:
        # load img and lbl
        data = transform(data_dict)

        # get pid
        pid = PurePath(data['image_meta_dict']['filename_or_obj']).parts[-1].split('.')[0]
        print(pid)

        # make pid dir
        pid_dir = os.path.join(dst_data_dir, pid)
        os.makedirs(pid_dir, exist_ok=True)

        # make img and lbl dir
        img_dir = os.path.join(pid_dir, 'image')
        lbl_dir = os.path.join(pid_dir, 'label')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        # save img and lbl slices
        slices = data['image'].shape[-1]
        for i, s in tqdm(enumerate(range(slices))):
            img = Image.fromarray(data['image'][0,:,:,s].numpy().astype(np.uint8), 'L')
            lbl = Image.fromarray(data['label'][0,:,:,s].numpy().astype(np.uint8), 'L')
            filename = f'{i}.bmp'
            img.save(os.path.join(img_dir, filename))
            lbl.save(os.path.join(lbl_dir, filename))


def get_rel_pth(pth):
    return os.path.join(*PurePath(pth).parts[-2:])


def get_rel_data_dicts(data_dicts):
    out_data_dicts = []
    for data_dict in data_dicts:
        out_data_dict = {}
        for k in data_dict.keys():
            out_data_dict[k] = get_rel_pth(data_dict[k])
        out_data_dicts.append(out_data_dict)
    return out_data_dicts


def build_data_dicts_json(
        get_data_dicts_fn,
        src_data_dir,
        dst_data_json,
        split_train_ratio,
        num_fold,
        fold
    ):

    data_dicts = get_data_dicts_fn(src_data_dir)
    train_files, val_files, test_files = split_data_dicts(
        data_dicts,
        fold,
        split_train_ratio,
        num_fold
    )
    out_data_dicts = {
        'train': get_rel_data_dicts(train_files),
        'val': get_rel_data_dicts(val_files),
        'test': get_rel_data_dicts(test_files)
    }
    save_json(out_data_dicts, dst_data_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="preprocess")
    
    parser.add_argument("--data_name", default=None, type=str, help="data dicts json")
    parser.add_argument("--data_dir", default=None, type=str, help="data dicts json")
    parser.add_argument("--dst_data_json", default=None, type=str, help="data dicts json")
    
    parser.add_argument("--split_train_ratio", default=0.9, type=float, help="split train ratio")
    parser.add_argument("--num_fold", default=5, type=int, help="num fold")
    parser.add_argument("--fold", default=4, type=int, help="index of fold")
    
    args = parser.parse_args()
    
    # equal to from datasets.xxx_dataset import get_data_dicts
    dataset = importlib.import_module(f'datasets.{args.data_name}_dataset')
    get_data_dicts = getattr(dataset, 'get_data_dicts', None)

    build_data_dicts_json(
        get_data_dicts,
        src_data_dir=args.data_dir,
        dst_data_json=args.dst_data_json,
        split_train_ratio=args.split_train_ratio,
        num_fold=args.num_fold,
        fold=args.fold,
    )

