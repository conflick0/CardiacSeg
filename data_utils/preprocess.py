import os
from pathlib import PurePath
import math

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import KFold

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


def generate_data_dicts_json_2d(
        data_dir, 
        data_dir_2d,
        get_data_dicts,
        get_data_dicts_2d,
        file_path, 
        fold, 
        num_fold, 
        train_data_ratio, 
        pid_dirs=None
    ):

    if pid_dirs == None:
        pid_dirs = sorted(os.listdir(data_dir), key=lambda x: int(x.split('_')[-1]))

    # split data
    train_data, val_data, test_data = split_data(pid_dirs, fold, num_fold, train_data_ratio)
    print(f'train data: {train_data}')
    print(f'val data: {val_data}')
    print(f'test data: {test_data}')
    data_dicts = {}
    data_dicts['training'] = get_data_dicts_2d(data_dir_2d, train_data)
    data_dicts['validation'] = get_data_dicts(data_dir, val_data)
    data_dicts['test'] = get_data_dicts(data_dir, test_data)

    save_json(data_dicts, file_path)


def get_data_by_idx(data, idxs):
    return [data[i] for i in idxs]


def split_data_by_ratio(data, train_data_ratio):
    num_data = len(data)
    num_train_data = math.ceil(num_data * train_data_ratio)

    train_idxs = list(range(0, num_train_data))
    test_idxs = list(range(num_train_data, num_data))
    
    train_data = get_data_by_idx(data, train_idxs)
    test_data = get_data_by_idx(data, test_idxs)
    return train_data, test_data


def split_data_by_k_fold(data, fold, num_fold):
    num_data = len(data)

    data_idx = [i for i, _ in enumerate(data)]

    kf = KFold(n_splits=num_fold)
    flods = list(kf.split(data_idx))
    train_idxs, test_idxs = flods[fold]

    train_data = get_data_by_idx(data, train_idxs)
    test_data = get_data_by_idx(data, test_idxs)

    return train_data, test_data


def split_data(data, fold, num_fold, train_data_ratio):
    train_val_data, test_data = split_data_by_ratio(data, train_data_ratio)
    train_data, val_data = split_data_by_k_fold(train_val_data, fold, num_fold)
    return train_data, val_data, test_data

