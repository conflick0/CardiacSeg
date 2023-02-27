import os
from pathlib import PurePath

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
    from datasets.chgh_dataset import get_data_dicts
    data_dir = r'/nfs/Workspace/dataset'
    dst_data_json = os.path.join(data_dir, 'data.json')

    build_data_dicts_json(
        get_data_dicts,
        src_data_dir=data_dir,
        dst_data_json=dst_data_json,
        split_train_ratio=0.7,
        num_fold=3,
        fold=2,
    )

