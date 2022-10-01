import os
import glob

from data_utils.nnunet_dataset import setup_nnunet_dataset
from data_utils.data_loader import MyDataLoader, split_data_dicts
from data_utils.io import load_json
from transforms.spleen_transform import (
    get_train_transform, 
    get_val_transform
)

def sort_func(x):
    return int(x.split('/')[-1].split('.')[0].split('_')[-1])


def get_data_dicts(data_dir):
    train_images = sorted(
        glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")),
        key=sort_func 
    )
    train_labels = sorted(
        glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")),
        key=sort_func
    )
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    return data_dicts


def get_loader(args):
    if args.data_dicts_json:
      data_dicts = load_json(args.data_dicts_json)
    else:
      data_dicts = get_data_dicts(args.data_dir)
    
    train_transform = get_train_transform(args)
    val_transform = get_val_transform(args)

    dl = MyDataLoader(
        data_dicts,
        train_transform,
        val_transform,
        args
    )

    return dl.get_loader()


def convert_to_nnunet_dataset(
        src_data_dir, 
        dst_data_dir,
        fold,
        split_train_ratio,
        num_fold
    ):
    '''convert dataset to nnunet dataset format'''
    os.makedirs(dst_data_dir, exist_ok=True)

    data_dicts = get_data_dicts(src_data_dir)

    # split data dicts to tr and tt
    tr_files, val_files, tt_files = split_data_dicts(data_dicts, fold, split_train_ratio, num_fold)
    tr_data_dicts = tr_files + val_files
    tt_data_dicts = tt_files

    # setup tr nnunet dataset
    setup_nnunet_dataset(tr_data_dicts, dst_data_dir, save_transform_lbl, test_mode=False)
    # setup tt nnunet dataset
    setup_nnunet_dataset(tt_data_dicts, dst_data_dir, save_transform_lbl, test_mode=True)


