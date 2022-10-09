import os

from data_utils.nnunet_dataset import setup_nnunet_dataset
from data_utils.data_loader import MyDataLoader, split_data_dicts
from data_utils.io import load_json
from transforms.segthor_transform import (
    get_train_transform, 
    get_val_transform, 
    save_transform_lbl
)

def get_data_dicts(data_dir):
    data_dicts = []
    for patient_dir in sorted(os.listdir(data_dir)):
        data_dicts.append({
            "image": os.path.join(os.path.join(data_dir, patient_dir, f'{patient_dir}.nii.gz')),
            "label": os.path.join(os.path.join(data_dir, patient_dir, f'GT.nii.gz'))
        })
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
