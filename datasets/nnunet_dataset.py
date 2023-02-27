import os
import shutil
from pathlib import PurePath
from tqdm import tqdm

from data_utils.data_loader_utils import split_data_dicts


def get_nnunet_data_dicts(data_dir):
    img_dir = os.path.join(data_dir, 'imagesTr')
    lbl_dir = os.path.join(data_dir, 'labelsTr')

    data_dicts = []
    for img_fn, lbl_fn in zip(sorted(os.listdir(img_dir)), sorted(os.listdir(lbl_dir))):
        data_dicts.append({
            "image": os.path.join(os.path.join(img_dir, img_fn)),
            "label": os.path.join(os.path.join(lbl_dir, lbl_fn))
        })
    return data_dicts


def setup_nnunet_dataset(data_dicts, dst_data_dir, save_transform_lbl, test_mode):
    '''convert data to nnunet dataset format'''

    # setup sub dir path
    if test_mode:
      dst_data_img_dir = os.path.join(dst_data_dir, 'imagesTs')
      if len(data_dicts[0]) == 2:
        dst_data_lbl_dir = os.path.join(dst_data_dir, 'labelsTs')
    else:
      dst_data_img_dir = os.path.join(dst_data_dir, 'imagesTr')
      dst_data_lbl_dir = os.path.join(dst_data_dir, 'labelsTr')

    # make dir
    os.makedirs(dst_data_dir, exist_ok=True)
    os.makedirs(dst_data_img_dir, exist_ok=True)
    os.makedirs(dst_data_lbl_dir, exist_ok=True)

    for data_dict in tqdm(data_dicts):
        img_pth, lbl_pth = data_dict.values()

        # setup filename
        pid = PurePath(img_pth).parts[-1].split('.')[0]

        # setup dst img and lbl path
        dst_img_pth = os.path.join(dst_data_img_dir, f'{pid}_0000.nii.gz')
        dst_lbl_pth = os.path.join(dst_data_lbl_dir, f'{pid}.nii.gz')

        shutil.copy(img_pth, dst_img_pth)
        save_transform_lbl(data_dict, dst_lbl_pth)


def convert_to_nnunet_dataset(
        src_data_dir,
        dst_data_dir,
        split_train_ratio,
        num_fold,
        fold,
        save_transform_lbl_fn,
        get_data_dicts_fn
    ):
    '''convert dataset to nnunet dataset format'''
    os.makedirs(dst_data_dir, exist_ok=True)

    data_dicts = get_data_dicts_fn(src_data_dir)

    # split data dicts to tr and tt
    tr_files, val_files, tt_files = split_data_dicts(data_dicts, fold, split_train_ratio, num_fold)
    tr_data_dicts = tr_files + val_files
    tt_data_dicts = tt_files

    # setup tr nnunet dataset
    setup_nnunet_dataset(tr_data_dicts, dst_data_dir, save_transform_lbl_fn, test_mode=False)
    # setup tt nnunet dataset
    setup_nnunet_dataset(tt_data_dicts, dst_data_dir, save_transform_lbl_fn, test_mode=True)
