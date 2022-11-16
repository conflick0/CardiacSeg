import os

from monai.transforms import AddChannel

from data_utils.data_loader import get_dl, MyDataLoader2d, split_data_dicts
from data_utils.io import load_json
from data_utils.preprocess import generate_data_dicts_json_2d
from transforms.chgh_transform_2d import (
    get_train_transform,
    get_val_transform,
    get_inf_transform
)


def get_data_dicts_2d(data_dir, pid_dirs):
    data_dicts = []
    for pid_dir in pid_dirs:
        pid_img_dir = os.path.join(data_dir, pid_dir, 'image')
        pid_lbl_dir = os.path.join(data_dir, pid_dir, 'label')
        
        # get sigle patient file names
        files = sorted(os.listdir(pid_img_dir), key=lambda x: int(x.split('.')[0]))

        # make sigle patient data dicts
        p_data_dicts = []
        for file in files:
            p_data_dicts.append(
                {
                    'image': os.path.join(pid_img_dir, file),
                    'label': os.path.join(pid_lbl_dir, file)
                }
            )
        
        data_dicts.extend(p_data_dicts)
    return data_dicts


def get_data_dicts(data_dir, pid_dirs):
    data_dicts = []
    for pid_dir in pid_dirs:
        data_dicts.append({
            "image": os.path.join(data_dir, pid_dir, f'{pid_dir}.nii.gz'),
            "label": os.path.join(data_dir, pid_dir, f'{pid_dir}_gt.nii.gz')
        })
    return data_dicts


def gen_data_dicts_json_2d(
        data_dir, 
        data_dir_2d,
        file_path, 
        fold, 
        num_fold, 
        train_data_ratio, 
        pid_dirs=None
    ):
    generate_data_dicts_json_2d(
        data_dir, 
        data_dir_2d,
        get_data_dicts,
        get_data_dicts_2d,
        file_path, 
        fold, 
        num_fold, 
        train_data_ratio, 
        pid_dirs
    )


def get_loader(args):
    data_dicts = load_json(args.data_dicts_json)

    train_transform = get_train_transform(args)
    val_transform = get_val_transform(args)

    dl = MyDataLoader2d(
        data_dicts,
        train_transform,
        val_transform,
        args
    )
    return dl.get_loader()


def get_infer_data(data_dict, args):
    keys = data_dict.keys()
    inf_transform = get_inf_transform(keys, args)
    data = inf_transform(data_dict)
    return data


def get_infer_loader(keys, args):
    data_dicts = [{'image': args.img_pth, 'label': args.lbl_pth}]
    inf_transform = get_inf_transform(keys, args)
    inf_loader = get_dl(
        files=data_dicts,
        transform=inf_transform,
        shuffle=False,
        args=args
    )
    return inf_loader
