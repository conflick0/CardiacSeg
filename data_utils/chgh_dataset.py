import os
import glob
from pathlib import PurePath

from monai.transforms import AddChannel

from data_utils.data_loader import MyDataLoader, get_dl
from data_utils.io import load_json
from transforms.chgh_transform import get_train_transform, get_val_transform, get_inf_transform


def get_data_dicts(data_dir):
    patient_dirs = sorted(os.listdir(data_dir), key=lambda x: int(x.split('_')[-1]))
    data_dicts = []
    for patient_dir in patient_dirs:
        data_dicts.append({
            "image": os.path.join(os.path.join(data_dir, patient_dir, f'{patient_dir}.nii.gz')),
            "label": os.path.join(os.path.join(data_dir, patient_dir, f'{patient_dir}_gt.nii.gz'))
        })
    return data_dicts


def get_multiple_label_data_dicts(data_dir):
    '''
    use '_' split out key as label, if split out key is none as image
    map image and label as data dict
    
    ex. 
    dir/
    -- pid_1000.nii.gz
    -- pid_1000_gt.nii.gz
    -- pid_1000_th.nii.gz
    
    return [
        {
            'image': 'dir/pid_1000.nii.gz', 
            'label': 'dir/pid_1000_gt.nii.gz'
            'label_cls': 'gt'
        },
        {
            'image': 'dir/pid_1000.nii.gz', 
            'label': 'dir/pid_1000_th.nii.gz',
            'label_cls': 'th'
        }
    ]
    '''
    files = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))
    
    img_pth = ''
    lbl_dict = {}
    for f_pth in files:
        file_name_parts = PurePath(f_pth).parts[-1].split('_')
        if len(file_name_parts) > 2:
            lbl_cls = file_name_parts[-1].split('.')[0]
            lbl_dict[lbl_cls]= f_pth
        else:
            img_pth = f_pth
    
    data_dicts = []
    for lbl_cls, lbl_pth in lbl_dict.items():
        data_dicts.append({
            "image": img_pth,
            "label": lbl_pth,
            "label_cls": lbl_cls 
        })
    
    return data_dicts


def multi_label_to_label_pred_data_dicts(data_dicts):
    '''
    ex.
    data_dicts = [
        {
            'image': 'dir/pid_1000.nii.gz', 
            'label': 'dir/pid_1000_gt.nii.gz'
            'label_cls': 'gt'
        },
        {
            'image': 'dir/pid_1000.nii.gz', 
            'label': 'dir/pid_1000_th.nii.gz',
            'label_cls': 'th'
        }
    ]
    
    return [
        {
            'image': 'dir/pid_1000.nii.gz', 
            'label': 'dir/pid_1000_gt.nii.gz'
            'pred':'dir/pid_1000_th.nii.gz',
        }
    ]
    '''
    out_data_dicts = []
    
    # find gt lbl pth
    gt_lbl_pth = ''
    for d in data_dicts:
        if d['label_cls'] == 'gt':
            gt_lbl_pth = d['label']
    
    for d in data_dicts:
        if d['label_cls'] != 'gt':
            out_data_dicts.append({
                'image': d['image'],
                'label': gt_lbl_pth,
                'pred': d['label']
            })
    
    return out_data_dicts


def get_loader(args):
    if args.data_dicts_json:
      data_dicts = load_json(args.data_dicts_json)
      data_dicts = [
        {
          "image": os.path.join(args.data_dir, data_dict['image']),
          "label": os.path.join(args.data_dir, data_dict['label'])
        } for data_dict in data_dicts
      ]
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
