import os
import glob
from pathlib import PurePath


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
            lbl_dict[lbl_cls] = f_pth
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
