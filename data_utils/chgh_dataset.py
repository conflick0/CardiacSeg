import os

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
