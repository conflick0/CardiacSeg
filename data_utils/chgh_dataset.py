import os

from monai.transforms import AddChannel

from data_utils.data_loader import MyDataLoader
from data_utils.io import load_json
from transforms.chgh_transform import CHGHTransform


def get_data_dicts(data_dir):
    patient_dirs = sorted(os.listdir(data_dir), key=lambda x: int(x.split('_')[-1]))
    data_dicts = []
    for patient_dir in patient_dirs:
        data_dicts.append({
            "image": os.path.join(os.path.join(data_dir, patient_dir, f'{patient_dir}.nii.gz')),
            "label": os.path.join(os.path.join(data_dir, patient_dir, f'GT.nii.gz'))
        })
    return data_dicts



def get_loader(args):
    tf = CHGHTransform(args)

    if args.data_dicts_json:
      data_dicts = load_json(args.data_dicts_json)
    else:
      data_dicts = get_data_dicts(args.data_dir)
    
    train_transform = tf.get_train_transform()
    val_transform = tf.get_val_transform()

    dl = MyDataLoader(
        data_dicts,
        train_transform,
        val_transform,
        args
    )

    return dl.get_loader()


def get_infer_data(data_dict, keys, args):
    tf = CHGHTransform(args)
    inf_transform = tf.get_inf_transform(keys)
    data = inf_transform(data_dict)
    return data

