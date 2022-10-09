import os

from monai.transforms import AddChannel

from data_utils.data_loader import MyDataLoader, split_data_dicts
from data_utils.io import load_json, save_json
from transforms.chgh_transform import get_train_transform, get_val_transform, get_inf_transform


def get_loader(args):
    data_dicts = load_json(args.data_dicts_json)
    
    train_transform = get_train_transform(args)
    val_transform = get_val_transform(args)

    dl = MyDataLoader(
        data_dicts,
        train_transform,
        val_transform,
        args
    )

    return dl.get_loader()


def generate_dataset_json(
        data_dicts,
        data_json_pth,
        dataset_name,
        labels,
        tensorImageSize='3D',
        modalities=['CT'],
        license='see challenge website',
        split_train_ratio=None,
        fold=None,
        num_fold=None,
        sort_keys=True,
    ):
    '''generate dataset json, if split_train_ratio is None, only has training path'''

    if split_train_ratio is None:
        tr_files = data_dicts
        tt_files = []
    else:
        # split data dicts to tr and tt
        tr_files, val_files, tt_files = split_data_dicts(data_dicts, fold, split_train_ratio, num_fold)

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['tensorImageSize'] = tensorImageSize
    json_dict['licence'] = license
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(tr_files)
    json_dict['numTest'] = len(tt_files)
    json_dict['training'] = tr_files
    json_dict['validation'] = val_files
    json_dict['test'] = tt_files
    if not data_json_pth.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(data_json_pth), sort_keys=sort_keys)




