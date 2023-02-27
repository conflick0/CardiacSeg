import os
import glob


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
