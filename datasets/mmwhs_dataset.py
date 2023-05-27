import os


def get_data_dicts(data_dir):
    img_fns = sorted(os.listdir(data_dir))[::2]
    lbl_fns = sorted(os.listdir(data_dir))[1::2]
    img_pths = list(map(lambda fn: os.path.join(data_dir, fn), img_fns))
    lbl_pths = list(map(lambda fn: os.path.join(data_dir, fn), lbl_fns))
    data_dicts = []
    for img_pth, lbl_pth in zip(img_pths, lbl_pths):
        data_dicts.append({
            "image": img_pth,
            "label": lbl_pth
        })
    return data_dicts


def get_inf_data_dicts(data_dir):
    img_fns = sorted(os.listdir(data_dir))
    img_pths = list(map(lambda fn: os.path.join(data_dir, fn), img_fns))
    data_dicts = []
    for img_pth in img_pths:
        data_dicts.append({
            "image": img_pth
        })
    return data_dicts
