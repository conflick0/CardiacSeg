import os


def get_data_dicts(data_dir):
    img_dir = os.path.join(data_dir, 'img')
    lbl_dir = os.path.join(data_dir, 'label')
    img_fns = sorted(os.listdir(img_dir))
    lbl_fns = sorted(os.listdir(lbl_dir))
    img_pths = list(map(lambda fn: os.path.join(img_dir, fn), img_fns))
    lbl_pths = list(map(lambda fn: os.path.join(lbl_dir, fn), lbl_fns))
    data_dicts = []
    for img_pth, lbl_pth in zip(img_pths, lbl_pths):
        data_dicts.append({
            "image": img_pth,
            "label": lbl_pth
        })
    return data_dicts

