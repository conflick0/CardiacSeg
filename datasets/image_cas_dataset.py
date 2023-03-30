import os


def get_data_dicts(data_dir):
    patient_dirs = sorted(os.listdir(data_dir), key=lambda x: int(x.split('_')[-1]))
    data_dicts = []
    for patient_dir in patient_dirs:
        data_dicts.append({
            "image": os.path.join(os.path.join(data_dir, patient_dir, f'img.nii.gz')),
            "label": os.path.join(os.path.join(data_dir, patient_dir, f'label.nii.gz'))
        })
    return data_dicts
