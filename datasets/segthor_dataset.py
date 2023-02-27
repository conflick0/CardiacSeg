import os


def get_data_dicts(data_dir):
    data_dicts = []
    for patient_dir in sorted(os.listdir(data_dir)):
        data_dicts.append({
            "image": os.path.join(os.path.join(data_dir, patient_dir, f'{patient_dir}.nii.gz')),
            "label": os.path.join(os.path.join(data_dir, patient_dir, f'GT.nii.gz'))
        })
    return data_dicts

