import os


def get_data_dicts(data_dir, pid_dirs):
    data_dicts = []
    for pid_dir in pid_dirs:
        pid_img_dir = os.path.join(data_dir, pid_dir, 'image')
        pid_lbl_dir = os.path.join(data_dir, pid_dir, 'label')
        
        # get sigle patient file names
        files = sorted(os.listdir(pid_img_dir), key=lambda x: int(x.split('.')[0]))

        # make sigle patient data dicts
        p_data_dicts = []
        for file in files:
            p_data_dicts.append(
                {
                    'image': os.path.join(pid_img_dir, file),
                    'label': os.path.join(pid_lbl_dir, file)
                }
            )
        
        data_dicts.extend(p_data_dicts)
    return data_dicts
