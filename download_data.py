import os
import toml

# config file
config_pth = 'config.toml'

# base dir
root_data_dir = 'dataset'
exp_dir = os.path.join('exps', 'exps')

# data and model names
data_names = ['chgh', 'mmwhs']
model_names = ['unetcnx_a1', 'swinunetr', 'unetr', 'cotr', 'attention_unet', 'unet3d']

# model zip file name
file_names = {
    'chgh': ['t_5', 't_4', 't_4', 't_4', 't_4', 't_4'],
    'mmwhs': ['exp_2_tn2', 'exp_2_tn2', 'exp_2_tn2', 'exp_2_tn2', 'exp_2_tn2', 'exp_2_tn2'],
}

# data zip file name
sub_data_dirs = {
    'chgh': 'dataset_2',
    'mmwhs': 'ct_train',
}


# downlod file func, cd into dir, gdown by id, unzip file
def download_file(dir, file_name, id):
    os.system(f'''
    cd {dir} && gdown {id} && unzip {file_name}.zip
    ''')


# read config
with open(config_pth, 'r') as f:
    config = toml.load(f)


    
# download data
for d  in data_names:
    data_dir = os.path.join(root_data_dir, d)
    target_dir =  os.path.join(data_dir, sub_data_dirs[d])
    if os.path.exists(target_dir):
        print('dir exist:', target_dir)
    else:
        print('download:', target_dir)
        download_file(data_dir, sub_data_dirs[d], id=config['dataset'][d])


# download models    
for d in data_names:
    for m, fn in zip(model_names, file_names[d]):
        model_exp_dir = os.path.join(exp_dir, m, d, 'tune_results')
        target_dir =  os.path.join(model_exp_dir, fn)
        if os.path.exists(target_dir):
            print('dir exist:', target_dir)
        else:
            print('download:', target_dir)
            download_file(model_exp_dir, fn, id=config['model'][d][m])
            
            

        