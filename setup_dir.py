import os

# make dataset dir
root_data_dir = 'dataset'
data_names = ['chgh', 'mmwhs']
for d in data_names:
    data_dir = os.path.join(root_data_dir, d)
    os.makedirs(data_dir, exist_ok=True)
    print('mkdir data:', data_dir)


# make model dir
exp_dir = os.path.join('exps', 'exps')
model_names = ['unetcnx_a1', 'swinunetr', 'unetr', 'cotr', 'attention_unet', 'unet3d']
for m in model_names:
    for d in data_names:
        # make model dir
        model_exp_dir = os.path.join(exp_dir, m, d, 'tune_results')
        os.makedirs(model_exp_dir, exist_ok=True)
        print('mkdir exp:', model_exp_dir)
        
        # make infer dir
        if d == 'mmwhs':
            task_id = 'exp_2_tn2'
        else:
            task_id = 't_5' if m == 'unetcnx_a1' else 't_4'
        model_exp_dir = os.path.join(exp_dir, m, d, 'infers', task_id)
        os.makedirs(model_exp_dir, exist_ok=True)
        print('mkdir exp:', model_exp_dir)
        
        
