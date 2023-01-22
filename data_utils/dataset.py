import importlib


def DataLoader(data_name, args):
    import data_utils as du
    module = importlib.import_module(f'data_utils.{data_name}_dataset')
    fn = getattr(module, 'get_loader', None)
    
    if fn is None:
        raise ValueError(f'not found data name: {data_name}')
    
    return fn(args)


def get_label_names(data_name):
    label_names_map = {
        'chgh': ['C'],
        'mmwhs': ['LV', 'RV', 'LA', 'RA', 'MLV', 'AA', 'PA'],
        'hvsmr': ['M', 'B'],
        'segthor': ['C']
    }
    return label_names_map[data_name]