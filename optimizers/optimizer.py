import inspect


def _get_cls(base_cls, cls_name, param, args):
    cls = getattr(base_cls, cls_name, None)
    
    if cls is None:
        raise ValueError(f'not found class name: {cls_name}')
    
    # skip first param self
    cls_param_names = inspect.getfullargspec(cls).args[1:]
    
    arg_dicts = vars(args)
    kwargs = {}
    for name in cls_param_names:
        value = arg_dicts.get(name, None)
        if value:
            kwargs[name] = value
            
    print(kwargs)
    
    return cls(param, **kwargs)


def Optimizer(optim_name, model_params, args):
    import torch.optim as opt
    import lion_pytorch

    if optim_name == 'Lion':
        return _get_cls(lion_pytorch, optim_name, model_params, args)
    else:
        return _get_cls(opt, optim_name, model_params, args)


def LR_Scheduler(lr_schd_name, optim, args):
    import optimizers.lr_scheduler as schd    
    return _get_cls(schd, lr_schd_name, optim, args)



