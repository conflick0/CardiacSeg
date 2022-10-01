import torch


def optim(model, args):
    print(f'optimzer: {args.optim_name}')

    if args.optim_name == 'adamw':
        return torch.optim.AdamW(
          model.parameters(), 
          lr=args.optim_lr,
          weight_decay=args.reg_weight
        )

    elif args.optim_name == 'sgd':
        return torch.optim.SGD(
          model.parameters(),
          lr=args.optim_lr,
          weight_decay=args.reg_weight,
          momentum=args.momentum, 
          nesterov=True
        )
    else:
      raise ValueError(f'not found optimzer name: {args.optim_name}')


