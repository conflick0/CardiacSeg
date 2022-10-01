from monai.losses import DiceCELoss
from losses.cotr_loss import CoTrLoss

def loss_func(loss_name):
    print(f'loss: {loss_name}')

    if loss_name == 'dice_ce':
        return DiceCELoss(to_onehot_y=True, softmax=True)

    elif loss_name == 'cotr':
        # the loss for CoTr deep supervision
        dc_loss = DiceCELoss(to_onehot_y=True, softmax=True)
        return CoTrLoss(dc_loss)
    
    else:
        raise ValueError(f'not found loss name: {loss_name}')


