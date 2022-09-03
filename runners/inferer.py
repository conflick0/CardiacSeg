import torch
from monai.data import decollate_batch


def infer(model, data, model_inferer, device):
    model.eval()
    with torch.no_grad():
        output = model_inferer(data['image'].to(device))
        output = torch.argmax(output, dim=1)
    return output


def run_infering(
        model,
        data,
        model_inferer,
        device
    ):

    return infer(model, data, model_inferer, device)

    