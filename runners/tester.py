from monai.data import decollate_batch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import torch
from tqdm import tqdm


def eval(data_loader, model, model_inferer, post_label, post_pred):
    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
        get_not_nans=False
    )
    hd95_metric = HausdorffDistanceMetric(
        include_background=False,
        percentile=95,
        reduction="mean",
        get_not_nans=False
    )
    it = tqdm(data_loader, dynamic_ncols=True)
    steps = len(it)
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(it):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = model_inferer(val_inputs)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            hd95_metric(y_pred=val_output_convert, y=val_labels_convert)
            it.set_description(f"eval ({step} / {steps} Steps)")
    dc_vals = dice_metric.get_buffer().detach().cpu().numpy().squeeze()
    hd95_vals = hd95_metric.get_buffer().detach().cpu().numpy().squeeze()
    return dc_vals, hd95_vals


def run_testing(
        model,
        test_loader,
        model_inferer,
        post_label,
        post_pred,
    ):

    return eval(test_loader, model, model_inferer, post_label, post_pred)
