import os

from monai.data import decollate_batch
from monai.inferers import sliding_window_inference

import torch

from tqdm import tqdm


def val_epoch(loader, model, model_inferer, acc_func, post_label, post_pred, global_step, args):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader):
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
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            loader.set_description(
                "Validate (%d / %d Steps)" % (global_step, 10.0)
            )
        mean_dice_val = acc_func.aggregate().item()
        acc_func.reset()
    return mean_dice_val


def train_epoch(loader, model, optimizer, loss_func, writer, global_step, epoch, args):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].to(args.device), batch["label"].to(args.device))
        logit_map = model(x)
        loss = loss_func(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "[Epoch %d] Training (%d Steps) (loss=%2.5f)"
            % (epoch, global_step, loss)
        )
        writer.add_scalar("tr_loss", loss, global_step=global_step)
        global_step += 1
    return global_step


def save_checkpoint(filename, model, epoch, best_acc, early_stop_count, args, optimizer=None, scheduler=None):
    state_dict = model.state_dict()
    save_dict = {
      "epoch": epoch,
      "best_acc": best_acc,
      "early_stop_count": early_stop_count,
      "state_dict": state_dict,
    }
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.model_dir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
        start_epoch,
        best_acc,
        early_stop_count,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func,
        acc_func,
        model_inferer,
        post_label,
        post_pred,
        writer,
        args,
    ):

    global_step = start_epoch * len(train_loader)
    val_acc_best = best_acc

    for epoch in range(start_epoch, args.max_epoch+1):
        if early_stop_count == args.max_early_stop_count:
            break

        global_step = train_epoch(
            train_loader,
            model,
            optimizer,
            loss_func,
            writer,
            global_step,
            epoch,
            args
        )

        if (
            epoch % args.val_every == 0 and epoch != 0
        ) or epoch == args.max_epoch:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            val_avg_acc = val_epoch(
                epoch_iterator_val,
                model,
                model_inferer,
                acc_func,
                post_label,
                post_pred,
                global_step,
                args
            )
            writer.add_scalar("val_dice", val_avg_acc, global_step=global_step)

            if val_avg_acc > val_acc_best:
                val_acc_best = val_avg_acc
                early_stop_count = 0
                save_checkpoint(
                    'best_model.pth',
                    model,
                    epoch,
                    val_acc_best,
                    early_stop_count,
                    args,
                    optimizer,
                    scheduler
                )
                print(
                    "Best Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        val_acc_best, val_avg_acc
                    )
                )
            else:
                early_stop_count += 1
                print("Early stop count: ", early_stop_count)
                if args.save_checkpoint_freq != 0 and early_stop_count % args.save_checkpoint_freq == 0:
                    print(
                        "Final Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            val_acc_best, val_avg_acc
                        )
                    )
                    save_checkpoint(
                        'final_model.pth',
                        model,
                        epoch,
                        val_acc_best,
                        early_stop_count,
                        args,
                        optimizer,
                        scheduler
                    )
                else:
                  print(
                      "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                          val_acc_best, val_avg_acc
                      )
                  )

        if scheduler is not None:
            scheduler.step()

