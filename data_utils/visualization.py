from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib as mpl


def show_color_bar(cmap, num_classes):
    '''show cmap color bar, label range [0, num_classes]'''
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
    cb = mpl.colorbar.ColorbarBase(
        ax,
        orientation='horizontal',
        cmap=cmap,
        norm=mpl.colors.Normalize(0, num_classes)
    )


def get_slicer_cmap(num_classes):
    '''get 3d slicer cmap (7 label for cardiac)'''
    colorarray = [
        [128 / 256, 174 / 256, 128 / 256, 1],
        [241 / 256, 214 / 256, 145 / 256, 1],
        [177 / 256, 122 / 256, 101 / 256, 1],
        [111 / 256, 184 / 256, 210 / 256, 1],
        [216 / 256, 101 / 256, 79 / 256, 1],
        [221 / 256, 130 / 256, 101 / 256, 1],
        [144 / 256, 238 / 256, 144 / 256, 1],
    ]
    return ListedColormap(colorarray[:num_classes])


def show_img_lbl_pred(img, lbl, pred, slice_idx, num_classes, axis_off=True, alpha=0.9, fig_size=(20, 10)):
    cmap = get_slicer_cmap(num_classes)

    plt.figure("check", fig_size)

    plt.subplot(1, 3, 1)
    plt.title(f"image (slice: {slice_idx})")
    plt.imshow(img, cmap="gray")
    if axis_off:
        plt.axis('off')

    titles = ['label', 'predict']
    ims = [lbl, pred]
    for i, (t, im) in enumerate(zip(titles, ims), 2):
        plt.subplot(1, 3, i)
        plt.title(f"image & {t} (slice: {slice_idx})")
        plt.imshow(img, cmap="gray")
        im_masked = np.ma.masked_where(im == 0, im)
        plt.imshow(
            im_masked,
            cmap,
            interpolation='none',
            alpha=alpha,
            vmin=1,
            vmax=num_classes
        )
        if axis_off:
            plt.axis('off')

    plt.tight_layout()
    plt.show()


def show_img_lbl(img, lbl, slice_idx, num_classes, axis_off=True, alpha=0.9, fig_size=(20, 10)):
    if num_classes < 8:
      cmap = get_slicer_cmap(num_classes)
    else:
      cmap = 'viridis'
    
    plt.figure("check", fig_size)

    plt.subplot(1, 2, 1)
    plt.title(f"image (slice: {slice_idx})")
    plt.imshow(img, cmap="gray")
    if axis_off:
        plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"image & label (slice: {slice_idx})")
    plt.imshow(img, cmap="gray")
    im_masked = np.ma.masked_where(lbl == 0, lbl)
    plt.imshow(
        im_masked,
        cmap,
        interpolation='none',
        alpha=alpha,
        vmin=1,
        vmax=num_classes
    )
    if axis_off:
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def show_img_lbl_preds(imgs, lbls, preds, pred_titles, slice_idxs, num_classes, axis_off=True, alpha=0.9, fig_size=(20, 10), show_img=True, show_lbl_dc=False):
    cmap = get_slicer_cmap(num_classes)
    row_num = len(imgs)
    col_num = len(preds[0]) + 1
    if show_img:
      col_num += 1

    lbl_title = 'label (dice:1.00)' if show_lbl_dc else 'label'
    
    subplot_idx = 1
    plt.figure("check", fig_size)
    for (img, lbl, pred_ls, slice_idx) in zip(imgs, lbls, preds, slice_idxs):
        if show_img:
            plt.subplot(row_num, col_num, subplot_idx)
            plt.title(f"image (slice: {slice_idx})")
            plt.imshow(img, cmap="gray")
            subplot_idx += 1
            if axis_off:
                plt.axis('off')

        titles = [lbl_title] + pred_titles
        ims = [lbl] + pred_ls

        for (t, im) in zip(titles, ims):
            plt.subplot(row_num, col_num, subplot_idx)
            plt.title(f"{t} (slice: {slice_idx})")
            plt.imshow(img, cmap="gray")
            im_masked = np.ma.masked_where(im == 0, im)
            plt.imshow(
                im_masked,
                cmap,
                interpolation='none',
                alpha=alpha,
                vmin=1,
                vmax=num_classes
            )
            if axis_off:
                plt.axis('off')
            subplot_idx += 1

    plt.tight_layout()
    plt.show()



def norm_img(img):
    img = np.array(img)
    _min = np.min(img)
    _max = np.max(img)
    norm = (img - _min) * 255.0 / (_max - _min)
    return np.uint8(norm)


def get_pred_label_overlap_img(image, pred, label):
    '''
    output red color is predict mask,
    green color is gt mask,
    yellow color is the intersection of prediction mask and gt mask
    '''
    image = norm_img(image)
    image = np.array(image)
    pred = np.array(pred)
    label = np.array(label)
    image = image.astype(np.uint8)

    pred_mask = (image * pred).astype(np.uint8)
    label_mask = (image * label).astype(np.uint8)

    union_mask = (image * np.logical_or(pred_mask, label_mask)).astype(np.uint8)

    no_mask = image - union_mask

    r = no_mask + pred_mask
    g = no_mask + label_mask
    b = no_mask

    rgb_img = np.stack([r, g, b], -1)
    rgb_img = Image.fromarray(rgb_img)
    return rgb_img


def show_img_lbl_preds_overlap(
      imgs, 
      lbls, 
      preds, 
      pred_titles, 
      slice_idxs, 
      num_classes, 
      axis_off=True, 
      alpha=0.9, 
      fig_size=(20, 10), 
      show_img=True, 
      show_lbl_dc=False
    ):

    cmap = get_slicer_cmap(num_classes)
    row_num = len(imgs)
    col_num = len(preds[0])
    if show_img:
      col_num += 1

    lbl_title = 'label (dice:1.00)' if show_lbl_dc else 'label'
    
    subplot_idx = 1
    plt.figure("check", fig_size)
    for (img, lbl, pred_ls, slice_idx) in zip(imgs, lbls, preds, slice_idxs):
        if show_img:
            plt.subplot(row_num, col_num, subplot_idx)
            plt.title(f"image (slice: {slice_idx})")
            plt.imshow(img, cmap="gray")
            subplot_idx += 1
            if axis_off:
                plt.axis('off')
        
        titles = pred_titles
        for (t, prd) in zip(titles, pred_ls):
            plt.subplot(row_num, col_num, subplot_idx)
            plt.title(f"{t} (slice: {slice_idx})")

            overlap_img = get_pred_label_overlap_img(img, prd, lbl)
            plt.imshow(overlap_img)

            if axis_off:
                plt.axis('off')
            subplot_idx += 1

    plt.tight_layout()
    plt.show()

