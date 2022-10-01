from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
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

