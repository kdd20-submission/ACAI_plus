import os
import matplotlib.pyplot as plt
import PIL.Image
from os.path import join
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from acaiplus.utils.interpolation import *
from acaiplus.utils.data import get_dataset
import seaborn as sns


# font settings for all plots
sns.set(style='white', context='talk')
sns.set_palette('bright')
FONT = {'weight': 'bold',
        'size': 22}
plt.rc('font', **FONT)
DPI = 400
QUALITY = 100


def _flatten_lines(lines, padding=2):
    # increase the image size by adding padding
    if lines.ndim == 4:
        lines = np.swapaxes(lines, 1, -1)
    padding = np.ones((lines.shape[0], padding) + lines.shape[2:])
    lines = np.concatenate([padding, lines, padding], 1)
    lines = np.concatenate(lines, 0)

    # transpose image into imgSize, imgSize*numberImages, (remaining
    # dimensions if there are e.g. colors)
    return np.transpose(lines, [1, 0] + list(range(2, lines.ndim)))


def show_figure(
        line_interpolation,
        save=True,
        folder='figures',
        save_extend=''):
    fig = plt.figure(figsize=(15, 1))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(
        _flatten_lines(line_interpolation),
        cmap=plt.cm.gray,
        interpolation='nearest')
    plt.gca().set_axis_off()

    if save:
        plt.savefig(os.path.join(folder,
                                 f'line_{save_extend}.pdf'), aspect='normal')
    return plt


def save_images(x, out, interpolated_data, cfg, temporal=False):
    orig_x = x.data.cpu().numpy().squeeze()[:cfg.VIZ.NR_LINES]
    recon_x = out.data.cpu().numpy().squeeze()[:cfg.VIZ.NR_LINES]

    plt_interp = save_interpolation(
        interpolated_data,
        cfg,
        save_extend=f'epoch_{cfg.SOLVER.EVAL_EPOCH + 1}')

    plt_recon = save_reconstruction(
        orig_x,
        recon_x,
        cfg,
        save_extend=f'epoch_{cfg.SOLVER.EVAL_EPOCH + 1}')

    return plt_interp, plt_recon


def save_reconstruction(orig_x, recon_x, cfg, save_extend=''):
    list_data = [(orig_x, 'Original'), (recon_x, 'Reconstructed')]
    fig, axes = plt.subplots(2, 1, figsize=(48, 10))

    for data_i, (data_x, title) in enumerate(list_data):
        if len(data_x.shape) == 2:
            data_x = np.reshape(data_x, [data_x.shape[0], 28, 28])
            data_x = np.swapaxes(data_x, 1, -1)

        axes[data_i].imshow(
            _flatten_lines(data_x),
            cmap=plt.cm.gray,
            interpolation='nearest')
        axes[data_i].set_title(f"{title}", fontsize=40)
        axes[data_i].axis('off')
    plt.savefig(os.path.join(cfg.DIRS.FIGURES,
                             f'{cfg.DIRS.CHKP_PREFIX}_reconstruction_{save_extend}.pdf'),
                aspect='normal')
    return axes


def save_interpolation(line_interpolation, cfg, save_extend=''):
    fig = plt.figure(figsize=(15, 1))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(_flatten_lines(line_interpolation),
              cmap=plt.cm.gray,
              interpolation='nearest')
    plt.gca().set_axis_off()
    plt.savefig(
        os.path.join(
            cfg.DIRS.FIGURES,
            f'{cfg.DIRS.CHKP_PREFIX}_interpolation_{save_extend}.pdf'),
        aspect='normal')
    return ax


def to_single_rgb(img):
    img = np.asarray(img)
    if len(img.shape) == 4:  # take first frame from animations
        return img[0, :, :, :]
    if len(img.shape) == 2:  # convert gray to rgb
        img = img[:, :, np.newaxis]
        return np.repeat(img, 3, 2)  # might np.tile(img, [1,1,3]) be faster?
    if img.shape[-1] == 4:  # drop alpha
        return img[:, :, :3]
    else:
        return img


def imsave(img, model_name='', fmt='png', save=False, work_dir=None, save_ext=''):
    if img is None:
        raise TypeError('input image not provided')
    if len(img.shape) == 1:
            raise ValueError('input is one-dimensional', img.shape)
    if len(img.shape) == 3 and img.shape[-1] == 1:
        img = img.squeeze()
    img = np.uint8(np.clip(img, 0, 255))
    if fmt == 'jpg' or fmt == 'jpeg':
        img = to_single_rgb(img)
    if save:
        file_name = os.path.join(work_dir, f'{model_name}_mosaic_{save_ext}')
        PIL.Image.fromarray(img).save(file_name, fmt)


@torch.no_grad()
def save_latent_digits(encoder, decoder, cfg):
    training_set, _, _ = get_dataset(cfg, fine_tune=False)

    idx = training_set.targets == cfg.VIZ.LATENT_DIGIT
    training_set.targets = training_set.targets[idx]
    training_set.data = training_set.data[idx]

    train_loader = DataLoader(training_set,
                              batch_size=cfg.DATA.BATCH_SIZE,
                              shuffle=True, drop_last=True)

    # encode all data points of that digit
    x = next(iter(train_loader))[0].to(cfg.MODEL.DEVICE)
    img = random_samples(x=x, encoder=encoder, decoder=decoder, cfg=cfg, temporal=False)

    imsave(img=img * 255,
           model_name=cfg.DIRS.CHKP_PREFIX,
           work_dir=cfg.DIRS.FIGURES,
           save=True,
           save_ext=f'latent_digits_epoch_{cfg.SOLVER.EVAL_EPOCH + 1}')


@torch.no_grad()
def save_svd_plots(encoder, folder, save_extend, cfg):
    all_singulars = np.zeros((cfg.DATA.NCLASS, cfg.VIZ.SVD_COMPONENTS))
    for digit in range(0, 10):
        training_set, _, _ = get_dataset(cfg, fine_tune=False)
        idx = training_set.targets == digit
        training_set.targets = training_set.targets[idx]
        training_set.data = training_set.data[idx]

        train_loader = DataLoader(training_set,
                                  batch_size=cfg.DATA.BATCH_SIZE,
                                  shuffle=True, drop_last=True)

        # encode all data points of that digit
        all_z = list()
        for batch in train_loader:
            x, target = batch[0].to(cfg.MODEL.DEVICE), batch[1].to(cfg.MODEL.DEVICE)

            with torch.no_grad():
                z = encoder(x)
                all_z.append(z)

        # shape: (nr batches, batch_size, channels, width, height)
        z = torch.stack(all_z)

        # shape: (all_images, channels, width, height)
        z = z.view([-1] + list(z.shape[2:])).cpu().numpy()

        # shape: (all_images, all_dims)
        z = np.reshape(z, [z.shape[0], -1])

        svd = PCA(n_components=cfg.VIZ.SVD_COMPONENTS)
        svd.fit(z)
        all_singulars[digit] = svd.singular_values_

    x_plot = np.arange(cfg.VIZ.SVD_COMPONENTS)
    x_labels = [str(i + 1) if ((i + 1) % 10 == 0 or i + 1 == 1) else '' \
                for i in x_plot]
    f, axes = plt.subplots(3, 4, figsize=(30, 30))
    digits_iter = np.linspace(0, 9, 9, dtype=int)

    for digit, ax in zip(digits_iter, axes.flatten()):
        g = sns.lineplot(x=x_plot, y=all_singulars[digit], palette="rocket", ax=ax)
        g.axhline(0, color="k", clip_on=False)
        g.set_ylabel("PCA", fontsize=25)
        g.set_xlabel('Component', fontsize=25)
        g.set_title(f'Class {digit}', fontsize=40)
        g.set_xticklabels(x_labels)
        sns.despine()

    # saving numpy array and SVD plot
    npy_file_name = join(folder, f'{cfg.DIRS.CHKP_PREFIX}_PCA_{save_extend}')
    svg_file_name = join(folder, f'figures/{cfg.DIRS.CHKP_PREFIX}_PCA_{save_extend}')
    np.save(npy_file_name + '.npy', all_singulars)
    f.savefig(svg_file_name + '.png', dpi=DPI, bbox_inches='tight')

    # plot normalized average PCA across classes
    all_singulars = all_singulars / np.linalg.norm(all_singulars)
    all_singulars_avg = all_singulars.mean(axis=0)
    all_singulars_std = all_singulars.std(axis=0)

    # create plot
    f, ax = plt.subplots(1, 1, figsize=(15, 15))
    g_avg = sns.lineplot(x=x_plot, y=all_singulars_avg,  # all_singulars_avg,
                         palette='rocket', ax=ax)
    g_avg.set_title(f'PCA: Class Average', fontsize=30)
    g_avg.set_xlabel('Component', fontsize=25)
    g_avg.set_ylabel('Explained Variance', fontsize=25)
    g_avg.set_ylim(0, 0.3)
    g_avg.set_xticklabels(x_labels)
    g_avg.errorbar(x=x_plot, y=all_singulars_avg,
                   yerr=all_singulars_std, color='black',
                   ecolor='black', elinewidth=2, fmt='|', capsize=0)  #
    sns.despine()

    # saving numpy array and SVD plot
    npy_file_name = join(folder, f'{cfg.DIRS.CHKP_PREFIX}_PCA_AVERAGES_{save_extend}')
    svg_file_name = join(folder, f'figures/{cfg.DIRS.CHKP_PREFIX}_PCA_AVERAGES_{save_extend}')
    np.save(npy_file_name + '.npy', all_singulars)
    f.savefig(svg_file_name + '.svg', dpi=DPI, bbox_inches='tight')

    return all_singulars, all_singulars_avg, all_singulars_std
