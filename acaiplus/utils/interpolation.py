import numpy as np
import torch
from acaiplus.utils.utils import swapaxes, make_mosaic


def status(x, encoder, decoder, args, temporal=False):
    chunks = [reconstruct(x, encoder, decoder, args, temporal=temporal),
              interpolate_2(x, encoder, decoder, args, temporal=temporal),
              interpolate_4(x, encoder, decoder, args, temporal=temporal),
              random_samples(x, encoder, decoder, args, temporal=temporal)]
    chunks = [np.pad(e, (0, 1), mode='constant', constant_values=255)
              for e in chunks]
    chunks = np.array(chunks)

    return make_mosaic(chunks.squeeze())


@torch.no_grad()
def interpolate_data(encoder, decoder, x, cfg):
    side = cfg.VIZ.NR_LINES
    with torch.no_grad():
        z = encoder(x)
    z = z.data.cpu().numpy()

    aa, bb = z[0], z[-1]
    aa = np.expand_dims(aa, axis=0)
    bb = np.expand_dims(bb, axis=0)

    z_interp = [aa * (1 - t) + bb * t for t in np.linspace(0, 1, side - 2)]
    z_interp = np.vstack(z_interp)

    with torch.no_grad():
        x_interp = decoder(torch.FloatTensor(z_interp).to(cfg.MODEL.DEVICE))

    x_interp = x_interp.cpu().data.numpy()
    x_fixed = x.squeeze().data.cpu().numpy()

    if len(x.shape) == 2:
        x_dims = [cfg.VIZ.NR_LINES] + list(x.squeeze().shape[1:])
    else:
        x_dims = [cfg.VIZ.NR_LINES] + list(x.shape[1:])

    x_all = np.zeros(x_dims) + 1
    x_all[0] = x_fixed[0]
    x_all[-1] = x_fixed[-1]
    x_all[1:-1] = x_interp
    x_all = x_all.squeeze()

    if len(x_all.shape) == 2:
        x_all = np.reshape(x_all, [cfg.VIZ.NR_LINES,
                                   cfg.DATA.WIDTH, cfg.DATA.HEIGHT])
    return x_all


@torch.no_grad()
def interpolate_lines_benchmark(encoder, decoder, x, cfg):
    side = cfg.VIZ.NR_LINES
    with torch.no_grad():
        z = encoder(x)
    z = z.data.cpu().numpy()

    half_batch_size = int(z.shape[0] / 2)
    aa, bb = z[: half_batch_size], z[-half_batch_size:]

    z_interp = np.array([aa * (1 - t) + bb * t for t in np.linspace(0, 1, side - 2)])
    z_interp_shape = z_interp.shape
    z_interp = np.reshape(z_interp, [z_interp.shape[0] * z_interp.shape[1]]
                          + list(z_interp.shape[2:]))

    with torch.no_grad():
        x_interp = decoder(torch.FloatTensor(z_interp).to(cfg.MODEL.DEVICE))

    x_interp = x_interp.cpu().data.numpy()
    x_interp = np.reshape(x_interp, [z_interp_shape[0], z_interp_shape[1]]
                          + list(x_interp.shape[1:]))

    # (batch, interpolations, channels, height, width)
    # -> (batch, interpolations, height, width, channels)
    return x_interp.squeeze()[..., np.newaxis]


@torch.no_grad()
def reconstruct(x, encoder, decoder, cfg, temporal=False):
    out = decoder(encoder(x[:cfg.VIZ.GRID_LINES * 4]))
    out = swapaxes(out, 1, -1)

    if temporal:
        # [batch, sensor_cols, sequence, 1] -> [batch, 1, sequence]
        out = out.squeeze()[..., 0].unsqueeze(1)
    return make_mosaic(out.cpu().data.numpy())


@torch.no_grad()
def interpolate_2(x, encoder, decoder, cfg, temporal=False):
    halv_grid = cfg.VIZ.GRID_LINES // 2
    z = encoder(x)
    z = z.data.cpu().numpy()

    a, b = z[:halv_grid], z[-halv_grid:]
    z_interp = [a * (1 - t) + b * t for t
                in np.linspace(0, 1, halv_grid - 2)]
    z_interp = np.vstack(z_interp)
    x_interp = decoder(torch.FloatTensor(z_interp).to(cfg.MODEL.DEVICE))
    x_interp = x_interp.cpu().data.numpy()
    x_fixed = x.data.cpu().numpy()

    if temporal:
        x_interp = x_interp[:, 0, :]
        x_fixed = x_fixed[:, 0, :]

    out = []
    out.extend(x_fixed[:halv_grid])
    out.extend(x_interp)
    out.extend(x_fixed[-halv_grid:])
    out = np.asarray(out)
    out = swapaxes(out, 1, -1)  # if args['colors'] == 3 else out.squeeze()

    if temporal:
        # [batch, sensor_cols, sequence, 1] -> [batch, 1, sequence]
        out = out.squeeze()[..., 0]
        out = np.expand_dims(out, axis=1)
    return make_mosaic(out)


@torch.no_grad()
def interpolate_4(x, encoder, decoder, cfg, temporal=False):
    halv_grid = cfg.VIZ.GRID_LINES // 2
    z = encoder(x[:halv_grid])
    z = z.data.cpu().numpy()

    n = halv_grid * halv_grid
    xv, yv = np.meshgrid(np.linspace(0, 1, halv_grid),
                         np.linspace(0, 1, halv_grid))
    xv = xv.reshape(n, 1, 1, 1)
    yv = yv.reshape(n, 1, 1, 1)

    z_interp = \
        z[0] * (1 - xv) * (1 - yv) + \
        z[1] * xv * (1 - yv) + \
        z[2] * (1 - xv) * yv + \
        z[3] * xv * yv

    x_fixed = x.data.cpu().numpy()

    x_interp = decoder(torch.FloatTensor(z_interp.squeeze()).to(cfg.MODEL.DEVICE))
    x_interp = x_interp.data.cpu().numpy()
    x_interp[0] = x_fixed[0]
    x_interp[halv_grid - 1] = x_fixed[1]
    x_interp[n - halv_grid] = x_fixed[2]
    x_interp[n - 1] = x_fixed[3]
    x_interp = swapaxes(x_interp, 1, -1)

    if temporal:
        # [batch, sensor_cols, sequence, 1] -> [batch, 1, sequence]
        x_interp = x_interp.squeeze()[..., 0]
        x_interp = np.expand_dims(x_interp, axis=1)

    return make_mosaic(x_interp)


# random samples based on a reference distribution
@torch.no_grad()
def random_samples(x, encoder, decoder, cfg, temporal=False):
    z = encoder(x[:cfg.VIZ.GRID_LINES * 4])
    z = z.data.cpu().numpy()
    z_sample = np.random.normal(
        loc=z.mean(
            axis=0), scale=z.std(
            axis=0), size=z.shape)
    x_sample = decoder(torch.FloatTensor(z_sample).to(cfg.MODEL.DEVICE))
    x_sample = x_sample.data.cpu().numpy()
    x_sample = swapaxes(x_sample, 1, -1)

    if temporal:
        # [batch, sensor_cols, sequence, 1] -> [batch, 1, sequence]
        x_sample = x_sample.squeeze()[..., 0]
        x_sample = np.expand_dims(x_sample, axis=1)

    return make_mosaic(x_sample)
