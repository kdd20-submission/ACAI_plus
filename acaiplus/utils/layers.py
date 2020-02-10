import functools
import math
import torch
import torch.nn.functional as F
from torch import nn


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super(BaseModel, self).__init__()

        # dataset information
        self.dataset = cfg.DATA.DATASET
        self.temporal = cfg.DATA.TEMPORAL_BOOL
        self.nclass = cfg.DATA.NCLASS
        self.colors = cfg.DATA.COLORS
        self.width = cfg.DATA.WIDTH
        self.height = cfg.DATA.HEIGHT

        # model hyperparams
        self.depth = cfg.MODEL.DEPTH
        self.latent = cfg.MODEL.LATENT
        self.latent_width = cfg.MODEL.LATENT_WIDTH
        self.neg_slope = cfg.MODEL.NEG_SLOPE
        self.clf_dims = self.latent * (self.latent_width ** 2)

        # solver params
        self.LR = cfg.SOLVER.LR

    def forward(self, x):
        raise NotImplementedError


class Encoder(nn.Module):

    def __init__(self, depth, latent, colors, neg_slope):
        super(Encoder, self).__init__()
        self.neg_slope = neg_slope

        self.conv_1 = nn.Conv2d(in_channels=colors,
                                out_channels=depth,
                                kernel_size=(1, 1),
                                stride=(1, 1),
                                padding=0,
                                groups=1,
                                bias=True)
        nn.init.kaiming_normal_(self.conv_1.weight)

        self.conv_2 = _conv_2d_layer(depth, depth)
        self.conv_3 = _conv_2d_layer(depth, depth)
        self.conv_4 = _conv_2d_layer(depth, depth * 2)
        self.conv_5 = _conv_2d_layer(depth * 2, depth * 2)
        self.conv_6 = _conv_2d_layer(depth * 2, depth * 4)
        self.conv_7 = _conv_2d_layer(depth * 4, depth * 4)
        self.conv_8 = _conv_2d_layer(depth * 4, depth * 8)
        self.conv_9 = _conv_2d_layer(depth * 8, latent)

        self.avg_pool = functools.partial(F.avg_pool2d, kernel_size=2)

    def forward(self, x):
        x = self.conv_1(x)

        x = _conv_block(x=x, conv1=self.conv_2, conv2=self.conv_3,
                        pool_func=self.avg_pool, neg_slope=self.neg_slope)

        x = _conv_block(x=x, conv1=self.conv_4, conv2=self.conv_5,
                        pool_func=self.avg_pool, neg_slope=self.neg_slope)

        x = _conv_block(x=x, conv1=self.conv_6, conv2=self.conv_7,
                        pool_func=self.avg_pool, neg_slope=self.neg_slope)

        x = F.leaky_relu(self.conv_8(x), negative_slope=self.neg_slope)
        return self.conv_9(x)


class Decoder(nn.Module):

    def __init__(self, depth, latent, colors, neg_slope):
        super(Decoder, self).__init__()
        self.neg_slope = neg_slope

        self.conv_1 = _conv_2d_layer(latent, depth * 4)
        self.conv_2 = _conv_2d_layer(depth * 4, depth * 4)
        self.conv_3 = _conv_2d_layer(depth * 4, depth * 2)
        self.conv_4 = _conv_2d_layer(depth * 2, depth * 2)
        self.conv_5 = _conv_2d_layer(depth * 2, depth)
        self.conv_6 = _conv_2d_layer(depth, depth)
        self.conv_7 = _conv_2d_layer(depth, depth)
        self.conv_8 = _conv_2d_layer(depth, colors)

        self.interpolate = functools.partial(F.interpolate, scale_factor=2)

    def forward(self, z):
        z = _conv_block(z, conv1=self.conv_1, conv2=self.conv_2,
                        pool_func=self.interpolate, neg_slope=self.neg_slope)

        z = _conv_block(z, conv1=self.conv_3, conv2=self.conv_4,
                        pool_func=self.interpolate, neg_slope=self.neg_slope)

        z = _conv_block(z, conv1=self.conv_5, conv2=self.conv_6,
                        pool_func=self.interpolate, neg_slope=self.neg_slope)

        z = F.leaky_relu(self.conv_7(z), negative_slope=self.neg_slope)
        return self.conv_8(z)


class EncoderLines(Encoder):

    def __init__(self, depth, latent, colors, neg_slope):
        super(EncoderLines, self).__init__(depth=depth,
                                           latent=latent,
                                           colors=colors,
                                           neg_slope=neg_slope)

        self.conv_9 = _conv_2d_layer(depth * 8, depth * 8)
        self.conv_10 = _conv_2d_layer(depth * 8, depth * 16)
        self.conv_11 = _conv_2d_layer(depth * 16, latent)

    def forward(self, x):
        x = self.conv_1(x)

        x = _conv_block(x=x, conv1=self.conv_2, conv2=self.conv_3,
                        pool_func=self.avg_pool, neg_slope=self.neg_slope)

        x = _conv_block(x=x, conv1=self.conv_4, conv2=self.conv_5,
                        pool_func=self.avg_pool, neg_slope=self.neg_slope)

        x = _conv_block(x=x, conv1=self.conv_6, conv2=self.conv_7,
                        pool_func=self.avg_pool, neg_slope=self.neg_slope)

        x = _conv_block(x=x, conv1=self.conv_8, conv2=self.conv_9,
                        pool_func=self.avg_pool, neg_slope=self.neg_slope)

        x = F.leaky_relu(self.conv_10(x), negative_slope=self.neg_slope)
        return self.conv_11(x)


class DecoderLines(nn.Module):

    def __init__(self, depth, latent, colors, neg_slope):
        super(DecoderLines, self).__init__()
        self.conv__2 = _conv_2d_layer(latent, depth * 8)
        self.conv__1 = _conv_2d_layer(depth * 8, depth * 8)

        self.decoder = Decoder(depth=depth,
                               latent=latent,
                               colors=colors,
                               neg_slope=neg_slope)

        self.decoder.conv_1 = _conv_2d_layer(depth * 8, depth * 4)

    def forward(self, x):
        x = _conv_block(x, conv1=self.conv__2, conv2=self.conv__1,
                        pool_func=self.decoder.interpolate,
                        neg_slope=self.decoder.neg_slope)

        return self.decoder(x)


class Discriminator(nn.Module):
    def __init__(self, lstm=False, lines=False, vade=False, **kwargs):
        super(Discriminator, self).__init__()
        self.__dict__.update(kwargs)

        if lines:
            self.encoder = EncoderLines(depth=self.depth,
                                        latent=self.latent,
                                        colors=self.colors,
                                        neg_slope=self.neg_slope)
        elif vade:
            self.encoder = buildNetwork([self.input_dim] + self.encodeLayer)
        else:
            self.encoder = Encoder(depth=self.depth,
                                   latent=self.latent,
                                   colors=self.colors,
                                   neg_slope=self.neg_slope)

    def forward(self, x):

        z = self.encoder(x)

        z = z.contiguous().view(z.shape[0], -1)

        return torch.mean(z, -1)


class Classifier(torch.nn.Module):

    def __init__(self, encoder, input_dims, nclass):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.out = nn.Linear(input_dims, nclass)

    def forward(self, x):
        with torch.no_grad():
            z = self.encoder(x)

        try:
            z = z.contiguous().view(z.shape[0], -1)
        except AttributeError:
            z, mu, logvar = z
            z = z.contiguous().view(z.shape[0], -1)

        return F.log_softmax(self.out(z), dim=1)


class ClusteringLayer(nn.Module):

    def __init__(self, n_clusters, ds_size, hidden_size, alpha=1):
        super().__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initialized_centroids = False
        self.initialized_p = False
        self.p = nn.Parameter(torch.Tensor(ds_size, n_clusters), requires_grad=True)
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, hidden_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def assign_centroids(self, centroids):
        self.cluster_layer.data = centroids
        self.initialized_centroids = True

    def set_p(self, p):
        self.p = nn.Parameter(p)
        self.initialized_p = True

    def load_state_dict(self, state_dict):
        self.assign_centroids(state_dict['cluster_layer'])
        self.set_p(state_dict['p'])

    def estimate_initial_target_distribution(self, z):
        q = self.forward(z).data
        return target_distribution(q.data)

    def forward(self, z):
        """probability of cluster assignment Q (referred as q_ij in the paper)"""

        assert (self.initialized_centroids == True), "centroids haven't been initialized"

        z = z.view(z.shape[0], -1)
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


class VaeEncoder(nn.Module):
    def __init__(self, depth, latent, colors, hidden_size, dataset, device, neg_slope):
        super(VaeEncoder, self).__init__()
        self.device = device

        if dataset == 'lines':
            self.encoder = EncoderLines(depth, latent, colors, neg_slope=neg_slope)
        else:
            self.encoder = Encoder(depth, latent, colors, neg_slope=neg_slope)

        # Latent vectors mu and sigma
        self.mu = nn.Linear(hidden_size, hidden_size)
        self.logvar = nn.Linear(hidden_size, hidden_size)
        self.eps = nn.Parameter(torch.Tensor(hidden_size, hidden_size)).to(self.device)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().to(self.device)
        draw_normal = torch.distributions.Normal(loc=mu, scale=std)
        self.eps = draw_normal.sample()

        return mu + std * self.eps

    def reparameterize_trick(self, z):
        z_shape = z.shape
        flattened = z.view(z.shape[0], -1)

        mu = self.mu(flattened)
        logvar = self.logvar(flattened)

        z = self.reparameterize(mu, logvar)
        return z.view(z_shape), mu, logvar

    def forward(self, x):
        z_raw = self.encoder(x)
        z_shape = z_raw.shape
        z_sample, mu, logvar = self.reparameterize_trick(z_raw)
        if self.training:
            return z_sample, mu, logvar
        else:
            return mu.view(z_shape)


class VaeDecoder(nn.Module):
    def __init__(self, depth, latent, colors, latent_width, hidden_size, dataset, neg_slope):
        super(VaeDecoder, self).__init__()

        # reshaping dimensions
        self.latent = latent
        self.latent_width = latent_width
        self.colors = colors

        if dataset == 'lines':
            self.decoder = DecoderLines(depth, latent, colors, neg_slope=neg_slope)
        else:
            self.decoder = Decoder(depth, latent, colors, neg_slope=neg_slope)
        self._dec = nn.Linear(hidden_size, hidden_size)

    def forward(self, z):
        z_shape = z.shape
        out = self._dec(z.view(z.shape[0], -1))
        out = self.decoder(out.view(z_shape))
        return 2*torch.sigmoid(out) - 1


def adjust_learning_rate(init_lr, optimizer, epoch):
    lr = max(init_lr * (0.9 ** (epoch//10)), 0.0002)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr, optimizer


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def _conv_2d_layer(in_channels, out_channels):
    conv_layer = nn.Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=(3, 3), stride=(1, 1),
                           padding=1, groups=1, bias=True)
    nn.init.kaiming_normal_(conv_layer.weight)

    return conv_layer


def _conv_block(x, conv1, conv2, pool_func, neg_slope):
    x = F.leaky_relu(conv1(x), negative_slope=neg_slope)

    x = F.leaky_relu(conv2(x), negative_slope=neg_slope)

    return pool_func(x)


def custom_loss(recon_x, x, mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    bce = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)
