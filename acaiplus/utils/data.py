import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, SVHN
from torchvision.transforms import Compose, ToTensor, Resize, Lambda
from acaiplus.utils.utils import create_dir, swapaxes

__all__ = ['get_dataset', 'get_data_loaders', 'draw_line']


def get_dataset(cfg, fine_tune):
    data_transform = Compose([Resize((32, 32)), ToTensor()])
    mnist_transform = Compose([Resize((32, 32)), ToTensor(),
                               Lambda(lambda x: swapaxes(x, 1, -1))])
    vade_transform = Compose([ToTensor()])

    if cfg.DATA.DATASET == 'mnist':
        transform = vade_transform if 'vade' in cfg.DIRS.CHKP_PREFIX \
            else mnist_transform

        training_set = MNIST(download=True, root=cfg.DIRS.DATA,
                             transform=transform, train=True)
        val_set = MNIST(download=False, root=cfg.DIRS.DATA,
                        transform=transform, train=False)
        plot_set = copy.deepcopy(val_set)

    elif cfg.DATA.DATASET == 'svhn':
        training_set = SVHN(download=True,
                            root=create_dir(cfg.DIRS.DATA, 'SVHN'),
                            transform=data_transform,
                            split='train')
        val_set = SVHN(download=True,
                       root=create_dir(cfg.DIRS.DATA, 'SVHN'),
                       transform=data_transform,
                       split='test')
        plot_set = copy.deepcopy(val_set)

    elif cfg.DATA.DATASET == 'cifar':
        training_set = CIFAR10(download=True,
                               root=create_dir(cfg.DIRS.DATA, 'CIFAR'),
                               transform=data_transform,
                               train=True)
        val_set = CIFAR10(download=True,
                          root=create_dir(cfg.DIRS.DATA, 'CIFAR'),
                          transform=data_transform,
                          train=False)
        plot_set = copy.deepcopy(val_set)

    elif cfg.DATA.DATASET == 'lines':
        vae = True if 'vae' in cfg.DIRS.CHKP_PREFIX else False
        training_set = LinesDataset(args=cfg,
                                    multiplier=1000,
                                    dataset_type='train',
                                    vae=vae)
        val_set = LinesDataset(args=cfg,
                               multiplier=10,
                               dataset_type='test',
                               vae=vae)
        plot_set = LinesDataset(args=cfg,
                                multiplier=1,
                                dataset_type='plot',
                                vae=vae)

    if 'idec' in cfg.DIRS.CHKP_PREFIX and fine_tune:
        training_set = IdecDataset(training_set)
        val_set = IdecDataset(val_set)
        plot_set = IdecDataset(plot_set)

    return training_set, val_set, plot_set


def get_data_loaders(cfg, drop_last=True, fine_tune=False):

    training_set, val_set, plot_set = get_dataset(cfg, fine_tune=fine_tune)

    train_loader = DataLoader(training_set,
                              batch_size=cfg.DATA.BATCH_SIZE,
                              shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_set,
                            batch_size=cfg.DATA.BATCH_SIZE,
                            shuffle=True, drop_last=drop_last)

    plot_loader = DataLoader(val_set,
                             batch_size=cfg.DATA.BATCH_SIZE,
                             shuffle=False, drop_last=drop_last)

    return train_loader, val_loader, plot_loader


class IdecDataset(Dataset):
    def __init__(self, dataset):
        super(IdecDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        img, target = self.dataset.__getitem__(index)
        return img, target, index

    def __len__(self):
        return self.dataset.data.shape[0]


class GetRandomData:
    def __init__(self, data_list, limit, plot_data):
        self.data_list = list(data_list) * limit
        self.counter = 0

        if plot_data:
            self._get_data = self._get_plot_data
        else:
            self._get_data = self._get_random_choice

    def _get_random_choice(self):
        return np.random.choice(self.data_list, 1)

    def _get_plot_data(self):
        return self.data_list[self.counter]

    def __call__(self):
        self.counter += 1
        return self._get_data()


class LinesDataset(Dataset):
    def __init__(self, args, multiplier, dataset_type,
                 plot_data=False, vae=False):
        from os.path import join
        import pickle

        vae = '_vae' if vae else ''
        lines_limit = args['batch_size'] * multiplier

        lines_data_dir = create_dir(args['data_dir'], 'LINES')
        file_name = join(lines_data_dir,
                         f'{dataset_type}_{lines_limit}{vae}.pickle')

        try:
            with open(file_name, 'rb') as f:
                self.ds = pickle.load(f)
        except OSError:
            print(f'--- CREATING {dataset_type} DATASET ---')
            self.ds = input_lines(limit=lines_limit,
                                  plot_data=plot_data)
            with open(file_name, 'wb') as f:
                pickle.dump(self.ds, f)

        self.data = self.ds['x']
        self.targets = self.ds['target']

    def __getitem__(self, index):
        img = self.data[index]
        label = self.targets[index]
        return img, label

    def __len__(self):
        return self.data.shape[0]


def draw_line(angle, height, width, w=2.):
    import math

    m = np.zeros((height, width, 1))
    x0 = height * 0.5
    y0 = width * 0.5
    x1 = x0 + (x0 - 1) * math.cos(-angle)
    y1 = y0 + (y0 - 1) * math.sin(-angle)
    flip = False

    # check the orientation of the line by comparing x/y coordinates
    # flip coordinates for certain angles
    if abs(y0 - y1) < abs(x0 - x1):
        x0, x1, y0, y1 = y0, y1, x0, x1
        flip = True
    if y1 < y0:
        x0, x1, y0, y1 = x1, x0, y1, y0
    x0, x1 = x0 - w / 2, x1 - w / 2
    dx = x1 - x0
    dy = y1 - y0
    ds = dx / dy if dy != 0 else 0
    yi = int(math.ceil(y0)), int(y1)
    points = []

    # draw the actual lines
    for y in range(int(y0), int(math.ceil(y1))):
        if y < yi[0]:
            weight = yi[0] - y0
        elif y > yi[1]:
            weight = y1 - yi[1]
        else:
            weight = 1
        xs = x0 + (y - y0 - .5) * ds
        xe = xs + w
        xi = int(math.ceil(xs)), int(xe)
        if xi[0] != xi[1]:
            points.append((y, slice(xi[0], xi[1]), weight))
        if xi[0] != xs:
            points.append((y, int(xs), weight * (xi[0] - xs)))
        if xi[1] != xe:
            points.append((y, xi[1], weight * (xe - xi[1])))
    if flip:
        points = [(x, y, z) for y, x, z in points]
    for y, x, z in points:
        m[y, x] += 2 * z

    m = m / 2.0
    m = m.clip(0, 1)
    return m


def input_lines(size=(1, 32, 32), limit=None, plot_data=False):
    import math
    import random

    c, h, w = size
    count = 0
    ds = {'x': [],
          'target': []}

    random_angles = [random.random() for _ in range(limit + 1)]
    get_angle = GetRandomData(random_angles, 1, plot_data)

    while limit is None or count < limit:
        angle = 2 * get_angle() * math.pi
        m = draw_line(angle, h, w)
        label = int(10 * angle / (2 * math.pi - 1e-6))
        count += 1
        ds['x'].append(m)
        ds['target'].append(label)

    # turn list [h, w, 1] into torch(1, h, w)
    ds['x'] = torch.Tensor(np.array(ds['x']).squeeze()).unsqueeze(1)
    ds['target'] = torch.LongTensor(ds['target'])

    return ds
