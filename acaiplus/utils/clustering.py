"""
Cluster latents representations in AEs.
"""
import numpy as np
import torch
from sklearn.cluster import KMeans
from munkres import Munkres
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment


class CentroidDeviation:

    def __init__(self, clustering_model):
        self.clustering_model = clustering_model
        self.n_clusters = clustering_model.n_clusters
        self.eps = 1e-9
        self.previous_centroids = self.clustering_model.w.cpu().detach().clone().numpy()

    def __call__(self, clustering_model):
        current_centroids = clustering_model.w.cpu().detach().numpy()
        w_prev = np.reshape(self.previous_centroids, (self.n_clusters, -1))
        w_curr = np.reshape(current_centroids, (self.n_clusters, -1))
        deviation = np.abs((w_curr - w_prev).sum(-1) / (w_prev.sum(-1) + self.eps))

        self.w_prev = w_curr
        self.previous_centroids = current_centroids

        return deviation


@torch.no_grad()
def inference(model, data_loader, device=None):
    output = []
    y = []
    model.eval()

    if device:
        model.to(device)

    for batch in data_loader:
        x = batch[0].to(device)
        out = model(x)
        output.append(out.unsqueeze(1))
        y.append(batch[1].to(device))

    return output, y


@torch.no_grad()
def inference_idec(model, idec_function, data_loader, device=None):
    output = []
    extra_out = []
    y = []
    model.eval()

    if device:
        model.to(device)

    for batch in data_loader:
        x = batch[0].to(device)
        out = model(x)
        extra = idec_function(out)

        output.append(out.unsqueeze(1))
        y.append(batch[1].to(device))
        extra_out.append(extra)

    return output, y, torch.cat(extra_out)


def estimate_initial_centroids(encoder_model,
                               data_loader,
                               n_clusters,
                               device=None):

    latents_orig, y = inference(encoder_model, data_loader, device=device)
    latents = torch.cat(latents_orig).cpu().data.numpy()
    latents_flat = np.reshape(latents, (latents.shape[0], -1))
    y = torch.cat(y).cpu().data.numpy()

    kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(latents_flat)
    acc = error(y, kmeans.predict(latents_flat), k=n_clusters)
    print(f'INITIAL CLUSTER ACCURACY: {acc}')

    centroids_kmeans = torch.tensor(kmeans.cluster_centers_).to(device)

    return centroids_kmeans


def cluster_acc(y_true, y_pred):
    """Calculate clustering accuracy. Requires scikit-learn installed

    :param y_true: true labels, numpy.array with shape `(n_samples,)`
    :param y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    :return: accuracy
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, w


def error(target_cluster, cluster, k):
    """ Compute error between cluster and target cluster
    :param cluster: proposed cluster
    :param target_cluster: target cluster
    :return: error
    """
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            M[i][j] = np.sum(np.logical_and(cluster == i, target_cluster == j))
    m = Munkres()
    indexes = m.compute(-M)
    corresp = []
    for i in range(k):
        corresp.append(indexes[i][1])
    pred_corresp = [corresp[int(predicted)] for predicted in cluster]
    acc = np.sum(pred_corresp == target_cluster) / float(len(target_cluster))
    return acc


@torch.no_grad()
def _get_latents_and_labels(encoder, train_loader, test_loader, cfg):
    encoder.eval()
    encoder.to(cfg.MODEL.DEVICE)

    data = dict(train_latents=[],
                train_labels=[],
                test_latents=[],
                test_labels=[])

    for batch in train_loader:
        x, labels = batch[0].to(cfg.MODEL.DEVICE), batch[1].to(cfg.MODEL.DEVICE)

        if 'vade' in cfg.DIRS.CHKP_PREFIX:
            x = x.view(x.size(0), -1).float()
            z = encoder.sample(x)
        else:
            z = encoder(x)

        data['train_latents'].append(z.contiguous().view(z.shape[0], -1))
        data['train_labels'].append(labels)

    for batch in test_loader:
        x, labels = batch[0].to(cfg.MODEL.DEVICE), batch[1].to(cfg.MODEL.DEVICE)

        if 'vade' in cfg.DIRS.CHKP_PREFIX:
            x = x.view(x.size(0), -1).float()
            z = encoder.sample(x)
        else:
            z = encoder(x)

        data['test_latents'].append(z.contiguous().view(z.shape[0], -1))
        data['test_labels'].append(labels)

    print('\nDatasets shapes for clustering:')
    for key, val in data.items():
        # shape: [nr of images, latent, latent_width, latent_width]
        data[key] = torch.cat(val)
        data[key] = data[key].cpu().data.numpy()

        print(f'Shape of {key}: {data[key].shape}')

    return data


def cluster(encoder, train_loader, test_loader, cfg):
    data = _get_latents_and_labels(encoder, train_loader, test_loader, cfg)

    # SVD whintening of latent space
    print('\nSVD WHITENING')
    mean = data['train_latents'].mean(axis=0)
    data['train_latents'] -= mean
    data['test_latents'] -= mean
    s, vt = np.linalg.svd(data['train_latents'], full_matrices=False)[-2:]

    data['train_latents'] = (data['train_latents'].dot(vt.T) / (s + 1e-5))
    data['test_latents'] = (data['test_latents'].dot(vt.T) / (s + 1e-5))

    avg_acc_train = []
    avg_acc_test = []

    # for rnd_state in range(cfg['cluster_runs']):
    kmeans = KMeans(
        n_clusters=cfg.DATA.NCLASS,
        random_state=cfg.SOLVER.SEED,
        max_iter=1000,
        n_init=cfg.CLUSTER.N_RUNS,
        n_jobs=6)
    kmeans.fit(data['train_latents'])

    avg_acc_train.append(
        error(
            data['train_labels'],
            kmeans.predict(
                data['train_latents']),
            k=cfg.DATA.NCLASS))
    avg_acc_test.append(
        error(
            data['test_labels'],
            kmeans.predict(
                data['test_latents']),
            k=cfg.DATA.NCLASS))

    nmi_train = normalized_mutual_info_score(data['train_labels'],
                                             kmeans.predict(data['train_latents']))
    nmi_test = normalized_mutual_info_score(data['test_labels'],
                                            kmeans.predict(data['test_latents']))

    print(kmeans.cluster_centers_, '\n')
    print('Train/Test k-means objective = %.4f / %.4f' %
          (-kmeans.score(data['train_latents']), -kmeans.score(data['test_latents'])))
    print(f'Accuracy | train: {avg_acc_train[-1]} | test: {avg_acc_test[-1]}')
    print(f'NMI      | train: {nmi_train} | test: {nmi_test}')

    return np.sum(avg_acc_test) / float(len(avg_acc_test))


def vade_cluster_acc(y_pred, Y):
    assert y_pred.size == Y.size

    d = max(y_pred.max(), Y.max()) + 1
    w = np.zeros((d, d), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], Y[i]] += 1

    ind = linear_sum_assignment(w.max() - w)

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, w
