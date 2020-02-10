import argparse
from collections import OrderedDict
from ignite.engine import Events
import acaiplus.utils.layers as layers
from acaiplus.config import get_config
from acaiplus.utils.engines import create_acai_trainer, create_idec_acai_finetuning_trainer, create_evaluator
from acaiplus.utils.handlers import add_handlers
from acaiplus.utils.data import get_data_loaders
from acaiplus.utils.utils import *
from acaiplus.utils.clustering import error, cluster, estimate_initial_centroids, inference_idec


def main(cfg):

    set_random_seeds(cfg.SOLVER.SEED)

    train_loader, test_loader, plot_loader = get_data_loaders(cfg=cfg)
    centroids_loader, _, _ = get_data_loaders(cfg=cfg, drop_last=False)

    if cfg.DATA.DATASET == 'lines':
        encoder = layers.EncoderLines(
            depth=cfg.MODEL.DEPTH,
            latent=cfg.MODEL.LATENT,
            colors=cfg.DATA.COLORS,
            neg_slope=cfg.MODEL.NEG_SLOPE)

        decoder = layers.DecoderLines(
            depth=cfg.MODEL.DEPTH,
            latent=cfg.MODEL.LATENT,
            colors=cfg.DATA.COLORS,
            neg_slope=cfg.MODEL.NEG_SLOPE)
    else:
        encoder = layers.Encoder(
            depth=cfg.MODEL.DEPTH,
            latent=cfg.MODEL.LATENT,
            colors=cfg.DATA.COLORS,
            neg_slope=cfg.MODEL.NEG_SLOPE)

        decoder = layers.Decoder(
            depth=cfg.MODEL.DEPTH,
            latent=cfg.MODEL.LATENT,
            colors=cfg.DATA.COLORS,
            neg_slope=cfg.MODEL.NEG_SLOPE)

    discriminator = layers.Discriminator(
        depth=cfg.MODEL.ADVDEPTH or cfg.MODEL.DEPTH,
        latent=cfg.MODEL.LATENT,
        colors=cfg.DATA.COLORS,
        neg_slope=cfg.MODEL.NEG_SLOPE)
    discriminator.reg = cfg.MODEL.REG
    discriminator.interp = int(cfg.MODEL.ADDED_CONSTR)
    discriminator.advweight = cfg.MODEL.ADVWEIGHT

    # flattened dimension size of hidden space for classifier input
    clf_dims = cfg.MODEL.LATENT * (cfg.MODEL.LATENT_WIDTH ** 2)

    classifier = layers.Classifier(encoder=encoder,
                                   input_dims=clf_dims,
                                   nclass=cfg.DATA.NCLASS)

    clustering_model = layers.ClusteringLayer(n_clusters=cfg.DATA.NCLASS,
                                              ds_size=len(train_loader) * cfg.DATA.BATCH_SIZE,
                                              hidden_size=clf_dims)
    clustering_model.gamma = cfg.MODEL.IDEC.GAMMA

    opt_ae = torch.optim.Adam(list(encoder.parameters()) +
                              list(decoder.parameters()),
                              lr=cfg.SOLVER.LR, weight_decay=1e-5)
    opt_clf = torch.optim.Adam(classifier.out.parameters(),
                               lr=cfg.SOLVER.LR, weight_decay=1e-5)
    opt_d = torch.optim.Adam(discriminator.parameters(),
                             lr=cfg.SOLVER.LR, weight_decay=1e-5)
    opt_ae_clust = torch.optim.Adam(list(encoder.parameters()) +
                                    list(decoder.parameters()) +
                                    list(clustering_model.parameters()), amsgrad=True)

    if cfg.MODEL.MODE.lower() in 'training':

        fine_tune_files = [f for f in os.listdir(cfg.DIRS.CHKP_DIR_FINE) if '.pth' in f]

        if not fine_tune_files:

            cfg.DIRS.CHKP_DIR = cfg.DIRS.CHKP_DIR_PRE

            models_dict = OrderedDict({'encoder': encoder,
                                       'decoder': decoder,
                                       'discriminator': discriminator,
                                       'classifier': classifier,
                                       'opt_ae': opt_ae,
                                       'opt_d': opt_d,
                                       'opt_clf': opt_clf})

            models_dict, cfg = load_models(models_dict=models_dict,
                                           cfg=cfg,
                                           epoch=cfg.SOLVER.RESUME_EPOCH)

            trainer = create_acai_trainer(
                models=models_dict, cfg=cfg)

            evaluator = create_evaluator(
                models=models_dict,
                plot_data=plot_loader,
                cfg=cfg)

            trainer, evaluator = add_handlers(trainer=trainer,
                                              evaluator=evaluator,
                                              data=(train_loader, test_loader),
                                              models_dict=models_dict,
                                              cfg=cfg)

            trainer.run(train_loader, max_epochs=cfg.MODEL.IDEC.PRETRAIN)

            models_dict.update({'clustering': clustering_model,
                                'opt_combined': opt_ae_clust})

            cfg.DIRS.CHKP_DIR = cfg.DIRS.CHKP_DIR_FINE

        else:

            cfg.DIRS.CHKP_DIR = cfg.DIRS.CHKP_DIR_FINE

            models_dict = OrderedDict({'encoder': encoder,
                                       'decoder': decoder,
                                       'discriminator': discriminator,
                                       'clustering': clustering_model,
                                       'classifier': classifier,
                                       'opt_ae': opt_ae,
                                       'opt_d': opt_d,
                                       'opt_combined': opt_ae_clust,
                                       'opt_clf': opt_clf})

            models_dict, cfg = load_models(models_dict=models_dict,
                                           cfg=cfg,
                                           epoch=cfg.SOLVER.RESUME_EPOCH)

        train_loader, test_loader, plot_loader = get_data_loaders(cfg=cfg,
                                                                  fine_tune=True)
        centroids_loader, _, _ = get_data_loaders(cfg=cfg, drop_last=False)

        clust_trainer = create_idec_acai_finetuning_trainer(
            models=models_dict, cfg=cfg)

        clust_evaluator = create_evaluator(
            models=models_dict,
            plot_data=plot_loader,
            cfg=cfg)

        clust_trainer, clust_evaluator = add_handlers(trainer=clust_trainer,
                                                      evaluator=clust_evaluator,
                                                      data=(train_loader, test_loader),
                                                      models_dict=models_dict,
                                                      cfg=cfg)

        @clust_trainer.on(Events.EPOCH_COMPLETED)
        def update_target_distribution(engine):

            _, target, tmp_q = inference_idec(
                model=models_dict['encoder'],
                idec_function=models_dict['clustering'],
                data_loader=centroids_loader,
                device=cfg.MODEL.DEVICE)

            models_dict['clustering'].set_p(torch.nn.Parameter(layers.target_distribution(tmp_q.data)))
            pred_cluster = tmp_q.cpu().numpy().argmax(1)
            acc = error(torch.cat(target).cpu().data.numpy(), pred_cluster, k=cfg.DATA.NCLASS)
            cfg.RESULTS.CLUSTER_ACC.append(acc)
            print(f'Cluster Accuracy: {acc}')

        # initializing centroids
        if not models_dict['clustering'].initialized_centroids or \
                not models_dict['clustering'].initialized_p:
            print('CALCULATING INITIAL CENTROIDS')

            # need to load data loader with drop_last=False
            # otherwise we can't init last samples of dataset that may be
            # needed during training (since shuffle = True)
            init_centroids = estimate_initial_centroids(
                encoder_model=models_dict['encoder'],
                data_loader=centroids_loader,
                n_clusters=cfg.DATA.NCLASS,
                device=cfg.MODEL.DEVICE)
            models_dict['clustering'].assign_centroids(init_centroids)

            _, _, ps = inference_idec(
                model=models_dict['encoder'],
                idec_function=models_dict['clustering'].estimate_initial_target_distribution,
                data_loader=centroids_loader,
                device=cfg.MODEL.DEVICE)
            models_dict['clustering'].set_p(ps)

        clust_trainer.run(train_loader, max_epochs=cfg.SOLVER.EPOCHS)

    elif cfg.MODEL.MODE.lower() in 'clustering':

        cfg.DIRS.CHKP_DIR = cfg.DIRS.CHKP_DIR_FINE

        models_dict, cfg = load_models(models_dict={'encoder': encoder},
                                       cfg=cfg,
                                       epoch=cfg.SOLVER.RESUME_EPOCH)

        _ = cluster(encoder=models_dict['encoder'],
                    train_loader=train_loader,
                    test_loader=test_loader,
                    cfg=cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch ACAI training.")
    parser.add_argument(
        '-c',
        '--config_file',
        default=None,
        metavar="FILE",
        help='path to config file',
        type=str)
    parser.add_argument(
        '-d',
        '--data_dir',
        default=None,
        type=str,
        help='location of data directory')
    parser.add_argument(
        '-o',
        '--output_dir',
        default=None,
        type=str,
        help='location of output directory')
    args = parser.parse_args()

    overload_params = get_overload_parameters(
        args,
        argument_list=[('data_dir', 'DIRS.DATA'),
                       ('output_dir', 'DIRS.OUTPUT')])

    cfg = get_config(config_file=args.config_file,
                     overload_parameters=overload_params)

    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.DIRS.DATA = os.path.join(cfg.DIRS.DATA, 'DATA')
    cfg.DIRS.CHKP_MAIN = os.path.join(cfg.DIRS.OUTPUT, 'RESULTS')
    cfg.DIRS.DATA_CHKP = create_dir(work_dir=cfg.DIRS.CHKP_MAIN, new_dir=cfg.DATA.DATASET)
    cfg.DIRS.CHKP_PREFIX = os.path.basename(__file__).split('.')[0]

    # create checkpoint dir name based on hyperparams values
    cfg.DIRS.CHKP_DIR = get_checkpoint_folder(
        dataset_chkp=cfg.DIRS.DATA_CHKP,
        chkp_prefix=cfg.DIRS.CHKP_PREFIX,
        cfg=cfg)
    cfg.DIRS.FIGURES = create_dir(cfg.DIRS.CHKP_DIR, 'figures')
    cfg.DIRS.CHKP_DIR_PRE = create_dir(work_dir=cfg.DIRS.CHKP_DIR,
                                      new_dir='pre_training')
    cfg.DIRS.CHKP_DIR_FINE = create_dir(work_dir=cfg.DIRS.CHKP_DIR,
                                       new_dir='fine_tuning')

    cfg.COMPLETED_EPOCHS = 0

    # cfg.freeze()

    print(' --- RUNNING {} NOW ---'.format(cfg.DIRS.CHKP_PREFIX))
    main(cfg)
