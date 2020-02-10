import argparse
from collections import OrderedDict
import acaiplus.utils.layers as layers
from acaiplus.config import get_config
from acaiplus.utils.engines import create_vae_baseline_trainer, create_evaluator
from acaiplus.utils.handlers import add_handlers
from acaiplus.utils.data import get_data_loaders
from acaiplus.utils.utils import *
from acaiplus.utils.clustering import cluster


def main(cfg):

    set_random_seeds(cfg.SOLVER.SEED)

    train_loader, test_loader, plot_loader = get_data_loaders(cfg=cfg)

    # flattened dimension size of hidden space for classifier input
    clf_dims = cfg.MODEL.LATENT * (cfg.MODEL.LATENT_WIDTH ** 2)

    encoder = layers.VaeEncoder(
        depth=cfg.MODEL.DEPTH,
        latent=cfg.MODEL.LATENT,
        colors=cfg.DATA.COLORS,
        hidden_size=clf_dims,
        dataset=cfg.DATA.DATASET,
        device=cfg.MODEL.DEVICE,
        neg_slope=cfg.MODEL.NEG_SLOPE)

    decoder = layers.VaeDecoder(
        depth=cfg.MODEL.DEPTH,
        latent=cfg.MODEL.LATENT,
        colors=cfg.DATA.COLORS,
        latent_width=cfg.MODEL.LATENT_WIDTH,
        hidden_size=clf_dims,
        dataset=cfg.DATA.DATASET,
        neg_slope=cfg.MODEL.NEG_SLOPE)

    classifier = layers.Classifier(encoder=encoder,
                                   input_dims=clf_dims,
                                   nclass=cfg.DATA.NCLASS)

    opt_ae = torch.optim.Adam(list(encoder.parameters()) +
                              list(decoder.parameters()),
                              lr=cfg.SOLVER.LR, weight_decay=1e-5)
    opt_clf = torch.optim.Adam(classifier.out.parameters(),
                               lr=cfg.SOLVER.LR, weight_decay=1e-5)

    models_dict = OrderedDict({'encoder': encoder,
                               'decoder': decoder,
                               'classifier': classifier,
                               'opt_ae': opt_ae,
                               'opt_clf': opt_clf})

    if cfg.SOLVER.RESUME_EPOCH != '0':
        models_dict, cfg = load_models(models_dict=models_dict,
                                       cfg=cfg,
                                       epoch=cfg.SOLVER.RESUME_EPOCH)

    if cfg.MODEL.MODE.lower() in 'training':

        trainer = create_vae_baseline_trainer(models=models_dict,
					      custom_loss=layers.custom_loss,
					      cfg=cfg)

        evaluator = create_evaluator(models=models_dict,
                                     plot_data=plot_loader,
                                     cfg=cfg)

        trainer, evaluator = add_handlers(trainer=trainer,
                                          evaluator=evaluator,
                                          data=(train_loader, test_loader),
                                          models_dict=models_dict,
                                          cfg=cfg)

        trainer.run(train_loader, max_epochs=cfg.SOLVER.EPOCHS)

    elif cfg.MODEL.MODE.lower() in 'clustering':
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
    cfg.COMPLETED_EPOCHS = 0

    # cfg.freeze()

    print(' --- RUNNING {} NOW ---'.format(cfg.DIRS.CHKP_PREFIX))
    main(cfg)
