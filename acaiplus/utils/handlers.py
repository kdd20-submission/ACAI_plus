import numpy as np
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from tensorboardX import SummaryWriter
from tqdm import tqdm

from acaiplus.utils.utils import calc_mean_non_empty, create_dir, save_ignite_params

__all__ = ['add_handlers']


class Tensorboard:
    def __init__(self, log_dir):
        try:
            self.writer = SummaryWriter(logdir=log_dir)
        except TypeError:
            self.writer = SummaryWriter(log_dir=log_dir)

    def __call__(self, engine, prefix='', output_transform=lambda x: x):
        output_dict = output_transform(engine.state.output)
        for key, value in output_dict.items():
            self.writer.add_scalar(
                "{}/{}".format(prefix, key), value, engine.state.iteration)


class ProgressBar:
    """
    TQDM progress bar handler to log training progress and computed metrics

    Args:
        engine (ignite.Engine): an engine object
        loader (iterable or DataLoader): data loader object
        output_transform: transform a function that transforms engine.state.output
                into a dictionary of format {metric_name: metric_value}
        mode (str): 'iteration' or 'epoch' (default=epoch)
        log_interval (int or None): interval of which the metrics information is displayed.
                            If set to None, only the progress bar is shown and not
                            the metrics. (default=1)

    Example:
        (...)
        pbar = ProgressBar(trainer, train_loader, output_transform=lambda x: {'loss': x})
        trainer.add_event_handler(Events.ITERATION_COMPLETED, pbar)

    Note:
        1) Bear in mind that `output_transform` should return a dictionary, whose values are floats.
        This is due to the running average over every metric that is being computed. If you have
        metrics that are tensors or arrays, you will have to unroll each value to its own
        dictionary key.

        2) When adding this handler to an engine, it is recommend that you replace every print
        operation in the engine's handlers with `tqdm.write()` to guarantee the correct stdout format.
    """

    def __init__(
            self,
            engine,
            loader,
            output_transform=lambda x: x,
            mode='epoch',
            log_interval=1):
        self.num_iterations = len(loader)
        self.metrics = {}
        self.alpha = 0.98
        self.output_transform = output_transform
        self.pbar = None
        self.mode = mode
        self.log_interval = log_interval

        if log_interval is not None:
            assert log_interval >= 1, 'log_frequency must be positive'

        assert mode in {'iteration', 'epoch'}, \
            'incompatible mode {}, accepted modes {}'.format(
                mode, {'iteration', 'epoch'})

        log_event = Events.EPOCH_COMPLETED if mode == 'epoch' \
            else Events.ITERATION_COMPLETED

        engine.add_event_handler(Events.EPOCH_STARTED, self._reset)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self._close)
        engine.add_event_handler(log_event, self._log_message)

    def _calc_running_avg(self, engine):
        output = self.output_transform(engine.state.output)
        for k, v in output.items():
            old_v = self.metrics.get(k, v)
            new_v = self.alpha * old_v + (1 - self.alpha) * v
            self.metrics[k] = new_v

    def _reset(self, engine):
        self.pbar = tqdm(
            total=self.num_iterations,
            leave=False,
            bar_format='{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]')

    def _close(self, engine):
        self.pbar.close()

    def _log_message(self, engine):

        i = engine.state.epoch if self.mode == 'epoch' else engine.state.iteration

        if self.log_interval and i % self.log_interval == 0:
            if self.mode == 'epoch':
                message = 'Epoch {}'.format(engine.state.epoch)
            else:
                message = 'Iteration {}'.format(engine.state.iteration)
            for name, value in self.metrics.items():
                message += ' | {}={:.2e}'.format(name, value)

            tqdm.write(message)

    def _format_metrics(self):
        formatted_metrics = {}
        for key in self.metrics:
            formatted_metrics[key] = '{:.2e}'.format(self.metrics[key])
        return formatted_metrics

    def __call__(self, engine):
        self._calc_running_avg(engine)
        self.pbar.set_description('Epoch {}'.format(engine.state.epoch))
        self.pbar.set_postfix(**self._format_metrics())
        self.pbar.update()


class ClusteringEarlyStopping:
    def __init__(self, clustering_model, threshold=0.01):
        self.clustering_model = clustering_model
        self.n_clusters = clustering_model.n_clusters
        self.threshold = threshold
        self.eps = 1e-9
        self.previous_centroids = self.clustering_model.w.cpu().detach().clone().numpy()

    def __call__(self, engine):
        w_prev = np.reshape(self.previous_centroids, (self.n_clusters, -1))

        current_centroids = self.clustering_model.w.cpu().detach().numpy()
        w_curr = np.reshape(current_centroids, (self.n_clusters, -1))

        deviation = np.abs((w_curr - w_prev).sum(-1) / (w_prev.sum(-1) + self.eps))

        self.w_prev = w_curr
        self.previous_centroids = current_centroids
        print(deviation)
        print(w_prev)
        print(w_curr)

        if (deviation <= self.threshold).all():
            engine.terminate()

            print('Early Stopping: Stopping criterion has been reached!')
            print('Mean Centroid Deviation {:.2f}%'.format(deviation.mean() * 100))


def add_handlers(trainer, evaluator, data, models_dict, cfg):
    """

    :param trainer: ignite trainer object
    :param evaluator: ignite evaluator object
    :param data: tuple containing train and test dataloader
    :param models_dict: dict containing all models & optimizers to save
    :param cfg: configuration dict
    """
    train_loader, test_loader = data

    # add progressbar
    progbar = ProgressBar(trainer, train_loader)
    trainer.add_event_handler(
        event_name=Events.ITERATION_COMPLETED, handler=progbar)

    # initialize checkpoint savings funtion
    checkpoint = ModelCheckpoint(
        cfg.DIRS.CHKP_DIR,
        cfg.DIRS.CHKP_PREFIX,
        require_empty=False,
        save_interval=1,
        n_saved=100000,
        save_as_state_dict=True)

    writer = Tensorboard(create_dir(cfg.DIRS.CHKP_DIR, 'summaries'))
    trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED,
                              handler=writer)
    evaluator.add_event_handler(event_name=Events.ITERATION_COMPLETED,
                                handler=writer)

    # if models were loaded, resume training from left off epoch
    # otherwise start at epoch 0.
    @trainer.on(Events.STARTED)
    def epoch_start(engine):

        if (cfg.SOLVER.RESUME_EPOCH != '0') and \
                (cfg.SOLVER.COMPLETED_EPOCHS != 0):
            mess = 'TRAINING COMPLETE' if \
                ((cfg.SOLVER.COMPLETED_EPOCHS - cfg.SOLVER.EPOCHS) >= 0) \
                else 'RESUME TRAINING'

            engine.state.iteration = cfg.SOLVER.TRAINER_ITERATION
            engine.state.epoch = checkpoint._iteration = cfg.SOLVER.COMPLETED_EPOCHS

            print(' --- LOADED MODEL FOR EPOCH: {comp_epochs} / {conf} ---\
                 \n --------- {mess} ---------'.format(
                comp_epochs=cfg.SOLVER.COMPLETED_EPOCHS,
                conf=cfg.SOLVER.EPOCHS, mess=mess))

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_models(engine):

        cfg.SOLVER.COMPLETED_EPOCHS = engine.state.epoch
        cfg.SOLVER.TRAINER_ITERATION = engine.state.iteration

        if cfg.SOLVER.COMPLETED_EPOCHS % cfg.MODEL.SAVE_INTERVAL == 0:
            # checkpoint only counts nr of checkpoint calls, not epochs
            checkpoint._iteration = cfg.SOLVER.COMPLETED_EPOCHS - 1
            checkpoint(engine, models_dict)
            save_ignite_params(engine, engine_name='trainer', cfg=cfg)

    @trainer.on(Events.EPOCH_COMPLETED)
    def classification_validation(engine):
        cfg.RESULTS.LATENTS, cfg.RESULTS.CLF_ACC, cfg.RESULTS.MEAN_DISTANCE, \
        cfg.RESULTS.SMOOTHNESS, cfg.RESULTS.CLUSTER_ACC = [], [], [], [], []

        print('--- Evaluating model on validation set ---')
        evaluator.run(test_loader)

        clf_acc_str = calc_mean_non_empty('clf_acc', cfg.RESULTS.CLF_ACC)
        cluster_acc_str = calc_mean_non_empty('cluster_acc', cfg.RESULTS.CLUSTER_ACC)
        mean_dist_str = calc_mean_non_empty('mean_distance', cfg.RESULTS.MEAN_DISTANCE)
        smoothness_str = calc_mean_non_empty('smoothness', cfg.RESULTS.SMOOTHNESS)

        print('{clf}{cluster}{mean_dist}{smooth}'.format(
            clf=clf_acc_str,
            cluster=cluster_acc_str,
            mean_dist=mean_dist_str,
            smooth=smoothness_str))

    @evaluator.on(Events.STARTED)
    def continue_validation(engine):
        # create dict keys at first epoch
        if 'EVAL_ITERATION' not in cfg.SOLVER.keys():
            cfg.SOLVER.EVAL_ITERATION = 0
            cfg.SOLVER.EVAL_EPOCH = 0
        else:
            # after each iteration, dict gets updated
            # needed so that it continues counter at start of
            # every eval run (bc for every validation run
            # the evaluator is newly initialized)
            engine.state.iteration = cfg.SOLVER.EVAL_ITERATION

            # always put to 0 to run evaluation once
            # can't specify max epochs bc it would run
            # validation set several times after each other
            engine.state.epoch = 0

    @evaluator.on(Events.EPOCH_COMPLETED)
    def save_eval_state(engine):
        # save iteration after each validation run
        # continue at this counter for next run
        # (as number otherwise resets),
        # save number of eval epochs for saving/loading params
        cfg.SOLVER.EVAL_ITERATION = engine.state.iteration
        cfg.SOLVER.EVAL_EPOCH += 1

        if cfg.SOLVER.COMPLETED_EPOCHS % cfg.MODEL.SAVE_INTERVAL == 0:
            save_ignite_params(engine, engine_name='eval', cfg=cfg)

    return trainer, evaluator
