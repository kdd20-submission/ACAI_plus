import numpy as np
import torch
import torch.nn.functional as F
from ignite.engine import Engine
from acaiplus.utils import interpolation
from acaiplus.utils import evaluation
from acaiplus.utils import images
from acaiplus.utils.layers import adjust_learning_rate
from acaiplus.utils.utils import L2, lerp, swap_halves
from acaiplus.utils.clustering import vade_cluster_acc


def create_baseline_trainer(models, cfg):
    models['encoder'].to(cfg.MODEL.DEVICE)
    models['decoder'].to(cfg.MODEL.DEVICE)
    models['classifier'].to(cfg.MODEL.DEVICE)

    def _step(engine, batch):
        models['encoder'].train()
        models['decoder'].train()
        models['classifier'].train()

        # ---------------------------
        # 1. foward
        x, target = batch[0].to(cfg.MODEL.DEVICE), batch[1].to(cfg.MODEL.DEVICE)
        z = models['encoder'](x)
        out = models['decoder'](z)
        y_pred = models['classifier'](x)

        # ---------------------------
        # 2. backprop
        loss_ae = F.mse_loss(out, x)
        models['opt_ae'].zero_grad()
        loss_ae.backward(retain_graph=True)
        models['opt_ae'].step()
        losses = {'loss_AE': loss_ae.item()}

        loss_clf = F.nll_loss(y_pred, target)
        models['opt_clf'].zero_grad()
        loss_clf.backward()
        models['opt_clf'].step()

        # return all losses to save with handlers
        return losses

    return Engine(_step)


def create_acai_trainer(models, cfg):
    models['encoder'].to(cfg.MODEL.DEVICE)
    models['decoder'].to(cfg.MODEL.DEVICE)
    models['classifier'].to(cfg.MODEL.DEVICE)
    models['discriminator'].to(cfg.MODEL.DEVICE)

    def _step(engine, batch):
        models['discriminator'].train()
        models['encoder'].train()
        models['decoder'].train()
        models['classifier'].train()
        losses = {'loss_AE': None}

        # ---------------------------
        # 1. foward
        x, target = batch[0].to(cfg.MODEL.DEVICE), batch[1].to(cfg.MODEL.DEVICE)
        z = models['encoder'](x)
        out = models['decoder'](z)
        y_pred = models['classifier'](x)
        disc = models['discriminator'](torch.lerp(out, x, cfg.MODEL.REG))

        # (batch, c, h, w) -> (batch, 1, 1, 1)
        alpha_dimensions = [z.shape[0]] + [1] * (z.dim() - 1)
        alpha = torch.rand(alpha_dimensions).to(cfg.MODEL.DEVICE) / 2
        flipped_idx = [i for i in range(z.size(0) - 1, -1, -1)]
        flipped_idx = torch.LongTensor(flipped_idx)
        flipped_z = z.index_select(0, flipped_idx.to(cfg.MODEL.DEVICE))
        z_mix = lerp(z, flipped_z, alpha)

        out_mix = models['decoder'](z_mix)
        disc_mix = models['discriminator'](out_mix)

        # ---------------------------
        # 2. backprop
        loss_ae_mse = F.mse_loss(out, x)
        loss_ae_mse_two = F.mse_loss(out_mix, x) * cfg.MODEL.ADDED_CONSTR
        loss_ae_l2 = L2(disc_mix) * cfg.MODEL.ADVWEIGHT
        loss_ae = loss_ae_mse + loss_ae_mse_two + loss_ae_l2
        models['opt_ae'].zero_grad()
        loss_ae.backward(retain_graph=True)
        models['opt_ae'].step()

        loss_disc_mse = F.mse_loss(disc_mix, alpha.reshape(-1))
        loss_disc_l2 = L2(disc)
        loss_disc = loss_disc_mse + loss_disc_l2
        models['opt_d'].zero_grad()
        loss_disc.backward(retain_graph=True)
        models['opt_d'].step()

        losses['loss_disc'] = loss_disc.item()
        losses['loss_disc_mse'] = loss_disc_mse.item()
        losses['loss_disc_l2'] = loss_disc_l2.item()
        losses['loss_AE'] = loss_ae_mse.item()  # loss_ae.item()
        losses['loss_ae_l2'] = loss_ae_l2.item()
        losses['loss_ae_two'] = loss_ae_mse_two.item()

        loss_clf = F.nll_loss(y_pred, target)
        models['opt_clf'].zero_grad()
        loss_clf.backward()
        models['opt_clf'].step()

        return losses

    return Engine(_step)


@torch.no_grad()
def create_evaluator(models, plot_data, cfg):
    models['encoder'].to(cfg.MODEL.DEVICE)
    models['classifier'].to(cfg.MODEL.DEVICE)
    models['decoder'].to(cfg.MODEL.DEVICE)

    def _inference(engine, batch):
        models['encoder'].eval()
        models['decoder'].eval()
        models['classifier'].eval()

        x, target = batch[0].to(cfg.MODEL.DEVICE), batch[1].to(cfg.MODEL.DEVICE)
        y_pred = models['classifier'](x)
        loss_clf = F.nll_loss(y_pred, target)

        pred = y_pred.data.max(1)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        training_acc_clf = 100. * correct / x.shape[0]

        loss_dict = {'loss_clf': loss_clf.item(),
                     'acc_clf': training_acc_clf.item()}
        cfg.RESULTS.CLF_ACC.append(training_acc_clf.item())
        loss_dict['clf_acc'] = training_acc_clf.item()

        if cfg.DATA.DATASET == 'lines':
            interpolated_data = interpolation.interpolate_lines_benchmark(
                encoder=models['encoder'],
                decoder=models['decoder'],
                x=x,
                cfg=cfg)
            mean_distance, mean_smoothness = evaluation.line_eval(interpolated_data)

            mean_distance /= interpolated_data.shape[0]
            mean_smoothness /= interpolated_data.shape[0]

            loss_dict.update({'mean_distance': mean_distance,
                              'mean_smoothness': mean_smoothness})
            cfg['mean_distance'].append(mean_distance)
            cfg['smoothness'].append(mean_smoothness)

        # save data interpolation image
        if engine.state.iteration % cfg.DATA.BATCH_SIZE == 0:
            # always take the same first batch of the test_loader
            # to have the same interpolation images between models
            plot_d = iter(plot_data)
            for index in range(cfg.VIZ.PLOT_DATA_INDEX):
                x = next(plot_d)[0].to(cfg.MODEL.DEVICE)
            z = models['encoder'](x)
            out = models['decoder'](z)

            # calculate mean distance + smoothness for dataset
            interpolated_data = interpolation.interpolate_data(
                encoder=models['encoder'],
                decoder=models['decoder'],
                x=x,
                cfg=cfg)

            _, _ = images.save_images(x=x,
                                      out=out,
                                      interpolated_data=interpolated_data,
                                      cfg=cfg,
                                      temporal=cfg.DATA.TEMPORAL_BOOL)
            if cfg.DATA.DATASET == 'mnist':
                img = interpolation.status(x=x,
                                           encoder=models['encoder'],
                                           decoder=models['decoder'],
                                           args=cfg)

                images.imsave(img=img * 255,
                              model_name=cfg.DIRS.CHKP_PREFIX,
                              work_dir=cfg.DIRS.FIGURES,
                              save=True,
                              save_ext=f'epoch_{cfg.SOLVER.EVAL_EPOCH + 1}')

                images.save_latent_digits(encoder=models['encoder'],
                                          decoder=models['decoder'],
                                          cfg=cfg)

                if cfg.VIZ.SAVE_SVD_PLOTS:
                    _ = images.save_svd_plots(encoder=models['encoder'],
                                              folder=cfg.DIRS.CHKP_DIR,
                                              save_extend=f'epoch_{cfg.SOLVER.EVAL_EPOCH + 1}',
                                              cfg=cfg)

        return loss_dict

    return Engine(_inference)



def create_idec_baseline_finetuning_trainer(models, cfg):
    models['encoder'].to(cfg.MODEL.DEVICE)
    models['decoder'].to(cfg.MODEL.DEVICE)
    models['clustering'].to(cfg.MODEL.DEVICE)
    models['classifier'].to(cfg.MODEL.DEVICE)

    def _step(engine, batch):
        models['encoder'].train()
        models['decoder'].train()
        models['clustering'].train()
        models['classifier'].train()

        losses = {'loss_AE': None,
                  'loss_clustering': None}

        x, target = batch[0].to(cfg.MODEL.DEVICE), batch[1].to(cfg.MODEL.DEVICE)
        idx = batch[2].to(cfg.MODEL.DEVICE)

        z = models['encoder'](x)
        q = models['clustering'](z)
        out = models['decoder'](z)
        y_pred = models['classifier'](x)

        # ---------------------------
        # 2. backprop
        loss_ae_mse = F.mse_loss(out, x)
        loss_clust = F.kl_div(q.log(), models['clustering'].p[idx]) * cfg.MODEL.IDEC.GAMMA
        loss_ae = loss_ae_mse + loss_clust

        models['opt_combined'].zero_grad()
        loss_ae.backward()
        models['opt_combined'].step()

        # adding values to handler for tensorboard
        losses['loss_AE'] = loss_ae_mse.item()
        losses['loss_clustering'] = loss_clust.item()
        losses['loss_combined'] = loss_ae.item()

        # classifier loss
        loss_clf = F.nll_loss(y_pred, target)

        models['opt_clf'].zero_grad()
        loss_clf.backward()
        models['opt_clf'].step()

        # return all losses to save with handlers
        return losses

    return Engine(_step)


def create_idec_acai_finetuning_trainer(models, cfg):
    models['encoder'].to(cfg.MODEL.DEVICE)
    models['decoder'].to(cfg.MODEL.DEVICE)
    models['clustering'].to(cfg.MODEL.DEVICE)
    models['classifier'].to(cfg.MODEL.DEVICE)
    models['discriminator'].to(cfg.MODEL.DEVICE)

    def _step(engine, batch):
        models['encoder'].train()
        models['decoder'].train()
        models['clustering'].train()
        models['classifier'].train()
        models['discriminator'].train()

        losses = {'loss_AE': None,
                  'loss_clustering': None}

        x, target = batch[0].to(cfg.MODEL.DEVICE), batch[1].to(cfg.MODEL.DEVICE)
        idx = batch[2].to(cfg.MODEL.DEVICE)

        z = models['encoder'](x)
        q = models['clustering'](z)
        out = models['decoder'](z)
        y_pred = models['classifier'](x)

        disc = models['discriminator'](torch.lerp(out, x, cfg.MODEL.REG))

        # (batch, c, h, w) -> (batch, 1, 1, 1)
        alpha_dimensions = [z.shape[0]] + [1] * (z.dim() - 1)
        alpha = torch.rand(alpha_dimensions).to(cfg.MODEL.DEVICE) / 2
        flipped_idx = [i for i in range(z.size(0) - 1, -1, -1)]
        flipped_idx = torch.LongTensor(flipped_idx)
        flipped_z = z.index_select(0, flipped_idx.to(cfg.MODEL.DEVICE))
        z_mix = lerp(z, flipped_z, alpha)

        # discriminator tries to predict alpha
        out_mix = models['decoder'](z_mix)
        disc_mix = models['discriminator'](out_mix)

        # ---------------------------
        # 2. backprop
        loss_ae_mse = F.mse_loss(out, x)
        loss_ae_mse_two = F.mse_loss(out_mix, x)
        loss_ae_l2 = L2(disc_mix)

        loss_clust = F.kl_div(q.log(), models['clustering'].p[idx]) * cfg.MODEL.IDEC.GAMMA
        loss_ae = loss_ae_mse + loss_clust + loss_ae_mse_two + loss_ae_l2

        models['opt_combined'].zero_grad()
        loss_ae.backward(retain_graph=True)
        models['opt_combined'].step()

        # backprop of discriminator loss
        loss_disc_mse = F.mse_loss(disc_mix, alpha.reshape(-1))
        loss_disc_l2 = L2(disc)
        loss_disc = loss_disc_mse + loss_disc_l2

        models['opt_d'].zero_grad()
        loss_disc.backward(retain_graph=True)
        models['opt_d'].step()

        # adding values to handler for tensorboard
        losses['loss_AE'] = loss_ae_mse.item()
        losses['loss_clustering'] = loss_clust.item()
        losses['loss_ae_l2'] = loss_ae_l2.item()
        losses['loss_ae_two'] = loss_ae_mse_two.item()
        losses['loss_combined'] = loss_ae.item()
        losses['loss_disc_mse'] = loss_disc_mse.item()
        losses['loss_disc_l2'] = loss_disc_l2.item()
        losses['loss_disc'] = loss_disc.item()

        # classifier loss
        loss_clf = F.nll_loss(y_pred, target)

        models['opt_clf'].zero_grad()
        loss_clf.backward()
        models['opt_clf'].step()

        # return all losses to save with handlers
        return losses

    return Engine(_step)


def create_vae_baseline_trainer(models, custom_loss, cfg):
    # send models to GPU if available
    models['encoder'].to(cfg.MODEL.DEVICE)
    models['decoder'].to(cfg.MODEL.DEVICE)
    models['classifier'].to(cfg.MODEL.DEVICE)

    def _step(engine, batch):
        models['encoder'].train()
        models['classifier'].train()
        models['decoder'].train()

        # ---------------------------
        # 1. foward

        x, target = batch[0].to(cfg.MODEL.DEVICE), batch[1].to(cfg.MODEL.DEVICE)
        z_sample, mu, logvar = models['encoder'](x)
        out = models['decoder'](z_sample)
        y_pred = models['classifier'](x)

        # ---------------------------
        # 2. backprop

        # AE loss
        loss_ae = custom_loss(recon_x=out,
                              x=x,
                              mu=mu,
                              logvar=logvar)

        models['opt_ae'].zero_grad()
        loss_ae.backward(retain_graph=True)
        models['opt_ae'].step()

        # classifier loss
        loss_clf = F.nll_loss(y_pred, target)

        models['opt_clf'].zero_grad()
        loss_clf.backward()
        models['opt_clf'].step()

        return {'loss_AE': loss_ae.item()}

    return Engine(_step)


def create_vae_acai_trainer(models, custom_loss, cfg):
    models['encoder'].to(cfg.MODEL.DEVICE)
    models['decoder'].to(cfg.MODEL.DEVICE)
    models['classifier'].to(cfg.MODEL.DEVICE)
    models['discriminator'].to(cfg.MODEL.DEVICE)

    def _step(engine, batch):
        models['encoder'].train()
        models['classifier'].train()
        models['decoder'].train()
        models['discriminator'].train()

        losses = {'loss_AE': None}

        # ---------------------------
        # 1. foward

        x, target = batch[0].to(cfg.MODEL.DEVICE), batch[1].to(cfg.MODEL.DEVICE)
        z, mu, logvar = models['encoder'](x)
        out = models['decoder'](z)
        y_pred = models['classifier'](x)

        z = z.view(z.shape[0], cfg.MODEL.LATENT, cfg.MODEL.LATENT_WIDTH, cfg.MODEL.LATENT_WIDTH)
        disc = models['discriminator'](torch.lerp(out, x, cfg.MODEL.REG))
        alpha = torch.rand(cfg.DATA.BATCH_SIZE, 1, 1, 1).to(cfg.MODEL.DEVICE) / 2
        z_mix = lerp(z, swap_halves(z), alpha)

        # discriminator tries to predict alpha
        out_mix = models['decoder'](z_mix)
        disc_mix = models['discriminator'](out_mix)

        # ---------------------------
        # 2. backprop

        # AE loss
        loss_ae_mse = custom_loss(out, x, mu, logvar)
        loss_ae_mse_two = F.mse_loss(out_mix, x)
        loss_ae_l2 = L2(disc_mix) * cfg.MODEL.ADVWEIGHT
        loss_ae = loss_ae_mse + loss_ae_mse_two + loss_ae_l2

        models['opt_ae'].zero_grad()
        loss_ae.backward(retain_graph=True)
        models['opt_ae'].step()

        # backprop of discriminator loss
        loss_disc_mse = F.mse_loss(disc_mix, alpha.reshape(-1))
        loss_disc_l2 = L2(disc)
        loss_disc = loss_disc_mse + loss_disc_l2

        models['opt_d'].zero_grad()
        loss_disc.backward(retain_graph=True)
        models['opt_d'].step()

        # adding loss values to engine for handlers
        losses['loss_ae_l2'] = loss_ae_l2.item()
        losses['loss_disc_mse'] = loss_disc_mse.item()
        losses['losss_disc_l2'] = loss_disc_l2.item()
        losses['loss_disc'] = loss_disc.item()
        losses['loss_AE'] = loss_ae.item()

        # classifier loss
        loss_clf = F.nll_loss(y_pred, target)

        models['opt_clf'].zero_grad()
        loss_clf.backward()
        models['opt_clf'].step()

        return losses

    return Engine(_step)
