import torch
import numpy as np
from tqdm import tqdm

from torch import nn
from torch.nn import functional as F

class Trainer:

    def __init__(self, data, model, optimizer, scheduler, logger,
                 device=torch.device('cpu'), is_master=False):

        self.train_loader = data[0]
        self.val_loader   = data[1]

        self.model     = model
        self.optim     = optimizer
        self.scheduler = scheduler
        self.logger    = logger
        self.device    = device
        self.is_master = is_master

        self.epoch        = 0
        self.seq_len      = 4
        self.log_step     = 25
        self.prior_factor = 1e-2
        self.recon_thresh = 0.20

        self.vae_train_step  = 0
        self.vae_val_step    = 0

        self.temp_weights_file = 'model.weights' if is_master else None

    def write(self, x):
        if self.is_master: tqdm.write(x)

    def run(self, num_epoch):
        for i in range(num_epoch):

            self.write('EPOCH #%d\n'%(self.epoch))

            with self.logger.experiment.train():
                # Train VAE
                recon_loss, entropy_loss, prior_loss = self.train_vae()
                self.logger.experiment.log_metrics({
                    'recon'   : recon_loss,
                    'entropy' : entropy_loss,
                    'prior'   : prior_loss
                }, prefix='VAE_total', step=(self.epoch))

            if recon_loss < self.recon_thresh:
                self.seq_len += 4

            with self.logger.experiment.validate():
                recon_loss, entropy_loss, prior_loss = self.val_vae()
                self.logger.experiment.log_metrics({
                    'recon'   : recon_loss,
                    'entropy' : entropy_loss,
                    'prior'   : prior_loss
                }, prefix='VAE_total', step=(self.epoch))

            self.save(temp=self.temp_weights_file)

            if self.scheduler:
                self.scheduler.step()

            # Increment epoch
            self.epoch += 1

            '''
            if self.epoch - last_update >= update_thresh:
                self.seq_len += 4
                last_update   = self.epoch
            '''

            '''
            if self.seq_len > 32:
                self.prior_factor = self.prior_factor * 1.1
                self.prior_factor = min(1.0, self.prior_factor)
            '''

    def save(self, **kwargs):
        data = {
            'epoch'      : self.epoch,
            'seq_len'    : self.seq_len,
            'parameters' : self.model.state_dict(),
        }
        self.logger.save('model_%d.weights'%self.epoch, data, **kwargs)

    def train_vae(self):

        log_seq_len = -1 if self.seq_len == np.inf else self.seq_len

        self.model.train()

        total_recon_loss   = 0
        total_entropy_loss = 0
        total_prior_loss   = 0

        if self.is_master:
            data_loader = tqdm(self.train_loader)
        else:
            data_loader = self.train_loader

        for i, data in enumerate(data_loader):
            self.optim.zero_grad()

            G, atom_y, bond_y = data

            G.to(self.device)
            atom_y = atom_y.to(self.device)
            bond_y = bond_y.to(self.device)

            losses = self.model.module.calc_loss(G, atom_y, bond_y, self.seq_len)
            recon_loss, entropy_loss, prior_loss = losses
            loss = recon_loss
            loss = loss + self.prior_factor*prior_loss
            loss = loss + self.prior_factor*entropy_loss

            loss.backward()

            # clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optim.step()

            self.vae_train_step += 1

            recon_loss   = float(recon_loss.item())
            entropy_loss = float(entropy_loss.item())
            prior_loss   = float(prior_loss.item())

            total_recon_loss   += recon_loss
            total_entropy_loss += entropy_loss
            total_prior_loss   += prior_loss

            if (i+1) % self.log_step == 0:
                self.logger.experiment.log_metrics({
                    'recon_loss'   : recon_loss,
                    'entropy_loss' : entropy_loss,
                    'prior_loss'   : prior_loss,
                }, step=self.vae_train_step)
                self.write(
                    'VAE Train [%4d : %4d] | '
                    'Reconstruction Loss=[%6.5f] | '
                    'Entropy Loss=[%6.5f] | '
                    'Prior Loss=[%6.5f]'%
                    (i+1, log_seq_len,
                     recon_loss, entropy_loss, prior_loss)
                )

        total_recon_loss   /= (i + 1)
        total_prior_loss   /= (i + 1)
        total_entropy_loss /= (i + 1)

        self.write(
            'VAE Train Total | '
            'Reconstruction Loss=[%6.5f] | '
            'Entropy Loss=[%6.5f] | '
            'Prior Loss=[%6.5f]'%
            (total_recon_loss, total_entropy_loss, total_prior_loss)
        )

        return total_recon_loss, total_entropy_loss, total_prior_loss

    @torch.no_grad()
    def val_vae(self):

        log_seq_len = -1 if self.seq_len == np.inf else self.seq_len

        self.model.eval()

        total_recon_loss   = 0
        total_entropy_loss = 0
        total_prior_loss   = 0

        if self.is_master:
            data_loader = tqdm(self.val_loader)
        else:
            data_loader = self.val_loader

        for i, data in enumerate(data_loader):

            G, atom_y, bond_y = data

            G.to(self.device)
            atom_y = atom_y.to(self.device)
            bond_y = bond_y.to(self.device)

            losses = self.model.module.calc_loss(G, atom_y, bond_y, self.seq_len)
            recon_loss, entropy_loss, prior_loss = losses
            loss = recon_loss + entropy_loss + prior_loss

            self.vae_val_step += 1

            recon_loss   = float(recon_loss.item())
            entropy_loss = float(entropy_loss.item())
            prior_loss   = float(prior_loss.item())

            total_recon_loss   += recon_loss
            total_entropy_loss += entropy_loss
            total_prior_loss   += prior_loss

            if (i+1) % self.log_step == 0:
                self.logger.experiment.log_metrics({
                    'recon_loss'   : recon_loss,
                    'entropy_loss' : entropy_loss,
                    'prior_loss'   : prior_loss,
                }, step=self.vae_val_step)
                self.write(
                    'VAE Val [%4d : %4d] | '
                    'Reconstruction Loss=[%6.5f] | '
                    'Entropy Loss=[%6.5f] | '
                    'Prior Loss=[%6.5f]'%
                    (i+1, log_seq_len,
                     recon_loss, entropy_loss, prior_loss)
                )

        total_recon_loss   /= (i + 1)
        total_prior_loss   /= (i + 1)
        total_entropy_loss /= (i + 1)

        self.write(
            'VAE Val Total | '
            'Reconstruction Loss=[%6.5f] | '
            'Entropy Loss=[%6.5f] | '
            'Prior Loss=[%6.5f]'%
            (total_recon_loss, total_entropy_loss, total_prior_loss)
        )

        return total_recon_loss, total_entropy_loss, total_prior_loss
