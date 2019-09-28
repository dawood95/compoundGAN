import torch
import numpy as np

from copy import deepcopy
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F

class Trainer:

    def __init__(self, data, model, optimizer, scheduler, logger, cuda=False):

        self.train_loader = data[0]
        self.val_loader   = data[1]

        self.train_datagen = self.load_data(self.train_loader)
        self.val_datagen   = self.load_data(self.val_loader)

        self.enc = model[0]
        self.gen = model[1]
        self.dis = model[2]

        self.enc_optim = optimizer[0]
        self.gen_optim = optimizer[1]
        self.dis_optim = optimizer[2]

        self.scheduler = scheduler
        self.logger    = logger
        self.cuda      = cuda

        self.epoch      = 0
        self.train_step = 0

        self.train_seq_len = 2

        self.vae_epoch_steps = 1_000
        self.vae_train_step  = 0
        self.vae_val_step    = 0

    def load_data(self, dataloader):
        while True:
            for data in dataloader:
                yield data

    def run(self, num_epoch):
        for i in range(num_epoch):
            # Increment epoch
            self.epoch += 1
            print('EPOCH #%d\n'%(self.epoch))

            with self.logger.experiment.train():
                # Train VAE
                kl_loss, pred_loss = self.train_vae()
                self.logger.experiment.log_metrics({
                    'kl': kl_loss,
                    'pred': pred_loss
                }, prefix='VAE_total', step=(self.epoch))

            with self.logger.experiment.validate():
                kl_loss, pred_loss = self.val_vae()
                self.logger.experiment.log_metrics({
                    'kl': kl_loss,
                    'pred': pred_loss
                }, prefix='VAE_total', step=(self.epoch))

            self.save()
            if self.scheduler: self.scheduler.step()

    def save(self):
        data = {
            'enc_state_dict' : self.enc.state_dict(),
            'gen_state_dict' : self.gen.state_dict(),
            'dis_state_dict' : self.dis.state_dict(),
        }
        self.logger.save('model_%d.weights'%self.epoch, data)

    def train_discriminator(self):
        raise NotImplementedError

    def train_generator(self):
        raise NotImplementedError

    def train_vae(self):

        self.enc.train()
        self.gen.train()

        total_kl_loss = 0
        total_pred_loss = 0

        pbar = tqdm(range(self.vae_epoch_steps))
        pbar.set_description('VAE Train')

        for i in pbar:
            G, atom_y, bond_y = next(self.train_datagen)
            if self.cuda:
                G.to(torch.device('cuda:0'))
                atom_y = atom_y.cuda(non_blocking=True)
                bond_y = bond_y.cuda(non_blocking=True)

            # Generate mu_x, logvar_x
            mu, logvar = self.enc(G)
            z = self.enc.reparameterize(mu, logvar)

            # Run compound generator and accumulate loss
            G_pred, pred_loss = self.gen.calc_loss(z, atom_y, bond_y, self.train_seq_len)

            if pred_loss < 0.10:
                self.train_seq_len = 2 * self.train_seq_len

            # Calculate KL-Divergence Loss
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Weighting KL loss
            kl_factor = 1e-2
            loss = kl_factor*kl_loss + pred_loss

            self.enc_optim.zero_grad()
            self.gen_optim.zero_grad()

            loss.backward()

            # clip gradients
            torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.enc.parameters(), 1)

            self.enc_optim.step()
            self.gen_optim.step()

            self.vae_train_step += 1

            kl_loss = float(kl_loss.item())
            pred_loss = float(pred_loss.item())

            total_kl_loss += kl_loss
            total_pred_loss += pred_loss

            if (i+1) % 10 == 0:
                self.logger.experiment.log_metrics({
                    'kl_loss': kl_loss,
                    'pred_loss': pred_loss
                }, step=self.vae_train_step)
                tqdm.write(
                    'VAE Train [%4d : %4d] | KL Loss=[%6.5f] | Pred Loss=[%6.5f]'%
                    (i+1, self.train_seq_len, kl_loss, pred_loss)
                )

            del mu, logvar, z, G_pred, pred_loss, kl_loss, loss

        total_kl_loss /= i+1
        total_pred_loss /= i+1

        print('VAE Train Total | Avg KL Loss=[%6.5f] | Avg Pred Loss=[%6.5f]'%
              (total_kl_loss, total_pred_loss))

        return total_kl_loss, total_pred_loss


    @torch.no_grad()
    def val_vae(self):

        self.enc.eval()
        self.gen.eval()

        total_kl_loss = 0
        total_pred_loss = 0

        pbar = tqdm(range(min(self.vae_epoch_steps, len(self.val_loader))))
        pbar.set_description('VAE Val')

        for i in pbar:
            G, atom_y, bond_y = next(self.val_datagen)
            if self.cuda:
                G.to(torch.device('cuda:0'))
                atom_y = atom_y.cuda(non_blocking=True)
                bond_y = bond_y.cuda(non_blocking=True)

            # Generate mu_x, logvar_x
            mu, logvar = self.enc(G)
            z = self.enc.reparameterize(mu, logvar, no_noise=True)

            # Run compound generator and accumulate loss
            G_pred, pred_loss = self.gen.calc_loss(z, atom_y, bond_y)

            # Calculate KL-Divergence Loss
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            self.vae_val_step += 1

            kl_loss = float(kl_loss.item())
            pred_loss = float(pred_loss.item())

            total_kl_loss += kl_loss
            total_pred_loss += pred_loss

            if (i+1) % 10 == 0:
                self.logger.experiment.log_metrics({
                    'kl_loss': kl_loss,
                    'pred_loss': pred_loss
                }, step=self.vae_val_step)
                tqdm.write(
                    'VAE Val [%4d] | KL Loss=[%6.5f] | Pred Loss=[%6.5f]'%
                    (i+1, kl_loss, pred_loss)
                )

            del mu, logvar, z, G_pred, \
                pred_loss, kl_loss

        total_kl_loss /= i+1
        total_pred_loss /= i+1

        print('VAE Val Total | Avg KL Loss=[%6.5f] | Avg Pred Loss=[%6.5f]'%
              (total_kl_loss, total_pred_loss))

        return total_kl_loss, total_pred_loss
