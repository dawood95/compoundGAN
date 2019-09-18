import torch
import numpy as np

from tqdm import tqdm
from torch import nn
from torch.nn import functional as F

class Trainer:

    def __init__(self, data, model, optimizer, scheduler, logger, cuda=False):

        self.dataloader = data

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
        self.vae_step   = 0

        # 10,000 compounds = 1 epoch
        self.vae_epoch_size = 1_000

    def run(self, num_epoch):
        for i in range(num_epoch):
            # Increment epoch
            self.epoch += 1
            print('EPOCH #%d\n'%(self.epoch))

            with self.logger.experiment.train():
                # Train discriminator
                for _ in range(1):
                    break
                    dis_acc, dis_loss = self.train_discriminator()

                # Train VAE
                vae_steps = 1
                for step in range(vae_steps):
                    kl_loss, pred_loss = self.train_vae()
                    self.logger.experiment.log_metrics({
                        'kl': kl_loss,
                        'pred': pred_loss
                    }, prefix='VAE_total', step=(self.epoch*vae_steps + step))

                # Train generator
                for _ in range(1):
                    break
                    gan_loss = self.train_generator()

            self.save()

            if self.scheduler:
                self.scheduler.step()

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
        pbar = tqdm(enumerate(self.dataloader), total=self.vae_epoch_size)
        pbar.set_description('VAE Train')
        for i, (G, atom_y, bond_y) in pbar:

            if self.cuda:
                G.to(torch.device('cuda:0'))
                atom_y = atom_y.cuda(non_blocking=True)
                bond_y = [b.cuda(non_blocking=True) for b in bond_y]

            # Generate mu_x, logvar_x
            mu, logvar = self.enc(G)
            z = self.enc.reparameterize(mu, logvar)

            # Run compound generator and accumulate loss
            G_pred, pred_loss = self.gen.calc_loss(z, atom_y, bond_y)

            # Calculate KL-Divergence Loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Weighting KL loss 
            e_max_10 = min(self.epoch - 10, 10)
            loss = 1e-7*e_max_10*kl_loss + pred_loss

            self.enc_optim.zero_grad()
            self.gen_optim.zero_grad()
            loss.backward()
            self.enc_optim.step()
            self.gen_optim.step()

            self.vae_step += 1
            total_kl_loss += kl_loss.item()
            total_pred_loss += pred_loss.item()

            if (i+1) % 10 == 0:
                self.logger.experiment.log_metrics({
                    'kl_loss': kl_loss.item(),
                    'pred_loss': pred_loss.item()
                }, step=self.vae_step)
                tqdm.write(
                    'VAE Train [%4d] | KL Loss=[%.5f] | Pred Loss=[%.5f]'%
                    (i+1, kl_loss, pred_loss)
                )

            if i == self.vae_epoch_size: break

        total_kl_loss /= self.vae_epoch_size
        total_pred_loss /= self.vae_epoch_size

        print('VAE Train Total | Avg KL Loss=[%.5f] | Avg Pred Loss=[%.5f]'%
              (total_kl_loss, total_pred_loss))

        return total_kl_loss, total_pred_loss
