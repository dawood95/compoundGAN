import torch
import dgl
import numpy as np
from tqdm import tqdm

from torch import nn
from torch import autograd
from torch.nn import functional as F

class Trainer:

    def __init__(self, data, model, optimizer, scheduler, logger, cuda=False):

        self.dataloader            = data
        self.datagen               = self.load_data() # Infinite data loading
        self.G, self.D             = model
        self.optim_G, self.optim_D = optimizer
        self.scheduler             = scheduler
        self.batch_size            = data.batch_size
        self.logger                = logger
        self.cuda                  = cuda

        # 1 epoch can be N generator and M discriminator steps
        self.step_G = 0
        self.step_D = 0
        self.epoch  = 0

        # Max nodes for generator. Can be used for curriculum maybe
        # NOTE:
        # If using for curriculim, then remember to drop nodes
        # from GT discriminator input

        self.log_step    = 25
        self.max_G_steps = 1
        self.max_D_steps = 1
        self.num_iters   = 1000

        self.max_nodes = 50
        self.seq_len = 3
        self.clip_value = 0.01

        self.device = torch.device('cuda:0') if cuda else torch.device('cpu')

    def load_data(self):
        while True:
            for data in self.dataloader:
                yield data

    def run(self, num_epoch):
        last_update = 0
        for e in range(num_epoch):

            self.dataloader.dataset.max_seq_len = self.seq_len - 1

            tqdm.write('Epoch %d'%(self.epoch))

            total_loss = 0
            total_real = 0
            total_fake = 0

            for i in tqdm(range(self.num_iters)):
                loss, real, fake = self.train_discriminator()
                self.train_generator()
                total_real += real
                total_fake += fake
                total_loss += loss

            total_loss /= (self.num_iters*self.max_D_steps)
            total_real /= (self.num_iters*self.max_D_steps)
            total_fake /= (self.num_iters*self.max_D_steps)

            self.logger.experiment.log_metrics({
                'loss'      : total_loss,
                'fake score': total_fake,
                'real score': total_real,
                'sequence_length': self.seq_len,
            }, prefix='Total', step=(self.epoch))

            '''
            with self.logger.experiment.validate():
                kl_loss, pred_loss = self.val_vae()
                self.logger.experiment.log_metrics({
                    'kl': kl_loss,
                    'pred': pred_loss
                }, prefix='VAE_total', step=(self.epoch))
            '''

            self.save(temp='model.weights')

            if self.scheduler:
                self.scheduler.step()

            self.step_G = 0
            self.step_D = 0
            self.epoch += 1

            if self.epoch - last_update > 10:
                self.seq_len *= 2
                last_update = self.epoch
            self.seq_len = min(self.max_nodes, self.seq_len)

    def save(self, **kwargs):
        data = { 'G' : self.G.state_dict(),
                 'D' : self.D.state_dict(), }
        self.logger.save('model_%d.weights'%self.epoch, data, **kwargs)

    def detach_and_clone_graph(self, G):
        newG = dgl.batch([dgl.DGLGraph(_g._graph) for _g in dgl.unbatch(G)])
        newG.edata['feats'] = G.edata['feats'].detach().clone()
        newG.ndata['feats'] = G.ndata['feats'].detach().clone()
        newG.to(self.device)
        return newG

    def train_discriminator(self):
        self.G.eval()
        self.D.train()

        for p in self.D.parameters():
            p.requires_grad = True

        for i in range(self.max_D_steps):

            # Real batch
            real_G = next(self.datagen)
            real_G.to(self.device)

            # Fake batch
            with torch.no_grad():
                noise  = torch.randn(self.batch_size, 128).normal_(0, 1)
                noise  = noise.to(self.device)
                fake_G = self.G.create_graph(*self.G(noise, self.seq_len))

            # Pred
            pred_real = self.D(real_G).mean()
            pred_fake = self.D(fake_G).mean()

            # Loss ( decrease fake, increase real )
            loss = pred_fake - pred_real

            # Step optimizer
            self.optim_D.zero_grad()
            loss.backward()
            self.optim_D.step()

            # Clip parameters for WGAN
            for p in self.D.parameters():
                p.data.clamp_(-self.clip_value, self.clip_value)

            self.step_D += 1

            pred_fake = float(pred_fake.item())
            pred_real = float(pred_real.item())
            loss      = float(loss.item())

            total_fake += pred_fake
            total_real += pred_real
            total_loss += loss

            if self.step_D % self.log_step == 0:
                step = (self.epoch * self.max_D_steps) + self.step_D
                # Log this step's loss values
                self.logger.experiment.log_metrics({
                    'loss': loss,
                    'real score': pred_real,
                    'fake score': pred_fake,
                    'sequence length': self.seq_len
                }, step=step)
                # Print em out
                tqdm.write('Discriminator [%5d] [%3d] : Loss [%02.5e] | '
                           'Real [%02.5e] | Fake [%02.5e]'%
                           (self.step_D, self.seq_len, loss, pred_real, pred_fake))

        return total_loss, total_real, total_fake

    def train_generator(self):

        self.G.train()
        self.D.eval()

        # self.D.gcn.pool.train() # if using LSTM in discriminator

        for p in self.D.parameters():
            p.requires_grad = False

        total_loss = 0
        total_fake = 0

        for i in range(self.max_G_steps):

            # Generate noise
            noise = torch.randn(self.batch_size, 128).normal_(0, 1)
            noise = noise.to(self.device)

            # Generate graph from noise
            fake_G = self.G.create_graph(*self.G(noise, self.seq_len))

            # Find discriminator rating for generated graph
            pred_fake = self.D(fake_G)
            pred_fake = pred_fake.mean()

            # Loss
            loss = -1 * pred_fake

            # Step the optimizer for generator
            self.optim_G.zero_grad()
            loss.backward()

            # clip gradients for RNN stability
            # for p in list(filter(lambda p: p[1].grad is not None, self.G.named_parameters())):
            #     print(p[0], p[1].grad.data.norm(2).item())
            # torch.nn.utils.clip_grad_norm_(self.G.parameters(), 5)

            self.optim_G.step()

            self.step_G += 1

            loss = float(loss.item())
            pred_fake = float(pred_fake.item())

            total_fake += pred_fake
            total_loss += loss

            # if self.step_G % self.log_step == 0:
            #     step = (self.epoch * self.max_G_steps) + self.step_G
            #     self.logger.experiment.log_metrics({
            #         'generator_score': pred_fake,
            #     }, step=step)

        return total_loss, total_fake

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
            G_pred, pred_loss = self.gen.calc_loss(z, atom_y, bond_y)#, self.train_seq_len)

            # Calculate KL-Divergence Loss
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Weighting KL loss
            kl_factor = 5e-2
            loss = kl_factor*kl_loss + pred_loss

            self.enc_optim.zero_grad()
            self.gen_optim.zero_grad()

            loss.backward()

            # clip gradients
            torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(self.enc.parameters(), 5)

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

        if total_pred_loss < self.loss_thresh:
            self.train_seq_len += 8
            self.loss_thresh -= 0.01
            self.loss_thresh = max(self.loss_thresh, 0.05)
        else:
            self.train_seq_len -= 1
            self.loss_thresh += 0.005

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
