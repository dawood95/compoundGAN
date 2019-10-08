import torch
import dgl
import numpy as np
from tqdm import tqdm

from torch import nn
from torch import autograd
from torch.nn import functional as F

class Trainer:

    def __init__(self, data, model, optimizer, scheduler, logger, cuda=False):

        self.datagen               = self.load_data(data) # Infinite data loading
        self.G, self.D             = model
        self.optim_G, self.optim_D = optimizer
        self.scheduler             = scheduler
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
        self.max_nodes = 50

        self.max_G_steps = 1000
        self.max_D_steps = 1000

        self.gp_lambda = 10

        self.device = torch.device('cuda:0') if cuda else torch.device('cpu')

    def load_data(self, dataloader):
        while True:
            for data in dataloader:
                yield data

    def run(self, num_epoch):
        for i in range(num_epoch):
            print('EPOCH #%d\n'%(self.epoch))

            with self.logger.experiment.train():
                self.train_discriminator()
                self.train_generator()

            '''
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
            '''
            self.save()

            if self.scheduler: self.scheduler.step()

            self.epoch += 1

    def save(self):
        data = { 'G' : self.G.state_dict(),
                 'D' : self.D.state_dict(), }
        self.logger.save('model_%d.weights'%self.epoch, data)

    def detach_and_clone_graph(self, G):
        newG = dgl.batch([dgl.DGLGraph(_g._graph) for _g in dgl.unbatch(G)])
        newG.edata['feats'] = G.edata['feats'].detach().clone()
        newG.ndata['feats'] = G.ndata['feats'].detach().clone()
        newG.to(self.device)
        return newG

    def train_discriminator(self):
        self.G.eval()
        self.D.train()

        for i in tqdm(range(self.max_D_steps), desc='Discriminator'):
            # Fetch real batch
            real_G = next(self.datagen)
            real_G.to(self.device)

            # Fetch generated batch
            with torch.no_grad():
                noise = torch.randn(real_G.batch_size, 128)
                noise = noise.to(self.device)
                fake_G = self.G(noise)
            fake_G = self.detach_and_clone_graph(fake_G) # just to be sure

            # Discriminate
            real_pred = self.D(real_G).mean()
            fake_pred = self.D(fake_G).mean()

            # Calculate gradient penalty
            real_G_gp = self.detach_and_clone_graph(real_G)
            fake_G_gp = self.detach_and_clone_graph(fake_G)

            real_G_gp.ndata['feats'].requires_grad_(True)
            real_G_gp.edata['feats'].requires_grad_(True)
            fake_G_gp.ndata['feats'].requires_grad_(True)
            fake_G_gp.edata['feats'].requires_grad_(True)

            real_feat = self.D(real_G_gp, return_feat=True)
            fake_feat = self.D(fake_G_gp, return_feat=True)

            alpha = torch.zeros(real_feat.shape[0], 1).uniform_()
            alpha = alpha.expand(real_feat.shape[0], real_feat.shape[1])
            alpha = alpha.contiguous().to(self.device)

            interpolated_feat = alpha*real_feat + (1 - alpha)*fake_feat

            interpolated_pred = self.D.classifier(interpolated_feat)

            grad_output = torch.ones(interpolated_pred.size()).to(self.device)

            gp_gradients = autograd.grad(
                outputs = interpolated_pred,
                inputs = [real_G_gp.ndata['feats'], real_G_gp.edata['feats'],
                          fake_G_gp.ndata['feats'], fake_G_gp.edata['feats']],
                grad_outputs = grad_output,
                create_graph=True, retain_graph=True, only_inputs=True
            )
            gp_gradients = gp_gradients[0]
            gp_gradients = gp_gradients.view(gp_gradients.shape[0], -1)

            gradient_penalty = ((gp_gradients.norm(2, dim=1) - 1) ** 2).mean()
            wdist = fake_pred - real_pred

            loss = wdist + gradient_penalty

            self.optim_D.zero_grad()
            loss.backward()
            self.optim_D.step()


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
