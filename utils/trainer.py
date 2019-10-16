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
        self.max_nodes = 50

        self.log_step    = 25
        self.max_G_steps = 1
        self.max_D_steps = 1
        self.num_iters   = 1000

        self.penalty_lambda = 1
        self.max_seq_len = 50

        self.clip_value = 0.01

        self.device = torch.device('cuda:0') if cuda else torch.device('cpu')

    def load_data(self):
        while True:
            for data in self.dataloader:
                yield data

    def run(self, num_epoch):
        last_update = 0
        for e in range(num_epoch):

            self.dataloader.dataset.max_seq_len = self.max_seq_len - 1

            tqdm.write('Epoch %d'%(self.epoch))

            total_wdist  = 0
            total_gp     = 0
            total_critic = 0
            total_real = 0
            total_fake = 0
            for i in tqdm(range(self.num_iters)):
                self.train_discriminator()
                self.train_generator()

                # total_wdist += wdist
                # total_gp += gp
                # total_real += real
                # total_fake += fake
                # total_critic += critic

            # total_wdist /= self.num_iters
            # total_gp /= self.num_iters
            # total_critic /= self.num_iters
            # total_real /= self.num_iters
            # total_fake /= self.num_iters

            #tqdm.write('Generator [Total] : Critic [%02.5e]'%(critic))
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
            self.save(temp='model.weights')

            if self.scheduler: self.scheduler.step()

            self.step_G = 0
            self.step_D = 0
            self.epoch += 1

            if self.epoch - last_update > 20:#total_fake > -1e-2:
                self.max_seq_len *= 2
                last_update = self.epoch

            self.max_seq_len = min(self.max_nodes, self.max_seq_len)

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
                fake_G = self.G.create_graph(*self.G(noise, self.max_seq_len))

            # Real Label
            # real_label  = torch.full((self.batch_size, 1), 1, device=self.device)
            # label_noise = torch.zeros_like(real_label).uniform_(-0.1, 0.1)
            # real_label  = real_label + label_noise

            # Fake Label
            # fake_label = torch.full((self.batch_size, 1), 0, device=self.device)
            # label_noise = torch.zeros_like(fake_label).uniform_(-0.1, 0.1)
            # fake_label  = fake_label + label_noise

            # Pred
            pred_real = self.D(real_G)
            pred_fake = self.D(fake_G)

            # Loss (Rel Avg LSGAN)
            # real_loss = (pred_real - pred_fake.mean() - real_label) ** 2
            # real_loss = real_loss.mean()

            # fake_loss = (pred_fake - pred_real.mean() + real_label) ** 2
            # fake_loss = fake_loss.mean()

            # real_loss = F.binary_cross_entropy_with_logits(pred_real, real_label)
            # fake_loss = F.binary_cross_entropy_with_logits(pred_fake, fake_label)
            # loss = (real_loss + fake_loss) / 2

            loss = pred_fake.mean() - pred_real.mean()

            self.optim_D.zero_grad()
            loss.backward()
            self.optim_D.step()

            for p in self.D.parameters():
                p.data.clamp_(-self.clip_value, self.clip_value)

            # pred_real = torch.sigmoid(pred_real)
            # pred_fake = torch.sigmoid(pred_fake)

            # loss = real_err + fake_err

            #fake_G = self.detach_and_clone_graph(fake_G) # just to be sure

            # Discriminate
            #feat_real = self.D(real_G, return_feat=True)
            #feat_pred = self.D(fake_G, return_feat=True)

            # pred_real  = self.D(real_G)
            # pred_fake = self.D(fake_G)

            '''
            # NOTE: This is kind of dumb to use maybe.
            # Using distance between features instead of distance between graphs
            # HOW DO WE GET DISTANCE BETWEEN GRAPHS
            dist_norm = feat_real.detach() - feat_pred.detach()
            dist_norm = (dist_norm**2).sum(1)**0.5
            '''

            # wdist = pred_fake - pred_real # minimize fake, maximize real

            # For random pair of points, make sure the function is 'smooth'
            '''
            penalty = (pred_fake - pred_real)**2
            penalty = penalty/(2*self.penalty_lambda*dist_norm) # minimize penalty
            penalty = penalty.mean()
            '''

            # loss = wdist #+ penalty

            # loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1)
            # for p in self.D.parameters():
            #    p.data.clamp_(-self.clip_value, self.clip_value)

            # self.optim_D.step()

            self.step_D += 1

            # wdist = float(wdist.item())
            # gp    = float(penalty.item())

            pred_fake = float(pred_fake.mean().item())
            pred_real = float(pred_real.mean().item())
            loss      = float(loss.item())

            # total_wdist += wdist
            # total_real += pred_real
            # total_fake += pred_fake

            # total_gp += gp

            # total_fake += pred_fake
            # total_real += pred_real
            # total_loss += loss

            if self.step_D % self.log_step == 0:
                step = (self.epoch * self.max_D_steps) + self.step_D
                # Log this step's loss values
                # self.logger.experiment.log_metrics({
                #     'wasserstein_distance': wdist,
                #'gradient_penalty': gp,
                # }, step=step)
                # Print em out
                # tqdm.write('Discriminator [%5d] [%3d] : WDist [%02.5e] | GP [%02.5e] | '
                #           'Real Value [%02.5e] | Fake Value [%02.5e]'%
                #           (self.step_D, self.max_seq_len, wdist, 0,
                #            pred_real, pred_fake))
                tqdm.write('Discriminator [%5d] [%3d] : Loss [%02.5e] | '
                           'Real [%02.5e] | Fake [%02.5e]'%
                           (self.step_D, self.max_seq_len, loss, pred_real, pred_fake))

        # Average loss over num iters
        # total_loss /= (i+1)
        # total_real /= (i+1)
        # total_fake /= (i+1)
        # total_gp    /= (i+1)

        #tqdm.write('D Total : Loss [%02.5e] | Real [%02.5e] | Fake [%02.5e]'
        #                   %(total_loss, total_real, total_fake))
        #tqdm.write('Discriminator [Total] : WDist [%02.5e] | GP [%02.5e]'%
        #           (total_wdist, total_gp))

        return None#total_loss, total_real, total_fake#, total_gp

    def train_generator(self):
        self.G.train()
        self.D.eval()
        # NOTE: RNN backpass can only be done in train mode
        # Other solution is to use a different pooling op
        # Maybe, DIFFPOOL
        # self.D.gcn.pool.train()

        for p in self.D.parameters():
            p.requires_grad = False

        total_loss = 0
        total_fake = 0

        for i in range(self.max_G_steps):

            # Generate noise
            noise = torch.randn(self.batch_size, 128).normal_(0, 1)
            noise = noise.to(self.device)

            # Generate graph from noise
            fake_G = self.G.create_graph(*self.G(noise, self.max_seq_len))

            # Find discriminator rating for generated graph
            pred_fake = self.D(fake_G)#.mean()

            # fake_label = torch.full((self.batch_size, 1), 1, device=self.device)
            # fake_err = F.binary_cross_entropy_with_logits(pred_fake, fake_label)

            # critic = self.D(fake_G).mean()

            # Maximise discriminator rating
            # loss = -pred_fake

            # pred_fake = torch.sigmoid(pred_fake).mean()
            loss = -1 * pred_fake.mean()

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
            pred_fake = float(pred_fake.mean().item())

            total_fake += pred_fake
            total_loss += loss

            if self.step_G % self.log_step == 0:
                step = (self.epoch * self.max_G_steps) + self.step_G
                #self.logger.experiment.log_metrics({
                #    'critic': critic,
                #}, step=step)
                #tqdm.write('Generator [%5d] [%3d]: Critic [%02.5e]'%
                #           (self.step_G, self.max_seq_len, pred_fake))
                #tqdm.write('\t\tGenerator [%5d] [%3d] : Loss [%02.5e] | Fake [%02.5e]'
                #           %(self.step_G, self.max_seq_len, loss, pred_fake))

        # total_fake /= (i+1)
        # total_loss /= (i+1)
        #tqdm.write('Generator [Total] : Critic [%02.5e]'%(critic))
        #tqdm.write('D Total : Loss [%02.5e] | Fake [%02.5e]'
        #           %(total_loss, total_fake))

        return None#total_loss

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
