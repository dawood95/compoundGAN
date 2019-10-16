import os
import git
import argparse
import random
import comet_ml
import torch

from copy import deepcopy
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam
from torch.nn.parallel import gather

from models.discriminator import Discriminator
from models.generator import Generator

from data.zinc import ZINC250K, ZINC_collate
from utils.trainer import Trainer
from utils.logger import Logger

parser = argparse.ArgumentParser()

parser.add_argument('--data-file', type=str, required=True)
parser.add_argument('--log-root', type=str, default='~/Experiments')

parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=0)

parser.add_argument('--pretrained', type=str, default='')
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--track', action='store_true', default=False)
parser.add_argument('--comment', type=str, default='')

parser.add_argument('--seed', type=int, default=0)

class MyDataParallel(torch.nn.DataParallel):
    def gather(self, outputs, output_device):
        num_nodes = []
        node_feats = []
        edge_feats = []
        max_seq_len = 0
        max_nodes = 0
        for num_node, node_feat, edge_feat in outputs:
            num_nodes.append(num_node)
            node_feats.append(node_feat)
            if len(edge_feat) > 0:
                edge_feats.append(edge_feat)
                max_seq_len = max(max_seq_len, len(edge_feat))

            max_nodes = max(max_nodes, node_feat.shape[1])

        for i in range(len(edge_feats)):
            s, b, f = edge_feats[i].shape
            if s == max_seq_len: continue
            pad = torch.zeros(max_seq_len-s, b, f).to(edge_feats[i].device)
            edge_feats[i] = torch.cat((edge_feats[i], pad), 0)

        for i in range(len(node_feats)):
            b, s, f = node_feats[i].shape
            if s == max_nodes: continue
            pad = torch.zeros(b, max_nodes-s, f).to(node_feats[i].device)
            node_feats[i] = torch.cat((node_feats[i], pad), 1)

        num_nodes = gather(num_nodes, output_device, dim=0)
        node_feats = gather(node_feats, output_device, dim=0)
        if len(edge_feats) > 0:
            edge_feats = gather(edge_feats, output_device, dim=1)

        return num_nodes, node_feats, edge_feats

if __name__ == "__main__":
    args = parser.parse_args()

    PROJECT_NAME = 'compound-gan'

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Dataloader.
    # TODO: Maybe make it iterable dataset instead
    dataset     = ZINC250K(args.data_file)
    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers,
                             collate_fn=ZINC_collate,
                             pin_memory=args.cuda,
                             drop_last=True)

    # Model
    node_feats_num = [43, 7, 3, 3, 2]
    edge_feats_num = [5, 2, 2, 4]
    G = Generator(128, node_feats_num, edge_feats_num, 4)
    D = Discriminator(sum(node_feats_num), sum(edge_feats_num))

    if args.pretrained:
        state_dict = torch.load(args.pretrained, map_location='cpu')
        G.load_state_dict(state_dict['G'])
        D.load_state_dict(state_dict['D'])

    # Optimizer
    optimizer_G = Adam(
        G.parameters(),
        lr=args.lr * 10,
        weight_decay=args.weight_decay,
        betas=(0.5, 0.999),
    )
    optimizer_D = Adam(
        D.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.5, 0.999),
    )

    # CUDA
    if args.cuda:
        G = G.cuda()
        D = D.cuda()

    # Logger
    dirname = os.path.dirname(os.path.realpath(__file__))
    repo    = git.repo.Repo(dirname)
    disable = not args.track

    if args.track and repo.is_dirty():
        print("Commit before running trackable experiments")
        exit(-1)

    logger  = Logger(args.log_root, PROJECT_NAME,
                     repo.commit().hexsha, args.comment, disable)

    # Trainer
    model      = [G, D]
    optimizer  = [optimizer_G, optimizer_D]
    trainer = Trainer(data_loader, model, optimizer, None, logger, args.cuda)
    trainer.run(args.epoch)
