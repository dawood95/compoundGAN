import os
import git
import argparse
import random
import comet_ml
import torch

from copy import deepcopy
from torch import distributed as dist
from torch.utils.data import DataLoader, distributed
from torch.optim import lr_scheduler, Adam
from torch.nn.parallel import DistributedDataParallel as DDP

from models.network import CVAEF

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

parser.add_argument('--node-dims', type=list, default=[43, 7, 3, 3])
parser.add_argument('--edge-dims', type=list, default=[5, 2, 4])

parser.add_argument('--latent-dim', type=int, default=128)

parser.add_argument('--cnf-hidden-dims', type=list, default=[256, 256, 256])
parser.add_argument('--cnf-context-dim', type=int, default=0)
parser.add_argument('--cnf-T', type=float, default=1.0)
parser.add_argument('--cnf-train-T', type=eval, default=False)

parser.add_argument('--ode-solver', type=str, default='dopri5')
parser.add_argument('--ode-atol', type=float, default=1e-5)
parser.add_argument('--ode-rtol', type=float, default=1e-5)
parser.add_argument('--ode-use-adjoint', action='store_true', default=False)

parser.add_argument('--decoder-num-layers', type=int, default=4)

parser.add_argument('--pretrained', type=str, default='')
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--track', action='store_true', default=False)
parser.add_argument('--comment', type=str, default='')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--global-rank', type=int)

# for torch distributed launch
parser.add_argument('--local_rank', type=int)

torch.set_flush_denormal(True)

if __name__ == "__main__":
    args = parser.parse_args()

    PROJECT_NAME = 'compound-gan'

    # Is this the master process?
    is_master = (args.global_rank == 0) and (args.local_rank == 0)

    # different seed for different processes
    rank_seed = '%d%d'%(args.global_rank, args.local_rank)
    rank_seed = int(rank_seed)
    seed = args.seed + rank_seed

    random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)

    dist.init_process_group(dist.Backend.NCCL)

    # Dataloader
    dataset = ZINC250K(args.data_file)

    train_dataset = dataset
    val_dataset   = deepcopy(dataset)
    split_len     = int(len(train_dataset)*0.8)
    train_dataset.data = train_dataset.data[:split_len]
    val_dataset.data   = val_dataset.data[split_len:]

    train_sampler = distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=ZINC_collate,
                              sampler=train_sampler,
                              pin_memory=args.cuda,
                              drop_last=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=ZINC_collate,
                            pin_memory=args.cuda,
                            drop_last=True)

    # Model
    model = CVAEF(args.node_dims, args.edge_dims, args.latent_dim,
                  args.cnf_hidden_dims, args.cnf_context_dim,
                  args.cnf_T, args.cnf_train_T,
                  args.ode_solver, args.ode_atol, args.ode_rtol,
                  args.ode_use_adjoint, args.decoder_num_layers)

    if args.pretrained:
        state_dict = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(state_dict['parameters'])

    # CUDA
    device = torch.device('cpu')
    if args.cuda:
        device = torch.device('cuda', args.local_rank)
        torch.cuda.set_device(device)

    model = model.to(device)
    model = DDP(model, device_ids=[args.local_rank])

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Logger
    dirname = os.path.dirname(os.path.realpath(__file__))
    repo    = git.repo.Repo(dirname)
    track   = args.track and is_master

    logger = Logger(args.log_root, PROJECT_NAME,
                    repo.commit().hexsha, args.comment, (not track))

    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.99)

    # Trainer
    data_loaders = [train_loader, val_loader]
    trainer = Trainer(data_loaders, model, optimizer, scheduler, logger, device,
                      is_master)
    trainer.run(args.epoch)
