import os
import git
import argparse
import random
import comet_ml
import torch

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam

from models.encoder import Encoder, Discriminator
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
parser.add_argument('--weight-decay', type=float, default=5e-4)

parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--track', action='store_true', default=False)
parser.add_argument('--comment', type=str, default='')

parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

PROJECT_NAME = 'compound-gan'

random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Dataloader
dataset = ZINC250K(args.data_file)
dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        collate_fn=ZINC_collate,
                        pin_memory=args.cuda,
                        drop_last=True)

# Model
enc = Encoder(59, 13, 128)
gen = Generator(128, [44, 7, 3, 3, 2], [5, 2, 2, 4])
dis = Discriminator(59, 13, 128)

# Optimizer
enc_optimizer = Adam(enc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
gen_optimizer = Adam(gen.parameters(), lr=args.lr, weight_decay=args.weight_decay)
dis_optimizer = Adam(dis.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# CUDA
if args.cuda:
    enc = enc.cuda()
    gen = gen.cuda()
    dis = dis.cuda()

# Logger
dirname = os.path.dirname(os.path.realpath(__file__))
repo    = git.repo.Repo(dirname)
disable = not args.track

if args.track and repo.is_dirty():
    print("Commit before running trackable experiments")
    exit(-1)

logger  = Logger(args.log_root, PROJECT_NAME, repo.commit().hexsha, args.comment, disable)

# Trainer
model     = [enc, gen ,dis]
optimizer = [enc_optimizer, gen_optimizer, dis_optimizer]
trainer = Trainer(dataloader, model, optimizer, None, logger, args.cuda)
trainer.run(args.epoch)


