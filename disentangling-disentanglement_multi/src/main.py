
# Standard Library Imports
import sys
import os
import datetime
import json
import subprocess
import argparse
from tempfile import mkdtemp
from collections import defaultdict

# Third-Party Library Imports
import torch
import numpy as np
from torchvision.utils import save_image
import torch.nn.functional as F

# Local Module Imports
from vis import visualize_latent_space
from utils import (Logger, Timer, save_model, save_vars, probe_infnan)
from distributions.sparse import Sparse
import objectives
import regularisers

from models.vae_rotated_mnist import VAE_RotatedMNIST, flattened_rotMNIST 

# initialisation et activation du benchmark cuDNN pour optimiser les performances
runId = datetime.datetime.now().isoformat().replace(':', '_')
torch.backends.cudnn.benchmark = True

# definition du argumentParser
parser = argparse.ArgumentParser(description='Disentangling Disentanglement in VAEs',
                                 formatter_class=argparse.RawTextHelpFormatter)

# General
parser.add_argument('--model', type=str, default='rotated_mnist', metavar='M', help='model name (default: rotated_mnist)')
parser.add_argument('--name', type=str, default='.', help='experiment name (default: None)')
parser.add_argument('--save-freq', type=int, default=15, help='print objective values every value (default: 15)')
parser.add_argument('--skip-test', action='store_true', default=False, help='skip test dataset computations')

# DL neural net config 
parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H', help='number of hidden layers in enc and dec (default: 1)')
parser.add_argument('--hidden-dim', type=int, default=100, help='number of units in hidden layers in enc and dec (default: 100)')
parser.add_argument('--fBase', type=int, default=32, help='parameter for DCGAN networks')
parser.add_argument('--epochs', type=int, default=30, metavar='E', help='number of epochs to train (default: 30)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for data (default: 64)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimiser (default: 1e-4)')

# VAE config
parser.add_argument('--latent-dim', type=int, default=10, metavar='L', help='latent dimensionality (default: 10)')
parser.add_argument('--alpha', type=float, default=1.0, metavar='A', help='prior regulariser factor (default: 0.0)')
parser.add_argument('--beta', type=float, default=1.0, metavar='B', help='overlap factor (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.8, help='weight of the spike component of the sparse prior')
parser.add_argument('--obj', type=str, default='decomp', choices=['decomp', 'iwae'], help='objective function to use (default: decomp)')

# - Algorithm
parser.add_argument('--beta1', type=float, default=0.9, help='first parameter of Adam (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999, help='second parameter of Adam (default: 0.900)')

# Prior / posterior
parser.add_argument('--prior', type=str, default='Normal', help='prior distribution (default: Normal)')
parser.add_argument('--posterior', type=str, default='Normal', help='posterior distribution (default: Normal)')
parser.add_argument('--df', type=float, default=2., help='degree of freedom of the Student-t')
parser.add_argument('--prior-variance', type=str, default='iso', choices=['iso', 'pca'], help='value prior variances initialisation')
parser.add_argument('--prior-variance-scale', type=float, default=1., help='scale prior variance by this value (default:1.)')
parser.add_argument('--learn-prior-variance', action='store_true', default=False, help='learn model prior variances')

# Ajout de l'argument `--regulariser` (optionnel)
parser.add_argument('--regulariser', type=str, default=None, choices=['mmd', 'mmd_dim'], help="Type de régularisation : 'mmd', 'mmd_dim' (default: None).")

# Computation
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA use')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# Setting random seed
if args.seed == 0:
    args.seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

directory_name = '../experiments/{}'.format(args.name)
if args.name != '.':
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    runPath = mkdtemp(prefix=runId, dir=directory_name)
else:
    runPath = mkdtemp(prefix=runId, dir=directory_name)
sys.stdout = Logger('{}/run.log'.format(runPath))
print('RunID:', runId)

# save args to run
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
with open('{}/args.txt'.format(runPath), 'w') as fp:
    git_hash = subprocess.check_output(['git', 'rev-parse', '--verify', 'HEAD'])
    command = ' '.join(sys.argv[1:])
    fp.write(git_hash.decode('utf-8') + '\n' + command)

torch.save(args, '{}/args.rar'.format(runPath))

if args.model == 'rotated_mnist':
    train_loader, test_loader = flattened_rotMNIST(num_tasks=5, 
                                                   per_task_rotation=45, 
                                                   batch_size=args.batch_size, 
                                                   root='./data')
    model = VAE_RotatedMNIST(args).to(device)
else:
    raise ValueError("The model '{}' is not supported.".format(args.model))

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(args.beta1, args.beta2))

objective = getattr(objectives, args.obj + '_objective', None)
t_objective = getattr(objectives, 'iwae_objective', None)
if t_objective is None:
    raise ValueError("Objective function 'iwae_objective' not found")

def compute_mmd_loss(z):
    z_prior = torch.randn_like(z)
    def rbf_kernel(x, y, sigma=1.0):
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)
        pairwise_dist = torch.sum((x - y) ** 2, dim=2)
        return torch.exp(-pairwise_dist / (2 * sigma ** 2))

    k_z_z = rbf_kernel(z, z)
    k_prior_prior = rbf_kernel(z_prior, z_prior)
    k_z_prior = rbf_kernel(z, z_prior)
    mmd_loss = k_z_z.mean() + k_prior_prior.mean() - 2 * k_z_prior.mean()
    return mmd_loss

def loss_function(recon_x, x, mu, logvar, regulariser=None, gamma=0.8, beta=1.0):
    recon_means = recon_x.mean if isinstance(recon_x, torch.distributions.Distribution) else recon_x
    recon_means = recon_means.view_as(x)

    recon_loss = F.binary_cross_entropy(recon_means, x, reduction='sum')
    kld_loss = beta * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    sparsity_penalty = torch.sum(torch.abs(mu))

    if regulariser == 'mmd':
        mmd_loss = compute_mmd_loss(mu)
        total_loss = recon_loss + kld_loss + gamma * mmd_loss + 0.1 * sparsity_penalty
    else:
        total_loss = recon_loss + kld_loss + 0.1 * sparsity_penalty

    return total_loss

def train(epoch, agg, regulariser, gamma):
    model.train()
    total_loss = 0.0
    for i, (data, _, angle_idx) in enumerate(train_loader): 
        data = data.view(data.size(0), -1).to(device)
        angle_idx = angle_idx.to(device)  # IMPORTANT : mettre angle_idx sur le device
        optimizer.zero_grad()

        mu, logvar, recon_batch = model(data, angle_idx)
        assert mu.shape[1] == args.latent_dim, (
            f"Erreur : Dimension latente incorrecte, attendu {args.latent_dim} mais obtenu {mu.shape[1]}"
        )

        loss = loss_function(recon_batch, data, mu, logvar, 
                             regulariser=regulariser, gamma=gamma, beta=args.beta)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader.dataset)
    agg['train_loss'].append(avg_loss)
    print(f"====> Epoch: {epoch:03d} Loss: {avg_loss:.2f}")

@torch.no_grad()
def test(epoch, beta, alpha, agg, regulariser, gamma):
    model.eval()
    total_loss = 0.0

    for i, (data, labels, angle_idx) in enumerate(test_loader):
        data = data.view(data.size(0), -1).to(device)
        angle_idx = angle_idx.to(device)  # IMPORTANT : mettre angle_idx sur le device

        mu, logvar, recon_batch = model(data, angle_idx)
        assert mu.shape[1] == args.latent_dim, (
            f"Erreur : Dimension latente incorrecte, attendu {args.latent_dim} mais obtenu {mu.shape[1]}"
        )

        loss = loss_function(recon_batch, data, mu, logvar, 
                             regulariser=regulariser, gamma=gamma, beta=beta)
        total_loss += loss.item()

        if (args.save_freq == 0 or epoch % args.save_freq == 0) and i == 0:
            model.reconstruct(data, runPath, epoch)

    avg_loss = total_loss / len(test_loader.dataset)
    agg['test_loss'].append(avg_loss)
    print(f"====> Test:      Loss: {avg_loss:.2f}")

if __name__ == '__main__':
    with Timer('ME-VAE') as t:
        agg = defaultdict(list)
        print('Starting training...')

        for epoch in range(1, args.epochs + 1):
            train(epoch, agg, args.regulariser, args.gamma)
            model.generate(runPath, epoch)
            data, _, angle_idx = next(iter(train_loader))
            data = data.view(data.size(0), -1).to(device)
            angle_idx = angle_idx.to(device) # mettre angle_idx sur device
            model.reconstruct(data, runPath, epoch)

            save_model(model, runPath + '/model.rar')
            save_vars(agg, runPath + '/losses.rar')

            if not args.skip_test:
                test(epoch, args.beta, args.alpha, agg, args.regulariser, args.gamma)

        print("p(z) params:")
        