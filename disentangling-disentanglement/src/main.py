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


# Import the RotatedMNIST class
from models.vae_rotated_mnist import VAE_RotatedMNIST, flattened_rotMNIST 

# initialisation et activation du benchmark cuDNN pour optimiser les performances
runId = datetime.datetime.now().isoformat().replace(':', '_')
torch.backends.cudnn.benchmark = True

# definition du argumentParser
parser = argparse.ArgumentParser(description='Disentangling Disentanglement in VAEs',
                                 formatter_class=argparse.RawTextHelpFormatter)

# General
# parser.add_argument('--model', type=str, default='mnist', metavar='M', help='model name (default: mnist)')
parser.add_argument('--model', type=str, default='rotated_mnist', metavar='M', help='model name (default: rotated_mnist)')
parser.add_argument('--name', type=str, default='.', help='experiment name (default: None)')
# parser.add_argument('--save-freq', type=int, default=0, help='print objective values every value (if positive)')
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
# parser.add_argument('--K', type=int, default=1, metavar='K', help='number of samples to estimate ELBO (default: 1)')
parser.add_argument('--alpha', type=float, default=0.0, metavar='A', help='prior regulariser factor (default: 0.0)')
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

# - weights
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
# print('seed', args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# torch.manual_seed(args.seed)
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
    json.dump(args  .__dict__, fp)
with open('{}/args.txt'.format(runPath), 'w') as fp:
    git_hash = subprocess.check_output(['git', 'rev-parse', '--verify', 'HEAD'])
    command = ' '.join(sys.argv[1:])
    # fp.write(git_hash.decode('utf-8') + command)
    # ajout du '\n' pour meilleur visibilité
    fp.write(git_hash.decode('utf-8') + '\n' + command)

torch.save(args, '{}/args.rar'.format(runPath))

# Load data and model
# modelC = getattr(models, 'VAE_{}'.format(args.model))
if args.model == 'rotated_mnist':
    # train_loader, test_loader = modelC.getDataLoaders(args.batch_size, device=device)
    train_loader, test_loader = flattened_rotMNIST(num_tasks=5, 
                                                   per_task_rotation=45, 
                                                   batch_size=args.batch_size, 
                                                   root='./data')
    # model = modelC(args).to(device)
    model = VAE_RotatedMNIST(args).to(device)
else:
    raise ValueError("The model '{}' is not supported.".format(args.model))

# set the optimizer
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, amsgrad=True, betas=(args.beta1, args.beta2))
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(args.beta1, args.beta2))

# objective = getattr(objectives, args.obj + '_objective')
objective = getattr(objectives, args.obj + '_objective', None)

# t_objective = getattr(objectives, 'iwae_objective')
t_objective = getattr(objectives, 'iwae_objective', None)

if t_objective is None:
    raise ValueError("Objective function 'iwae_objective' not found")

import torch

def compute_mmd_loss(z):
    """
    Calcule la perte Maximum Mean Discrepancy (MMD) entre deux distributions :
    - La distribution des échantillons latents `z`
    - Une distribution cible isotropique normale N(0, 1).

    Args:
        z (torch.Tensor): Tensor des échantillons latents de forme [batch_size, latent_dim].

    Returns:
        torch.Tensor: Une valeur scalaire représentant la perte MMD.
    """
    # Génère une distribution cible isotropique N(0, 1)
    z_prior = torch.randn_like(z)

    def rbf_kernel(x, y, sigma=1.0):
        """
        Calcule le noyau Gaussien (RBF - Radial Basis Function) entre deux tensors.

        Args:
            x (torch.Tensor): Tensor de forme [batch_size, latent_dim].
            y (torch.Tensor): Tensor de forme [batch_size, latent_dim].
            sigma (float): Paramètre de l'écart-type du noyau Gaussien.

        Returns:
            torch.Tensor: Matrice de noyau de forme [batch_size, batch_size].
        """
        x = x.unsqueeze(1)  # [batch_size, 1, latent_dim]
        y = y.unsqueeze(0)  # [1, batch_size, latent_dim]
        pairwise_dist = torch.sum((x - y) ** 2, dim=2)  # Distance carrée paire
        return torch.exp(-pairwise_dist / (2 * sigma ** 2))

    # Calcul des noyaux RBF entre les différentes paires
    k_z_z = rbf_kernel(z, z)
    k_prior_prior = rbf_kernel(z_prior, z_prior)
    k_z_prior = rbf_kernel(z, z_prior)

    # Calcul de la perte MMD selon la formule :
    # MMD = E[k(z, z)] + E[k(z_prior, z_prior)] - 2 * E[k(z, z_prior)]
    mmd_loss = k_z_z.mean() + k_prior_prior.mean() - 2 * k_z_prior.mean()

    return mmd_loss

def loss_function(recon_x, x, mu, logvar, regulariser=None, gamma=0.8, beta=1.0):
    """
    Calcule la fonction de perte pour un autoencodeur variationnel (VAE), incluant :
    - La perte de reconstruction
    - La divergence KL avec un poids beta
    - Une pénalité de sparsity optionnelle
    - Une régularisation optionnelle (comme MMD)

    Args:
        recon_x (torch.Tensor or torch.distributions.Distribution): Reconstruction des données d'entrée x.
        x (torch.Tensor): Données d'entrée originales.
        mu (torch.Tensor): Moyennes des variables latentes.
        logvar (torch.Tensor): Log-variances des variables latentes.
        regulariser (str, optional): Type de régularisation à appliquer (ex: 'mmd').
        gamma (float, optional): Coefficient pour la régularisation MMD.
        alpha (float, optional): Poids de la divergence KL.

    Returns:
        torch.Tensor: La perte totale (scalaire).
    """
    # Génération des moyennes de reconstruction si une distribution est donnée
    recon_means = recon_x.mean if isinstance(recon_x, torch.distributions.Distribution) else recon_x
    recon_means = recon_means.view_as(x)

    # Perte de reconstruction (BCE)
    recon_loss = F.binary_cross_entropy(recon_means, x, reduction='sum')

    # Calcul de la divergence KL pondérée
    kld_loss = beta * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Pénalité L1 pour la sparsity
    sparsity_penalty = torch.sum(torch.abs(mu))

    # Régularisation optionnelle (MMD)
    if regulariser == 'mmd':
        mmd_loss = compute_mmd_loss(mu)
        total_loss = recon_loss + kld_loss + gamma * mmd_loss + 0.1 * sparsity_penalty
    else:
        total_loss = recon_loss + kld_loss + 0.1 * sparsity_penalty

    return total_loss

# Fonction d'entraînement
def train(epoch, agg, regulariser, gamma):
    """
    Entraîne le modèle pour une époque donnée.

    Args:
        epoch (int): Le numéro de l'époque en cours.
        agg (dict): Dictionnaire pour suivre les pertes d'entraînement.
        regulariser (str): Type de régularisation à appliquer (ex: 'mmd').
        gamma (float): Coefficient pour la régularisation MMD.

    Returns:
        None
    """
    model.train()
    total_loss = 0.0

    for i, (data, _, _) in enumerate(train_loader):
        # Préparation des données
        data = data.view(data.size(0), -1).to(device)
        optimizer.zero_grad()

        # Passage avant et récupération des variables latentes
        mu, logvar, recon_batch = model(data)

        # Validation des dimensions des variables latentes
        assert mu.shape[1] == args.latent_dim, (
            f"Erreur : Dimension latente incorrecte, attendu {args.latent_dim} mais obtenu {mu.shape[1]}"
        )

        # Calcul de la perte et mise à jour des poids
        loss = loss_function(recon_batch, data, mu, logvar, 
                             regulariser=regulariser, gamma=gamma, beta=args.beta)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Mise à jour des pertes moyennes
    avg_loss = total_loss / len(train_loader.dataset)
    agg['train_loss'].append(avg_loss)
    print(f"====> Epoch: {epoch:03d} Loss: {avg_loss:.2f}")


# Fonction de test
@torch.no_grad()
def test(epoch, beta, alpha, agg, regulariser, gamma):
    """
    Évalue le modèle sur les données de test pour une époque donnée.

    Args:
        epoch (int): Le numéro de l'époque en cours.
        beta (float): Coefficient pour une régularisation spécifique.
        alpha (float): Poids de la divergence KL.
        agg (dict): Dictionnaire pour suivre les pertes de test.
        regulariser (str): Type de régularisation à appliquer (ex: 'mmd').
        gamma (float): Coefficient pour la régularisation MMD.

    Returns:
        None
    """
    model.eval()
    total_loss = 0.0

    for i, (data, labels, _) in enumerate(test_loader):
        # Préparation des données
        data = data.view(data.size(0), -1).to(device)
        
        # Passage avant et récupération des variables latentes
        mu, logvar, recon_batch = model(data)
        
        # Validation des dimensions des variables latentes
        assert mu.shape[1] == args.latent_dim, (
            f"Erreur : Dimension latente incorrecte, attendu {args.latent_dim} mais obtenu {mu.shape[1]}"
        )

        # Calcul de la perte
        loss = loss_function(recon_batch, data, mu, logvar, 
                             regulariser=regulariser, gamma=gamma, beta=beta)
        total_loss += loss.item()

        # Sauvegarde des reconstructions si nécessaire
        if (args.save_freq == 0 or epoch % args.save_freq == 0) and i == 0:
            model.reconstruct(data, runPath, epoch)

    # Mise à jour des pertes moyennes
    avg_loss = total_loss / len(test_loader.dataset)
    agg['test_loss'].append(avg_loss)
    print(f"====> Test:      Loss: {avg_loss:.2f}")


# Main training loop
if __name__ == '__main__':
    with Timer('ME-VAE') as t:
        agg = defaultdict(list)
        print('Starting training...')

        for epoch in range(1, args.epochs + 1):
            # Entraînement
            train(epoch, agg, args.regulariser, args.gamma)

            # Génération forcée à chaque époque
            model.generate(runPath, epoch)

            # Reconstruction forcée à partir d'un batch du train loader
            data, _, _ = next(iter(train_loader))  # Récupérer un batch du train loader
            data = data.view(data.size(0), -1).to(device)  # Mettez les données à plat
            model.reconstruct(data, runPath, epoch)

            # Sauvegarde des modèles et des variables
            save_model(model, runPath + '/model.rar')
            save_vars(agg, runPath + '/losses.rar')

            # Test
            if not args.skip_test:
                test(epoch, args.beta, args.alpha, agg, args.regulariser, args.gamma)

        # Affichage final
        print("p(z) params:")
        print(model.pz_params)
