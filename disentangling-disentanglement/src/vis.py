# Visualization libraries
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
sns.set()

# Numerical and statistical libraries
import numpy as np
import scipy.stats as stats

# Machine learning and deep learning libraries
import torch
import torch.nn.functional as F  # Neural network functional utilities

# Dimensionality reduction libraries
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from umap import UMAP

# System and OS utilities
import os
import sys

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def scatter_plot(points, filepath, labels=None):
    ax = plt.gca()
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    if isinstance(points, list):
        for i in range(len(points)):
            if labels is not None and i == 0:
                labels = labels.numpy().astype(int)[:, 0]
                color = np.array(sns.color_palette("hls", 8))[labels]
            else:
                color = sns.color_palette("hls", 8)[i]
            plt.scatter(points[i][:, 0], points[i][:, 1], alpha=0.5, s=30, color=color)
    else:
        plt.scatter(points[:, 0], points[:, 1], alpha=0.3)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.clf()

def posterior_plot_pinwheel(qz_x_mean, qz_x_std, filepath):
    ax = plt.gca()
    x_lim = 2
    ax.set_xlim([-x_lim, x_lim])
    ax.set_ylim([-x_lim, x_lim])
    colours = np.array(sns.color_palette("hls", 8))

    qz_x = torch.distributions.Normal(qz_x_mean, qz_x_std)

    # Build grid
    nb_points = 300
    space = torch.linspace(-x_lim, x_lim, nb_points)
    X = space.unsqueeze(-1).expand(nb_points, nb_points)
    Y = space.unsqueeze(0).expand(nb_points, nb_points)
    grid = torch.cat([X.flatten().unsqueeze(0), Y.flatten().unsqueeze(0)]).t()

    # Compute aggregate posterior pdf
    grid_pdf = qz_x.log_prob(grid.unsqueeze(1).expand(nb_points**2, qz_x_mean.shape[0], qz_x_mean.shape[1]).to(qz_x_mean.device))
    grid_pdf = grid_pdf.sum(-1).exp().mean(-1).data.cpu()
    grid_pdf = grid_pdf.reshape(nb_points, nb_points)
    cmap = 'Blues'

    # Plot heatmaps
    ax = sns.heatmap(grid_pdf, alpha=1., vmin=0., vmax=1.0, cmap=cmap, linewidths=0.0, rasterized=True, yticklabels=[''], xticklabels=[''], cbar=False)

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.clf()

def plot_latent_magnitude(xs, labels, path='', label=''):
    N, D = xs.shape
    plt.style.use(['default'])
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, axes = plt.subplots(N, 1, sharey=True, sharex=True, figsize=(20, 3 * N), facecolor='white')
    colors = sns.color_palette("Set2", 10)

    fontsize = 30
    barWidth = .25
    r = np.arange(D)
    for i, axis in enumerate(axes):
        axis.bar(r, xs[i], label=labels[i], width=barWidth, color=colors[i])
        
        # axis.grid(b=False)
        axis.grid(visible=False)

        axis.set_ylim([0, 1.1])
        axis.tick_params(axis='y', which='major', labelsize=fontsize, length=12, width=3, direction='in')
        axis.tick_params(axis='x', which='major', labelsize=fontsize, length=12, width=3, direction='out')
        axis.legend(loc='upper left', frameon=False, fontsize=25)
        axis.set_yticks([0., .5, 1.])
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

    fig.text(0.06, .5, 'Avg. latent magnitude', va='center', ha='center', rotation='vertical', fontsize=fontsize)
    plt.xlabel('Latent dimension', fontsize=fontsize)
    plt.rcParams.update({'font.size': fontsize})
    fig.savefig(path + '.pdf', bbox_inches='tight', transparent=False)
    plt.clf()

def visualize_latent_space(model, data_loader, method='PCA', num_samples=1000, save_path=None):
    """
    Visualise l'espace latent du modèle en utilisant PCA, t-SNE ou UMAP.

    Args:
        model (torch.nn.Module): Modèle entraîné contenant une méthode `encode`.
        data_loader (DataLoader): DataLoader fournissant les données et labels.
        method (str): 'PCA', 'TSNE' ou 'UMAP' pour choisir la méthode de réduction de dimension.
        num_samples (int): Nombre d'échantillons à utiliser pour la visualisation.
        save_path (str, optional): Chemin pour sauvegarder la figure. Si None, affiche la figure.
    """
    # Mode d'inférence
    model.eval()

    latents, labels = [], []

    # Extraction des vecteurs latents et labels
    for batch_idx, (data, label, _) in enumerate(data_loader):
        print(f"[DEBUG] Batch {batch_idx}: Data shape: {data.shape}")

        # Préparation des données
        data = data.view(data.size(0), -1).to(next(model.parameters()).device)
        mu, _ = model.encode(data)  # Extraction des représentations latentes

        # Stockage des données
        latents.append(mu.detach().cpu().numpy())
        labels.append(label.numpy())

        # Arrêt si le nombre d'échantillons est atteint
        if len(latents) * data.size(0) >= num_samples:
            break

    # Concaténation et limitation des échantillons
    latents = np.concatenate(latents, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]

    print(f"[DEBUG] Total latents shape: {latents.shape}")

    # Réduction de dimension
    if method.upper() == 'PCA':
        reducer = PCA(n_components=2)
    elif method.upper() == 'TSNE':
        reducer = TSNE(n_components=2, perplexity=40, n_iter=3000, random_state=42)
    # elif method.upper() == 'UMAP':
    #     reducer = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("'method' doit être 'PCA', 'TSNE'")

    reduced_latents = reducer.fit_transform(latents)
    print(f"[DEBUG] Reduced latents shape: {reduced_latents.shape}")

    # Visualisation des données
    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(
        reduced_latents[:, 0], reduced_latents[:, 1],
        c=labels, cmap='tab10', alpha=0.8, s=20
    )
    plt.colorbar(scatter, label='Labels')
    plt.title(f"Visualisation de l'espace latent ({method.upper()})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    # Ajout d'annotations pour les clusters (optionnel)
    for i, txt in enumerate(labels):
        plt.annotate(txt, (reduced_latents[i, 0], reduced_latents[i, 1]), fontsize=8, alpha=0.6)

    # Sauvegarde ou affichage
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Visualisation sauvegardée dans : {save_path}")
    else:
        plt.show()
