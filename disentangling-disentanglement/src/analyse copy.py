# Future Compatibility
from __future__ import print_function

# Standard Library Imports
import os
import random
import argparse
from collections import defaultdict

# Third-Party Library Imports
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# Local Module Imports
from utils import save_vars
from metrics import compute_sparsity
from models.vae_rotated_mnist import (flattened_rotMNIST, VAE_RotatedMNIST)

# Définir une graine fixe pour assurer la reproductibilité
seed = 42

# chemin du dossier experiments
EXPERIMENTS_PATH_FOLDER = "/home/keen_mcnulty/workdir/disentangling-disentanglement/experiments"

# Fixer la graine pour les bibliothèques PyTorch, NumPy et le module random
torch.manual_seed(seed)  # Graine pour les opérations aléatoires de PyTorch
np.random.seed(seed)     # Graine pour NumPy
random.seed(seed)        # Graine pour le module random standard de Python

# Forcer le déterminisme dans PyTorch pour une reproductibilité totale
torch.backends.cudnn.deterministic = True  # Garantit des résultats déterministes dans cuDNN
torch.backends.cudnn.benchmark = False     # Désactive l'optimisation automatique pour garantir le déterminisme


# definition du argumentParser
parser = argparse.ArgumentParser(description='Analysing GDVAE results')

parser.add_argument('--save-dir', type=str, metavar='N', help='save directory of results')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA use')
parser.add_argument('--disentanglement', action='store_true', default=False, help='compute disentanglement metric')
parser.add_argument('--sparsity', action='store_true', default=False, help='compute sparsity metric')
parser.add_argument('--logp', action='store_true', default=False, help='estimate tight marginal likelihood on completion')
parser.add_argument('--iwae-samples', type=int, default=1000, help='number of samples for IWAE computation (default: 1000)')

cmds = parser.parse_args()

# Configuration du chemin de sauvegarde
# runPath = cmds.save_dir
runPath = os.path.join(EXPERIMENTS_PATH_FOLDER, cmds.save_dir)
if not os.path.exists(runPath):
    os.makedirs(runPath)

# args = torch.load(runPath + '/args.rar')
args = torch.load(os.path.join(runPath, "args.rar"))

# needs_conversion = cmds.no_cuda and args.cuda
# conversion_kwargs = {'map_location': lambda st, loc: st} if needs_conversion else {}
args.cuda = not cmds.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# Chargement du modèle
# modelC = getattr(models, 'VAE_{}'.format(args.model))
model = VAE_RotatedMNIST(args)
# if args.cuda: model.cuda()
model.cuda() if args.cuda else model.cpu()

# state_dict = torch.load(runPath + '/model.rar', **conversion_kwargs)
state_dict = torch.load(os.path.join(runPath, "model.rar"))
model.load_state_dict(state_dict)

# Chargement des loaders
# train_loader, test_loader = model.getDataLoaders(args.batch_size, device=device)
train_loader, test_loader = flattened_rotMNIST(num_tasks=5, 
                                               per_task_rotation=45, 
                                               batch_size=args.batch_size, 
                                               root='./data')

# Fonction pour visualiser les magnitudes latentes moyennes des classes 0, 1 et 2
def plot_latent_magnitude_by_class(data, save_path):
    """
    Plot average latent magnitudes for three classes as grouped bar charts.
    
    Args:
        data (ndarray): Array of shape (3, latent_dim), containing the average magnitudes for each class.
        save_path (str): Path to save the generated plot.
    """
    # Setup: latent dimensions and bar width
    latent_dim = data.shape[1]
    x = np.arange(latent_dim)  # Positions for the latent dimensions
    bar_width = 0.25  # Width of each bar
    
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bar charts for each class
    ax.bar(x - bar_width, data[0], width=bar_width, color='red', label="Class 0")
    ax.bar(x, data[1], width=bar_width, color='blue', label="Class 1")
    ax.bar(x + bar_width, data[2], width=bar_width, color='green', label="Class 2")

    # Axis and title configuration
    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Average Magnitude")
    ax.set_title("Average Latent Magnitude per Dimension for Classes 0, 1, and 2")
    ax.set_xticks(np.arange(0, latent_dim, step=5))  # Tick every 5 dimensions
    ax.set_xticklabels(np.arange(0, latent_dim, step=5))
    ax.legend()

    # Save the figure and clean up
    plt.tight_layout()
    output_path = f"{save_path}_latent_magnitude_plot.png"
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_sparsity_curves(alpha_values, results, save_path):
    """
    Plot multiple sparsity curves for different Beta and Gamma configurations as a function of Alpha.
    
    Args:
        alpha_values (list): List of Alpha values.
        results (dict): Dictionary with keys as (Beta, Gamma) pairs and values as sparsity results.
        save_path (str): Path to save the generated plot.
    """
    # Figure setup: size, line styles, and colors
    plt.figure(figsize=(10, 6))
    line_styles = ['-', '--', '-.', ':']  # Different line styles for multiple configurations
    colors = ['red', 'blue']  # Red for Beta, Blue for Gamma (alternating)

    # Plot sparsity curves for each configuration
    for idx, ((beta, gamma), sparsity_values) in enumerate(results.items()):
        label = f"Beta={beta}, Gamma={gamma}"  # Curve label
        color = colors[idx % len(colors)]      # Alternate between red and blue
        style = line_styles[idx // len(colors) % len(line_styles)]  # Cycle through line styles

        # Plot the curve
        plt.plot(alpha_values, sparsity_values, 
                 label=label, color=color, linestyle=style)

    # Configure plot labels and title
    plt.xlabel("Alpha")
    plt.ylabel("Average Normalized Sparsity")
    plt.title("Sparsity vs Alpha for Multiple Beta and Gamma Configurations")
    plt.legend()
    plt.grid(True)

    # Save the figure and clean up
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

@torch.no_grad()  # Disable gradient computation for efficiency
def compute_sparsity_vs_alpha(alpha_values, beta, gamma):
    """
    Compute mean and standard deviation of sparsity across different Alpha values.

    Args:
        alpha_values (list): List of Alpha values to iterate over.
        beta (float): Beta value used for the model configuration.
        gamma (float): Gamma value used for the model configuration.

    Returns:
        tuple: A tuple containing two lists:
            - sparsity_means: Mean sparsity values for each Alpha.
            - sparsity_stds: Standard deviation of sparsity for each Alpha.
    """
    sparsity_means = []  # List to store mean sparsity values
    sparsity_stds = []   # List to store sparsity standard deviations

    # Iterate through the list of Alpha values
    for alpha in alpha_values:
        # print(f"Computing for Alpha={alpha}, Beta={beta}, Gamma={gamma}")
        
        # Initialize tensor to store latent means (zs_mean) for the entire test dataset
        zs_mean = torch.zeros(len(test_loader.dataset), args.latent_dim, device=device)

        # Process the test data in batches
        for i, (data, _, _) in enumerate(test_loader):
            data = data.to(device)  # Move input data to the specified device
            z_mu, _, _ = model(data)  # Obtain latent mean (z_mu) from the model
            
            # Store the latent means for the current batch
            start_idx = i * data.size(0)
            end_idx = start_idx + data.size(0)
            zs_mean[start_idx:end_idx, :] = z_mu

        # Calculate sparsity for the current Alpha
        sparsity = compute_sparsity(zs_mean, norm=True)
        sparsity_means.append(sparsity.item())  # Append mean sparsity value
        sparsity_stds.append(torch.std(sparsity).item())  # Append sparsity standard deviation

    return sparsity_means, sparsity_stds


@torch.no_grad()  # Disable gradient computation for efficiency
def compute_beta_gamma(alpha_values, beta, gamma):
    """
    Compute normalized sparsity results for different Alpha values.

    Args:
        alpha_values (list): List of Alpha values to iterate over.
        beta (float): Beta value used for the model configuration.
        gamma (float): Gamma value used for the model configuration.

    Returns:
        list: A list containing normalized sparsity values for each Alpha.
    """
    sparsity_results = []  # List to store sparsity results for each Alpha

    # Iterate through all Alpha values
    for alpha in alpha_values:
        # print(f"Computing for Alpha={alpha}, Beta={beta}, Gamma={gamma}")
        
        # Initialize a tensor to store the latent means (zs_mean) for the entire dataset
        zs_mean = torch.zeros(len(test_loader.dataset), args.latent_dim, device=device)

        # Process the test dataset in batches
        for batch_idx, (data, _, _) in enumerate(test_loader):
            data = data.to(device)  # Move data to the appropriate device
            z_mu, _, _ = model(data)  # Get latent means (z_mu) from the model
            
            # Store the batch latent means into the full zs_mean tensor
            start_idx = batch_idx * data.size(0)
            end_idx = start_idx + data.size(0)
            zs_mean[start_idx:end_idx, :] = z_mu

        # Compute normalized sparsity and append the result
        sparsity = compute_sparsity(zs_mean, norm=True)
        sparsity_results.append(sparsity.item())

    return sparsity_results

# note: les autres parties du code ont été supprimé

# Exécution principale
if __name__ == "__main__":
    # Liste des valeurs d'Alpha pour l'analyse
    valeurs_alpha = [0, 200, 400, 600, 800, 1000]

    # Configurations des valeurs de Beta et Gamma
    configurations_beta_gamma = [
        (0.1, 0.0),  # Configuration 1 : Beta = 0.1, Gamma = 0.0
        (0.1, 0.8),  # Configuration 2 : Beta = 0.1, Gamma = 0.8
        (1.0, 0.0),  # Configuration 3 : Beta = 1.0, Gamma = 0.0
        (1.0, 0.8),  # Configuration 4 : Beta = 1.0, Gamma = 0.8
        (5.0, 0.0),  # Configuration 5 : Beta = 5.0, Gamma = 0.0
        (5.0, 0.8),  # Configuration 6 : Beta = 5.0, Gamma = 0.8
    ]

    # Dictionnaire pour stocker les résultats des calculs
    resultats = {}

    # Calcul des résultats pour chaque combinaison de Beta et Gamma
    for beta, gamma in configurations_beta_gamma:
        resultats[(beta, gamma)] = compute_beta_gamma(valeurs_alpha, beta, gamma)

    # Génération et sauvegarde du graphique des courbes Beta et Gamma
    chemin_graphique = os.path.join(runPath, "beta_gamma_vs_alpha.png")
    plot_sparsity_curves(valeurs_alpha, resultats, chemin_graphique)

    # Initialisation des tenseurs pour les magnitudes latentes moyennes
    zs_moyennes = torch.zeros(len(test_loader.dataset), args.latent_dim, device=device)
    etiquettes = torch.zeros(len(test_loader.dataset), dtype=torch.long, device=device)

    # Calcul des vecteurs latents moyens pour chaque échantillon de test
    for index, (donnees, labels, _) in enumerate(test_loader):
        donnees = donnees.to(device)
        z_mu, _, _ = model(donnees)
        zs_moyennes[index * donnees.size(0): (index+1) * donnees.size(0), :] = z_mu
        etiquettes[index * donnees.size(0): (index+1) * donnees.size(0)] = labels.to(device)

    # Calcul des moyennes des magnitudes par classe (0, 1, et 2)
    moyennes_par_classe = []
    for classe in [0, 1, 2]:
        donnees_classe = zs_moyennes[etiquettes == classe]
        moyenne_classe = donnees_classe.abs().mean(dim=0).detach().cpu().numpy()
        moyennes_par_classe.append(moyenne_classe)

    # Conversion en tableau NumPy et génération du graphique des magnitudes
    moyennes_par_classe = np.stack(moyennes_par_classe)
    plot_latent_magnitude_by_class(moyennes_par_classe, os.path.join(runPath, 'plot_sparsity'))

    print("Analyse complétée avec succès.")
