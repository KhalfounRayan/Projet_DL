from __future__ import print_function

# Standard Library Imports
import os
import random
import argparse
from collections import defaultdict
import csv

# Third-Party Library Imports
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Local Module Imports
from utils import save_vars, load_vars
from models.vae_rotated_mnist import (flattened_rotMNIST, VAE_RotatedMNIST)

# Set seed for reproducibility
seed = 42
EXPERIMENTS_PATH_FOLDER = "/home/keen_mcnulty/workdir/disentangling-disentanglement/experiments"

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='Analyse GDVAE Results')
parser.add_argument('--save-dir', type=str, metavar='N', help='Save directory of results')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA use')
cmds = parser.parse_args()

runPath = os.path.join(EXPERIMENTS_PATH_FOLDER, cmds.save_dir)
if not os.path.exists(runPath):
    os.makedirs(runPath)

args = torch.load(os.path.join(runPath, "args.rar"))
if isinstance(args, dict):
    args = argparse.Namespace(**args)

args.cuda = not cmds.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

model = VAE_RotatedMNIST(args)
model.cuda() if args.cuda else model.cpu()

state_dict = torch.load(os.path.join(runPath, "model.rar"))
model_state_dict = model.state_dict()

# Chargement flexible
compatible_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
model_state_dict.update(compatible_state_dict)
model.load_state_dict(model_state_dict)

train_loader, test_loader = flattened_rotMNIST(num_tasks=5, per_task_rotation=45, batch_size=args.batch_size, root='./data')

sns.set(style="darkgrid")


def compute_latent_stats_by_angle(test_loader, model, device, args, angles_to_consider=None):
    angle_dict = defaultdict(list)
    for data, _, angle in test_loader:
        data = data.to(device)
        z_mu, _, _ = model(data)
        for i in range(data.size(0)):
            curr_angle = float(angle[i].item())
            if angles_to_consider is None or curr_angle in angles_to_consider:
                angle_dict[curr_angle].append(z_mu[i].unsqueeze(0))

    angle_stats = {}
    for ang, z_list in angle_dict.items():
        z_cat = torch.cat(z_list, dim=0)
        angle_stats[ang] = {
            'mean': z_cat.mean(dim=0),
            'std': z_cat.std(dim=0),
            'values': z_cat
        }
    return angle_stats


def save_angle_stats_to_csv(angle_stats, save_path):
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["angle", "dimension", "mean", "std"])
        for ang, stats in angle_stats.items():
            mean_vec = stats['mean'].detach().cpu().numpy()
            std_vec = stats['std'].detach().cpu().numpy()
            for dim_idx in range(mean_vec.shape[0]):
                writer.writerow([ang, dim_idx, mean_vec[dim_idx], std_vec[dim_idx]])
    print(f"Angle-latent stats saved to {save_path}")





def plot_reconstruction_comparison(test_loader, model, runPath):
    """
    Compare les images originales avec les reconstructions.
    """
    data_iter = iter(test_loader)
    original_data, _, _ = next(data_iter)
    original_data = original_data.view(original_data.size(0), -1).to(device)
    reconstructed_data, _, _ = model(original_data)

    num_samples = min(10, original_data.size(0))
    original_samples = original_data[:num_samples].cpu().detach().numpy().reshape(num_samples, 28, 28)

    # Vérification de la taille des données reconstruites
    reconstructed_data = reconstructed_data[:num_samples].cpu().detach().numpy()
    if reconstructed_data.shape[1] == 28 * 28:  # Vérifier si c'est une image plate
        reconstructed_samples = reconstructed_data.reshape(num_samples, 28, 28)
    elif reconstructed_data.shape[1] == 10:  # Si c'est en espace latent
        print("Warning: Unexpected reconstructed shape (latent space output). Skipping plot.")
        return
    else:
        raise ValueError(f"Unexpected reconstructed data shape: {reconstructed_data.shape}")

    fig, axes = plt.subplots(2, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        axes[0, i].imshow(original_samples[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed_samples[i], cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(runPath, "reconstruction_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Reconstruction comparison plot saved to {save_path}")



def plot_loss_curves(runPath):
    losses_path = os.path.join(runPath, "losses.rar")
    if not os.path.exists(losses_path):
        print("No losses file found. Skipping loss plots.")
        return
    agg = load_vars(losses_path)
    train_loss = agg.get('train_loss', [])
    test_loss = agg.get('test_loss', [])
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(runPath, "loss_curves.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curves plot saved to {save_path}")


def plot_latent_means_by_class(test_loader, model, runPath):
    latent_means = defaultdict(list)
    for data, labels, _ in test_loader:
        data = data.to(device)
        z_mu, _, _ = model(data)
        for label in torch.unique(labels):
            class_mean = z_mu[labels == label].mean(dim=0).cpu().detach().numpy()
            latent_means[label.item()].append(class_mean)

    plt.figure(figsize=(12, 6))
    for label, means in latent_means.items():
        avg_means = np.mean(means, axis=0)
        plt.plot(avg_means, label=f'Class {int(label)}')

    plt.xlabel("Latent Dimension")
    plt.ylabel("Mean Value")
    plt.title("Évolution des Moyennes Latentes par Classe")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(runPath, "latent_means_by_class.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Latent means by class plot saved to {save_path}")


def plot_reconstruction_error_by_angle(test_loader, model, runPath):
    """
    Graphique montrant l'erreur moyenne de reconstruction pour chaque angle de rotation.
    """
    angle_errors = defaultdict(list)
    for data, _, angles in test_loader:
        data = data.view(data.size(0), -1).to(device)  # Aplatir les données en vecteurs
        reconstructed, _, _ = model(data)

        # Vérifier si les données reconstruites sont dans l'espace latent ou sous forme d'image
        if reconstructed.size(1) != data.size(1):  # Si taille incompatible
            print("Warning: Reconstructed data is in latent space. Skipping error computation.")
            return

        errors = ((data - reconstructed) ** 2).mean(dim=1).cpu().detach().numpy()
        for angle in torch.unique(angles):
            angle_errors[angle.item()].extend(errors[angles == angle].tolist())

    avg_errors = {angle: np.mean(errs) for angle, errs in angle_errors.items()}
    angles, errors = zip(*sorted(avg_errors.items()))

    plt.figure(figsize=(10, 5))
    plt.bar(angles, errors)
    plt.xlabel("Angle (degrés)")
    plt.ylabel("Erreur Moyenne de Reconstruction")
    plt.title("Erreur Moyenne de Reconstruction par Angle")
    plt.tight_layout()
    save_path = os.path.join(runPath, "reconstruction_error_by_angle.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Reconstruction error by angle plot saved to {save_path}")



def plot_latent_tsne(test_loader, model, runPath, n_samples=1000):
    data_list, label_list = [], []
    for data, labels, _ in test_loader:
        data = data.to(device)
        z_mu, _, _ = model(data)
        data_list.append(z_mu.cpu().detach().numpy())
        label_list.append(labels.cpu().numpy())

    latents = np.concatenate(data_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    if latents.shape[0] > n_samples:
        indices = np.random.choice(latents.shape[0], n_samples, replace=False)
        latents = latents[indices]
        labels = labels[indices]

    tsne = TSNE(n_components=2, random_state=seed)
    tsne_latents = tsne.fit_transform(latents)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_latents[:, 0], tsne_latents[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter, label="Classe")
    plt.title("Visualisation TSNE des Latents")
    plt.tight_layout()
    save_path = os.path.join(runPath, "latent_tsne.png")
    plt.savefig(save_path)
    plt.close()
    print(f"TSNE latent visualization plot saved to {save_path}")

def plot_latent_variations_by_angle(angle_stats, runPath, dims_to_plot=5):
    """
    Graphique montrant les variations des dimensions latentes en fonction de l'angle de rotation.

    Parameters:
        angle_stats (dict): Statistiques des latents par angle.
        runPath (str): Chemin de sauvegarde.
        dims_to_plot (int): Nombre de dimensions latentes à tracer.
    """
    angles = sorted(angle_stats.keys())
    means_per_angle = np.stack([angle_stats[ang]['mean'].detach().cpu().numpy() for ang in angles])
    
    plt.figure(figsize=(12, 6))
    for dim in range(min(dims_to_plot, means_per_angle.shape[1])):
        plt.plot(angles, means_per_angle[:, dim], label=f"Dimension {dim}")

    plt.xlabel("Angle (degrés)")
    plt.ylabel("Valeur Moyenne")
    plt.title("Variations des Dimensions Latentes en Fonction de l'Angle")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(runPath, "latent_variations_by_angle.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Latent variations by angle plot saved to {save_path}")

if __name__ == "__main__":
    angles_disponibles = test_loader.dataset.rotation_angles
    angle_stats = compute_latent_stats_by_angle(test_loader, model, device, args, angles_to_consider=angles_disponibles)

    csv_path = os.path.join(runPath, "angle_latent_stats.csv")
    save_angle_stats_to_csv(angle_stats, csv_path)

    plot_reconstruction_comparison(test_loader, model, runPath)
    plot_loss_curves(runPath)
    plot_latent_means_by_class(test_loader, model, runPath)
    plot_reconstruction_error_by_angle(test_loader, model, runPath)
    plot_latent_tsne(test_loader, model, runPath)
    plot_latent_variations_by_angle(angle_stats, runPath, dims_to_plot=10)

    print("Analysis completed successfully.")
