
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as F
from torchvision.utils import save_image

class RotatedMNISTDataset(torch.utils.data.Dataset):
    '''
        This class provides MNIST images with random rotations sampled from
        a list of rotation angles. This list is dependent of the number of tasks
        `num_tasks` and the distance (measured in degrees) between tasks
        `per_task_rotation`.
    '''
    def __init__(self, root='./data', train=True, transform=None, download=True, num_tasks=5, per_task_rotation=45):

        if not isinstance(root, str):
            raise ValueError(f"Le chemin root doit être une chaîne, mais {type(root)} reçu.")

        self.root = root
        self.dataset = torchvision.datasets.MNIST(root=self.root, train=train, transform=None, download=download)
        self.transform = transform
        self.rotation_angles = [float(task * per_task_rotation) for task in range(num_tasks)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        angle_idx = np.random.randint(len(self.rotation_angles))
        angle = self.rotation_angles[angle_idx]  # Choix d'un angle basé sur l'index
        rotated_image = F.rotate(image, angle, fill=(0,))  # Rotation de l'image

        # Convertir en Tensor et garantir que les données restent dans [0, 1]
        if self.transform:
            rotated_image = self.transform(rotated_image)
        rotated_image = rotated_image.clamp(0, 1)  # Clamp pour garantir [0, 1]

        return rotated_image, label, angle_idx

def flattened_rotMNIST(num_tasks, per_task_rotation, batch_size, transform=[], root='./data'):
    '''
    returns
    - train_loader
    - test_loader
    '''

    g = torch.Generator()
    g.manual_seed(0)  # check: always setting generator to 0 ensures the same ordering of data

    extended_transform = transform.copy()
    extended_transform.extend([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    transforms = torchvision.transforms.Compose(extended_transform)

    train = RotatedMNISTDataset(root=root, train=True, download=True, transform=transforms, num_tasks=num_tasks,  per_task_rotation=per_task_rotation)

    test = RotatedMNISTDataset(root=root, train=False, download=True, transform=transforms, num_tasks=num_tasks,  per_task_rotation=per_task_rotation)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, generator=g)

    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, generator=g)

    return train_loader, test_loader

class Enc(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Enc, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 2 * latent_dim)  # Génère mu et logvar
        )

    def forward(self, x):
        h = self.encoder(x)
        z_mu, z_logvar = h[:, :h.size(1) // 2], h[:, h.size(1) // 2:]
        return z_mu, z_logvar

class Dec(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Dec, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, output_dim)
        )

    def forward(self, z):
        h = self.decoder(z)
        return torch.sigmoid(h)  # Activation Sigmoid pour mapper dans [0, 1]

class ModularEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_modules):
        super(ModularEncoder, self).__init__()
        self.submodules = nn.ModuleList([Enc(input_dim, latent_dim) for _ in range(num_modules)])
        print(f"Type of self.submodules in ModularEncoder: {type(self.submodules)}")  # Debug

    def forward(self, x, angle_idx):
        # angle_idx est un tenseur de taille [batch_size]
        B = x.size(0)
        z_mu_list = []
        z_logvar_list = []
        
        # On traite chaque échantillon individuellement selon l'angle_idx correspondant
        for i in range(B):
            idx = int(angle_idx[i].item())
            z_mu, z_logvar = self.submodules[idx](x[i:i+1])
            z_mu_list.append(z_mu)
            z_logvar_list.append(z_logvar)

        # Concaténation le long de la dimension batch
        z_mu = torch.cat(z_mu_list, dim=0)
        z_logvar = torch.cat(z_logvar_list, dim=0)

        return z_mu, z_logvar

class ModularDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, num_modules):
        super(ModularDecoder, self).__init__()
        self.submodules = nn.ModuleList([Dec(latent_dim, output_dim) for _ in range(num_modules)])

    def forward(self, z, angle_idx):
        # angle_idx doit être un tenseur de taille [batch_size]
        # Si angle_idx est un entier, on le transforme en tenseur
        if isinstance(angle_idx, int):
            angle_idx = torch.tensor([angle_idx]*z.size(0), dtype=torch.long, device=z.device)
        if z.size(0) != angle_idx.size(0):
            angle_idx = angle_idx[:z.size(0)]

        # Initialisez une liste pour stocker les sorties
        outputs = []

        # Parcourez chaque élément du batch
        for i in range(z.size(0)):
            module_idx = int(angle_idx[i].item())
            outputs.append(self.submodules[module_idx](z[i:i+1]))

        return torch.cat(outputs, dim=0)

class VAE_RotatedMNIST(nn.Module):
    def __init__(self, args):
        super(VAE_RotatedMNIST, self).__init__()
        self.latent_dim = args.latent_dim
        self.input_dim = 28 * 28

        # Initialisation des sous-modules
        self.encoder = ModularEncoder(self.input_dim, self.latent_dim, num_modules=5)
        self.decoder = ModularDecoder(self.latent_dim, self.input_dim, num_modules=5)

        # Couche d'alignement
        self.alignment_layer = nn.Linear(self.latent_dim, self.latent_dim)

    def encode(self, x, angle_idx):
        return self.encoder(x, angle_idx)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, angle_idx):
        return self.decoder(z, angle_idx)

    def forward(self, x, angle_idx):
        B = x.size(0)
        x = x.view(B, -1)

        # Ajuster angle_idx pour correspondre au batch actuel
        if angle_idx.size(0) != B:
            angle_idx = angle_idx[:B]

        z_mu, z_logvar = self.encode(x, angle_idx)
        z = self.reparameterize(z_mu, z_logvar)
        z_aligned = self.alignment_layer(z)
        x_recon = self.decode(z_aligned, angle_idx).view(B, -1).clamp(0, 1)
        px_z = torch.distributions.Bernoulli(probs=x_recon)
        return z_mu, z_logvar, px_z

    @property
    def pz_params(self):
        return (
            self.gamma,
            self._pz_mu,
            torch.sqrt(self.prior_variance_scale * self.latent_dim * torch.softmax(self._pz_logvar, dim=0))
        )

    def generate(self, runPath, epoch):
        N, K = 64, 8
        z = torch.randn(N, self.latent_dim).to(next(self.parameters()).device)
        angle_idx = torch.zeros(N, dtype=torch.long, device=z.device)  # angle par défaut = 0
        samples = self.decode(z, angle_idx).view(-1, 1, 28, 28)  # Angle par défaut : 0
        save_image(samples.data.cpu(), f'{runPath}/gen_samples_{epoch:03d}.png', nrow=K)

    def reconstruct(self, data, runPath, epoch):
        data = data.to(next(self.parameters()).device)
        angle_idx = torch.zeros(data.size(0), dtype=torch.int64, device=data.device)  # Exemple : reconstruire avec le premier module
        mu, logvar, recon_data = self(data, angle_idx)

        if isinstance(recon_data, torch.distributions.Bernoulli):
            recon_data = recon_data.probs

        recon_data = recon_data.view(-1, 1, 28, 28)
        data = data.view(-1, 1, 28, 28)
        comp = torch.cat([data, recon_data])
        save_image(comp.data.cpu(), f'{runPath}/recon_{epoch:03d}.png', nrow=8)
