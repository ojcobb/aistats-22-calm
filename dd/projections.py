from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn


class Proj(ABC):
    params: dict
    configure: bool=False
    @abstractmethod
    def __call__(self, x: torch.tensor) -> torch.tensor:
        raise NotImplementedError()


class NormalLogpdf(Proj):
    def __init__(self, dim: int, mean: float=0., var: float=1.):
        self.params = {'dim': dim, 'mean': mean, 'var': var}
        self.dist = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(dim) + torch.tensor(mean), torch.eye(dim) * torch.tensor(var).sqrt()
        )
    
    def __call__(self, x: torch.tensor) -> torch.tensor:
        return self.dist.log_prob(x)


class KNN(Proj):
    def __init__(self, k: int, kernel_cfg: dict, configure_kernel: bool=False):
        from .loaders import load_kernel

        self.configure = True
        self.params = {
            'k': k, 
            'kernel_cfg': kernel_cfg, 
            'configure_kernel': configure_kernel
        }
        self.kernel = load_kernel(kernel_cfg)

    def fit(self, X_train: torch.tensor) -> torch.tensor:
        self.X_train = X_train
        if self.params['configure_kernel']:
            _ = self.kernel(X_train, configure=True)
    
    def __call__(self, x: torch.tensor) -> torch.tensor:
        distances = self.kernel.max - self.kernel(x, self.X_train)
        sorted_distances = distances.sort(-1).values
        return sorted_distances[:,self.params['k']]


class PCA(Proj):
    # TODO: This could easily generalised to kernel PCA
    def __init__(self, n_pcs: int):
        self.configure = True
        self.params = {'n_pcs': n_pcs}

    def fit(self, X_train: torch.tensor) -> torch.tensor:
        U, S, V = torch.pca_lowrank(X_train)
        self.redidual_pcs = V[:, self.params['n_pcs']:]

    def __call__(self, x: torch.tensor) -> torch.tensor:
        residuals = torch.matmul(x, self.redidual_pcs)
        return residuals.square().sum(-1)


class UAE(Proj):

    def __init__(self, image_shape: List[int], enc_dim: int):

        self.params = {
            'image_shape': image_shape,
            'enc_dim': enc_dim
            }
        
        self.encoder_net = nn.Sequential(
            nn.Conv2d(image_shape[-1], 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 512, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, enc_dim)
        )
        self.encoder_net.requires_grad = False
        
    def __call__(self, x: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            x_proj =  self.encoder_net(x)
        return x_proj


class TAE(Proj):

    def __init__(self, image_shape: List[int], enc_dim: int):
        self.configure = True
        self.params = {
            'image_shape': image_shape,
            'enc_dim': enc_dim
            }

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 5, stride=3, padding=1),    # [batch, 8, 32, 32]
            nn.ReLU(),
            nn.Conv2d(8, 12, 4, stride=2, padding=1),   # [batch, 12, 16, 16]
            nn.ReLU(),
			nn.Conv2d(12, 16, 4, stride=2, padding=1),   # [batch, 16, 8, 8]
            nn.ReLU(),
			nn.Conv2d(16, 20, 4, stride=2, padding=1),   # [batch, 20, 4, 4]
            nn.ReLU(),
 			nn.Conv2d(20, enc_dim, 4, stride=1, padding=0),   # [batch, enc_dim, 1, 1]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(enc_dim, 20, 4, stride=1, padding=0),  # [batch, 20, 4, 4]
            nn.ReLU(),
			nn.ConvTranspose2d(20, 16, 4, stride=2, padding=1),  # [batch, 16, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(16, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
			nn.ConvTranspose2d(12, 8, 4, stride=2, padding=1),  # [batch, 8, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 5, stride=3, padding=1),   # [batch, 3, 96, 96]
            nn.Sigmoid(),
        )
        self.ae = nn.Sequential(
            self.encoder,
            self.decoder
        )

    def fit(
        self, 
        data: torch.tensor, 
        epochs: int, 
        lr: float=0.003, 
        batch_size: int=32,
        val_prop: float=0.2,
        verbose: bool=False
    ):
        from torch.utils.data import TensorDataset, DataLoader
        n_train = int(data.shape[0]*val_prop)
        train_data, val_data = torch.split(data, [n_train, data.shape[0]-n_train])
        train_dl = DataLoader(TensorDataset(train_data), batch_size, shuffle=True)
        val_dl = DataLoader(TensorDataset(val_data), batch_size, shuffle=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optim = torch.optim.Adam(self.ae.parameters(), lr=lr)

        self.ae = self.ae.to(device)

        for epoch in range(epochs):

            running_loss = 0.
            self.ae.train()
            for i, (x,) in enumerate(train_dl):
                x = x.to(device)
                optim.zero_grad()
                preds = self.ae(x)
                loss = (x-preds).square().mean()
                loss.backward()
                optim.step()

                running_loss += loss.item()
                if i % 10 == 0 and verbose:
                    print('[%d, %5d] loss: %.5f' %
                        (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.
            
            self.ae.eval()
            val_loss = 0.
            for i, (x,) in enumerate(val_dl):
                x = x.to(device)
                preds = self.ae(x)
                loss = (x-preds).square().mean()
                val_loss += loss.item()
            print(f'Epoch: {epoch}/{epochs}, Val mse: {val_loss/len(val_dl):.2g}')


        self.ae = self.ae.to('cpu')
        self.ae.eval()
        
    def __call__(self, x: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            x_proj =  nn.Flatten()(self.encoder(x))
        return x_proj