import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    # NOTE: (size, x, x, x) are being computed manually as of now (this is based on output of encoder)
    def forward(self, input, size=512): # size=128
        return input.view(input.size(0), size, 3, 3, 3)
        # return input.view(input.size(0), size, 6, 6, 6)


class CVAE_3D(nn.Module):
    def __init__(self, image_channels=3, h_dim=128, z_dim=32):
        super(CVAE_3D, self).__init__()
        print()
        print("[INFO] instantiating pytorch model: 3D CVAE")

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=image_channels, out_channels=16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            Flatten() # reshape layer
        )

        # fully connected layers to compute mu and sigma
        # z_dim is set by user
        # h_dim should be computed manually based on output of convs (in this case 27648)
        self.fc1 = nn.Linear(27648, z_dim)
        self.fc2 = nn.Linear(27648, z_dim)
        # self.fc1 = nn.Linear(h_dim, z_dim)
        # self.fc2 = nn.Linear(h_dim, z_dim)

        # self.fc3 = nn.Linear(z_dim, h_dim) # dense layer to connect to decoder
        self.fc3 = nn.Linear(z_dim, 27648)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=image_channels, kernel_size=4, stride=1, padding=0), # dimensions should be as original
            nn.BatchNorm3d(num_features=3),
            # nn.Sigmoid(),

            # if it does not work without sigmoid:
                # check another batchnorm or relu
                # recover original dims: use nn.linear and reshape to original size
                # nn.conv3d with kernel size equal to input size
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # std = logvar.mul(0.5).exp_()
        # eps = torch.randn(*mu.size())
        eps = torch.rand_like(std)
        # z = mu + std * eps
        z = eps.mul(std).add_(mu)
        return z

    def bottleneck(self, h):
        # print("[INFO] bottleneck h size:", h.size())
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        # print("[INFO] h size:", h.size()) # torch.Size([10, 27648])
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        # print("[INFO] Input data shape:", x.size())

        # Step 1: compute representation (fetch it separately for later clustering)
        z_representation = self.representation(x)
        # print("[INFO] Forward z_representation:", z_representation.size())
        # print("[INFO] Reshaped latent z", z_representation.view(z_representation.size(0), 8, 8).size())

        # Step 2: call full CVAE --> encode & decode
        z, mu, logvar = self.encode(x)
        z = self.fc3(z)
        # print("[INFO] Latent z after dense fc:", z.size())
        # print("[INFO] mu:", mu.size())
        # print("[INFO] logvar", logvar.size())

        return self.decode(z), mu, logvar, z_representation


class CVAE_3D_II(nn.Module):
    def __init__(self, image_channels=3, h_dim=128, z_dim=32):
        super(CVAE_3D_II, self).__init__()
        print()
        print("[INFO] instantiating pytorch model: 3D CVAE")

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=image_channels, out_channels=32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            Flatten() # reshape layer
        )

        # fully connected layers to compute mu and sigma
        # z_dim is set by user
        # h_dim should be computed manually based on output of convs
        self.fc1 = nn.Linear(13824, z_dim)
        self.fc2 = nn.Linear(13824, z_dim)
        # self.fc1 = nn.Linear(h_dim, z_dim)
        # self.fc2 = nn.Linear(h_dim, z_dim)

        # self.fc3 = nn.Linear(z_dim, h_dim) # dense layer to connect to decoder
        self.fc3 = nn.Linear(z_dim, 13824)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=512, out_channels=256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(num_features=16),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=16, out_channels=image_channels, kernel_size=4, stride=1, padding=0), # dimensions should be as original
            nn.BatchNorm3d(num_features=3))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # std = logvar.mul(0.5).exp_()
        # eps = torch.randn(*mu.size())
        eps = torch.rand_like(std)
        # z = mu + std * eps
        z = eps.mul(std).add_(mu)
        return z

    def bottleneck(self, h):
        # print("[INFO] bottleneck h size:", h.size())
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        # print("[INFO] h size:", h.size()) # torch.Size([10, 27648])
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        # print("[INFO] Input data shape:", x.size())

        # Step 1: compute representation (fetch it separately for later clustering)
        z_representation = self.representation(x)
        # print("[INFO] Forward z_representation:", z_representation.size())
        # print("[INFO] Reshaped latent z", z_representation.view(z_representation.size(0), 8, 8).size())

        # Step 2: call full CVAE --> encode & decode
        z, mu, logvar = self.encode(x)
        z = self.fc3(z)
        # print("[INFO] Latent z after dense fc:", z.size())
        # print("[INFO] mu:", mu.size())
        # print("[INFO] logvar", logvar.size())

        return self.decode(z), mu, logvar, z_representation
