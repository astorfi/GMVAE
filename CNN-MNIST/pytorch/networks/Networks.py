"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Gaussian Mixture Variational Autoencoder Networks

"""
import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from networks.Layers import *

# Inference Network


class InferenceNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim, device):
        super(InferenceNet, self).__init__()

        hidden_dims = [32, 64, 128]

        # Build Encoder
        modules = []
        in_channels = 3
        for h_dim_idx in range(len(hidden_dims)):
            if h_dim_idx == 0:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=hidden_dims[h_dim_idx],
                                kernel_size=3, stride=2, padding=1),
                        nn.LeakyReLU())
                )
                in_channels = hidden_dims[h_dim_idx]
            else:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels=hidden_dims[h_dim_idx],
                                kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(hidden_dims[h_dim_idx]),
                        nn.LeakyReLU())
                )
                in_channels = hidden_dims[h_dim_idx]

        self.encoderCNN = nn.Sequential(*modules)
        self.cnn_to_fc= nn.Sequential(
            nn.Linear(hidden_dims[-1]*16, x_dim),
            nn.BatchNorm1d(num_features=x_dim),
            nn.ReLU()
        )

        # # q(y|x)
        # self.inference_qyx = torch.nn.ModuleList([
        #     nn.Linear(x_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     GumbelSoftmax(512, y_dim, device)
        # ])

        # q(y|x)
        self.inference_qyx = torch.nn.ModuleList([
            GumbelSoftmax(x_dim, y_dim, device)
        ])

        # # q(z|y,x)
        # self.inference_qzyx = torch.nn.ModuleList([
        #     nn.Linear(x_dim + y_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     Gaussian(512, z_dim)
        # ])

        # q(z|y,x)
        self.inference_qzyx = torch.nn.ModuleList([
            nn.Linear(x_dim + y_dim, 512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            Gaussian(512, z_dim)
        ])

    # q(y|x)
    def qyx(self, x, temperature, hard):
        num_layers = len(self.inference_qyx)

        # Feature extraction
        x = self.encoderCNN(x)
        x = torch.flatten(x, start_dim=1)
        x = self.cnn_to_fc(x)

        for i, layer in enumerate(self.inference_qyx):
            if i == num_layers - 1:
                # last layer is gumbel softmax
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x

    # q(z|x,y)
    def qzxy(self, x, y):

        # Feature extraction
        x = self.encoderCNN(x)
        x = torch.flatten(x, start_dim=1)
        x = self.cnn_to_fc(x)

        concat = torch.cat((x, y), dim=1)
        for layer in self.inference_qzyx:
            concat = layer(concat)
        return concat

    def forward(self, x, temperature=1.0, hard=0):
        #x = Flatten(x)

        # q(y|x)
        logits, prob, y = self.qyx(x, temperature, hard)

        # q(z|x,y)
        mu, var, z = self.qzxy(x, y)

        output = {'mean': mu, 'var': var, 'gaussian': z,
                  'logits': logits, 'prob_cat': prob, 'categorical': y}
        return output


# Generative Network
class GenerativeNet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim, device):
        super(GenerativeNet, self).__init__()

        # p(z|y)
        self.y_mu = nn.Linear(y_dim, z_dim)
        self.y_var = nn.Linear(y_dim, z_dim)

        # p(x|z)
        self.generative_pxz = torch.nn.ModuleList([
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, x_dim),
            nn.ReLU()
        ])

        # Build Decoder
        modules = []

        self.hidden_dims = [32, 64, 128]
        self.fc_to_cnn = nn.Linear(x_dim, self.hidden_dims[-1] * 16)

        self.hidden_dims.reverse()

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dims[i],
                                      self.hidden_dims[i + 1],
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1),
                    nn.BatchNorm2d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoderCNN = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(self.hidden_dims[-1],
                                               self.hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(self.hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(self.hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            torch.nn.Sigmoid())

    # p(z|y)

    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))
        return y_mu, y_var

    # p(x|z)
    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)

        z = self.fc_to_cnn(z)
        z = z.view(-1, self.hidden_dims[0], 4, 4)
        z = self.decoderCNN(z)
        z = self.final_layer(z)

        return z

    def forward(self, z, y):
        # p(z|y)
        y_mu, y_var = self.pzy(y)

        # p(x|z)
        x_rec = self.pxz(z)

        output = {'y_mean': y_mu, 'y_var': y_var, 'x_rec': x_rec}
        return output


# GMVAE Network
class GMVAENet(nn.Module):
    def __init__(self, x_dim, z_dim, y_dim, device):
        super(GMVAENet, self).__init__()

        self.inference = InferenceNet(x_dim, z_dim, y_dim, device)
        self.generative = GenerativeNet(x_dim, z_dim, y_dim, device)

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, temperature=1.0, hard=0):
        # x = x.view(x.size(0), -1)
        out_inf = self.inference(x, temperature, hard)
        z, y = out_inf['gaussian'], out_inf['categorical']
        out_gen = self.generative(z, y)

        # merge output
        output = out_inf
        for key, value in out_gen.items():
            output[key] = value
        return output
