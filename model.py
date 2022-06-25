import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight, 1.0, 0.02)
        torch.nn.init.constant(m.bias, 0.0)


class Generator(nn.Module):
    def __init__(self, g_input_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(in_features=g_input_dim, out_features=7*7)
        self.c1 = nn.Conv2d(in_channels=1, out_channels=256,
                            kernel_size=3, stride=1, padding="same")
        self.c2 = nn.Conv2d(in_channels=256, out_channels=128,
                            kernel_size=3, stride=1, padding="same")
        self.c3 = nn.Conv2d(in_channels=128, out_channels=64,
                            kernel_size=3, stride=1, padding="same")
        self.c4 = nn.Conv2d(in_channels=64, out_channels=1,
                            kernel_size=3, stride=1, padding="same")

    def forward(self, x):
        # input is noise
        x = F.leaky_relu(self.fc1(x), 0.2)

        # transform to 2D space
        x = x.view(-1, 1, 7, 7)

        # apply convolutions and upsample
        x = F.leaky_relu(self.c1(x), 0.2)

        x = F.upsample_bilinear(x, size=(14, 14))
        x = F.leaky_relu(self.c2(x), 0.2)

        x = F.upsample_bilinear(x, size=(28, 28))
        x = F.leaky_relu(self.c3(x), 0.2)

        return torch.tanh(self.c4(x))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=64,
                            kernel_size=3, stride=2, padding=1)
        self.c2 = nn.Conv2d(in_channels=64, out_channels=128,
                            kernel_size=3, stride=2, padding=1)
        self.c3 = nn.Conv2d(in_channels=128, out_channels=256,
                            kernel_size=3, stride=1, padding="same")
        self.c4 = nn.Conv2d(in_channels=256, out_channels=128,
                            kernel_size=3, stride=1, padding="same")

        self.fc_num_features = 128*7*7
        self.fc1 = nn.Linear(in_features=self.fc_num_features, out_features=1)

    # forward method
    def forward(self, x):
        # x => [N, 1, 28, 28]

        x = F.leaky_relu(self.c1(x), 0.2)
        # x => [N, 64, 14, 14]
        x = F.dropout(x, 0.3)

        x = F.leaky_relu(self.c2(x), 0.2)
        # x => [N, 128, 7, 7]
        x = F.dropout(x, 0.3)

        x = F.leaky_relu(self.c3(x), 0.2)
        # x => [N, 256, 7, 7]
        x = F.dropout(x, 0.3)

        x = F.leaky_relu(self.c4(x), 0.2)
        x = F.dropout(x, 0.3)
        # x => [N, 128, 7, 7]

        x = x.view(-1, self.fc_num_features)
        # x => [N, 128*7*7]

        return torch.sigmoid(self.fc1(x))
