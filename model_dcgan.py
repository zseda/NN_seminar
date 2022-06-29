import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from loguru import logger

'''
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        logger.debug(f"init conv weights '{classname}'")
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        logger.debug(f"init batchnorm weights '{classname}'")
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias.data, 0.0)


def weight_init(self, layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(layer.bias.data, 0.0)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.constant_(layer.weight, 1.0)
        nn.init.constant_(layer.bias.data, 0.0)

'''


class Block(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels/2),
                            kernel_size=3, stride=1, padding="same")
        self.b1 = nn.BatchNorm2d(num_features=int(in_channels/2))

        self.c2 = nn.Conv2d(in_channels=int(in_channels/2), out_channels=int(in_channels/2),
                            kernel_size=3, stride=1, padding="same")
        self.b2 = nn.BatchNorm2d(num_features=int(in_channels/2))

        self.c3 = nn.Conv2d(in_channels=int(in_channels/2), out_channels=in_channels,
                            kernel_size=3, stride=1, padding="same")
        self.b3 = nn.BatchNorm2d(num_features=in_channels)

    def forward(self, x):
        features_in = x

        x = self.c1(x)
        x = self.b1(x)
        x = F.leaky_relu(x)

        x = self.c2(x)
        x = self.b2(x)
        x = F.leaky_relu(x)

        x = self.c3(x)
        x = self.b3(x)
        x = F.leaky_relu(x)

        x = features_in + x

        return x


class Generator(nn.Module):
    def __init__(self, g_input_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(in_features=g_input_dim, out_features=7*7*32)
        self.fc2 = nn.Linear(in_features=10, out_features=7*7*32)

        self.block1 = Block(in_channels=64)
        self.block2 = Block(in_channels=64)
        self.block3 = Block(in_channels=64)
        self.c4 = nn.Conv2d(in_channels=64, out_channels=1,
                            kernel_size=3, stride=1, padding="same")

    def forward(self, x, label):

        x = F.leaky_relu(self.fc1(x), 0.2)

        # transform to 2D space
        x = x.view(-1, 32, 7, 7)

        # y size : (batch_size, 10)
        y = F.leaky_relu(self.fc2(label), 0.2)
        # transform to 2D space
        y = y.view(-1, 32, 7, 7)

        # concat x and y
        x = torch.cat([x, y], dim=1)
        # xy size : (batch_size, 128, 7, 7)

        # apply convolutions and upsample
        x = self.block1(x)
        x = F.upsample_bilinear(x, size=(14, 14))

        x = self.block2(x)
        x = F.upsample_bilinear(x, size=(28, 28))

        x = self.block3(x)

        return torch.tanh(self.c4(x))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=64,
                            kernel_size=3, stride=2, padding=1)
        self.b1 = nn.BatchNorm2d(num_features=64)

        self.c2 = nn.Conv2d(in_channels=64, out_channels=128,
                            kernel_size=3, stride=2, padding=1)
        self.b2 = nn.BatchNorm2d(num_features=128)

        self.c3 = nn.Conv2d(in_channels=128, out_channels=256,
                            kernel_size=3, stride=1, padding="same")
        self.b3 = nn.BatchNorm2d(num_features=256)

        self.c4 = nn.Conv2d(in_channels=256, out_channels=128,
                            kernel_size=3, stride=1, padding="same")
        self.b4 = nn.BatchNorm2d(num_features=128)

        self.fc_num_features = 128*7*7
        self.fc1 = nn.Linear(in_features=10, out_features=7*7*32)
        self.fc1 = nn.Linear(128*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1)
    # forward method

    def forward(self, x, label):
        # x => [N, 1, 28, 28]

        x = F.leaky_relu(self.b1(self.c1(x)), 0.2)
        # x => [N, 32, 14, 14]
        x = F.dropout(x, 0.3)

        x = F.leaky_relu(self.b2(self.c2(x)), 0.2)
        # x => [N, 128, 7, 7]
        x = F.dropout(x, 0.3)

        x = F.leaky_relu(self.b3(self.c3(x)), 0.2)
        # x => [N, 256, 7, 7]
        x = F.dropout(x, 0.3)

        x = F.leaky_relu(self.b4(self.c4(x)), 0.2)
        x = F.dropout(x, 0.3)
        # x => [N, 128, 7, 7]

        x = x.view(-1, self.fc_num_features)
        # x => [N, 128*7*7]
        y_ = self.fc3(label)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return F.sigmoid(x)
