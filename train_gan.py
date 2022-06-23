import typer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from loguru import logger

from data_loader import get_dataloader
from model import Discriminator, Generator


def D_train(D, G, x, batch_size, device, z_dim, criterion, mnist_dim, D_optimizer):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(batch_size, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on synthetic
    z = Variable(torch.randn(batch_size, z_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(batch_size, 1).to(device))

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


def G_train(D, G, batch_size, device, z_dim, criterion, G_optimizer):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = Variable(torch.randn(batch_size, z_dim).to(device))
    y = Variable(torch.ones(batch_size, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()


def main(
    epochs: int = typer.Option(20),
    batch_size: int = typer.Option(16),
    lr: float = typer.Option(0.0002),
    z_dim: int = typer.Option(100)
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"batch_size: {batch_size}")

    # load data
    loader_train, loader_test = get_dataloader(batch_size=batch_size)

    # initialize G
    G = Generator(g_input_dim=z_dim)
    G.to(device)

    # initialize D
    D = Discriminator()
    D.to(device)

    # optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr)
    D_optimizer = optim.Adam(D.parameters(), lr=lr)

    # loss
    criterion = nn.BCELoss()

    for e in range(1, epochs+1):
        logger.info(f"training epoch {e}/{epochs}")
        for batch in loader_train:
            img, class_idx = batch
            img = img.to(device)
            class_idx = class_idx.to(device)
            z = Variable(torch.randn(batch_size, z_dim).to(device))

            # generate images via G
            generated_images = G(z)

            # discriminator
            D_out = D(generated_images)


if __name__ == "__main__":
    typer.run(main)
