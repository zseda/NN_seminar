import typer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from loguru import logger

from data_loader import get_dataloader
from matplotlib import pyplot
from model import Discriminator, Generator

# plot the generated images


def create_plot(examples, n):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        image = examples[i, :, :].detach().numpy()
        image = image[0]/255
        pyplot.imshow(image)
    pyplot.show()


def main(
    epochs: int = typer.Option(20),
    batch_size: int = typer.Option(100),
    lr: float = typer.Option(0.0002),
    z_dim: int = typer.Option(100)

):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"batch_size: {batch_size}")

    # load data
    loader_train, loader_test, mnist_dim = get_dataloader(
        batch_size=batch_size)

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

    D_losses, G_losses = [], []
    for e in range(1, epochs+1):
        logger.info(f"training epoch {e}/{epochs}")
        for batch in loader_train:
            img, class_idx = batch
            img = img.to(device)
            class_idx = class_idx.to(device)
            x_real, y_real = Variable(
                img.view(-1, mnist_dim).to(device)), Variable(torch.ones(batch_size, 1).to(device))
            z = Variable(torch.randn(batch_size, z_dim).to(device))
            x_fake, y_fake = G(z), Variable(
                torch.zeros(batch_size, 1).to(device))

            # discriminator
            D_out = D(img)
            D_real_loss = criterion(D_out, y_real)
            D_out = D(x_fake)
            D_fake_loss = criterion(D_out, y_fake)

            # gradient backprop & optimize ONLY D's parameters
            D_loss = D_real_loss + D_fake_loss
            D_loss.backward()
            D_optimizer.step()
            D_losses.append(D_loss.data.item())

            # generate images via G
            G_output = G(z)
            D_out = D(G_output)
            G_loss = criterion(D_out, y_real)

            # gradient backprop & optimize ONLY G's parameters
            G_loss.backward()
            G_optimizer.step()
            G_losses.append(G_loss.data.item())
        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (e), epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

        # scale from [-1,1] to [0,1]
        #G_output = (G_output + 1) / 2.0
        # plot the result
        #create_plot(G_output, 10)


if __name__ == "__main__":
    typer.run(main)
