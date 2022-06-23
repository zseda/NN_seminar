import typer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable 
from loguru import logger

from data_loader import get_dataloader
from model import Discriminator, Generator



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




