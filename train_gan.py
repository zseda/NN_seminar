import typer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from loguru import logger
from pathlib import Path
import uuid
from data_loader import get_dataloader
from matplotlib import pyplot
from model import Discriminator, Generator
from torchvision.utils import make_grid
from model import weights_init_normal
from torch.utils.tensorboard import SummaryWriter


def main(
    root_path: str = typer.Option('.'),
    epochs: int = typer.Option(20),
    batch_size: int = typer.Option(100),
    lr: float = typer.Option(9e-6),
    z_dim: int = typer.Option(100),
    experiment_id: str = typer.Option(f"debug-{uuid.uuid4()}"),

):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"batch_size: {batch_size}")
    tb_path = Path(root_path, "logs", experiment_id)
    tb_path.mkdir(parents=True, exist_ok=False)
    tb_writer = SummaryWriter(log_dir=tb_path.as_posix())

    # load data
    loader_train, loader_test, mnist_dim = get_dataloader(
        batch_size=batch_size)

    # initialize G
    G = Generator(g_input_dim=z_dim)
    G.to(device)
    G.apply(weights_init_normal)

    # initialize D
    D = Discriminator()
    D.to(device)
    D.apply(weights_init_normal)

    # optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr)
    D_optimizer = optim.Adam(D.parameters(), lr=lr)

    # loss
    criterion = nn.BCELoss()
    global_step = 0
    D_losses, G_losses = [], []
    for e in range(1, epochs+1):
        logger.info(f"training epoch {e}/{epochs}")
        for batch in loader_train:
            """
                -----------
                preparation
                -----------
            """
            # for logging
            global_step += 1

            # decompose batch data
            img, class_idx = batch
            img = img.to(device)
            class_idx = class_idx.to(device)

            # reset gradients
            D.zero_grad()
            G.zero_grad()

            # create labels
            y_real = Variable(torch.ones(batch_size, 1).to(device))
            y_fake = Variable(torch.zeros(batch_size, 1).to(device))

            # create noise for G
            z = Variable(torch.randn(batch_size, z_dim).to(device))

            # generate images for D
            x_fake = G(z)

            """ 
                -------
                train D
                -------
            """
            D_out = D(img)
            D_real_loss = criterion(D_out, y_real)

            D_out = D(x_fake)
            D_fake_loss = criterion(D_out, y_fake)

            # gradient backprop & optimize ONLY D's parameters
            D_loss = D_real_loss + D_fake_loss
            D_loss.backward()
            D_optimizer.step()

            """ 
                -------
                train G
                -------
            """
            # reset gradients
            D.zero_grad()
            G.zero_grad()

            # generate images via G
            z = Variable(torch.randn(batch_size, z_dim).to(device))
            G_output = G(z)
            D_out = D(G_output)
            G_loss = criterion(D_out, y_real)

            # gradient backprop & optimize ONLY G's parameters
            G_loss.backward()
            G_optimizer.step()

            """
                -------
                logging
                -------
            """
            plot_img = (img + 1.0) / 2.0
            plot_output = (G_output + 1.0) / 2.0
            
            if global_step % 50 == 0:
                tb_writer.add_scalar(
                    'train/disciriminator_loss', D_loss.item(), global_step=global_step)
                tb_writer.flush()
            D_losses.append(D_loss.data.item())
            if global_step % 50 == 0:
                tb_writer.add_scalar(
                    'train/generator_loss', G_loss.item(), global_step=global_step)
                tb_writer.flush()
            G_losses.append(G_loss.data.item())

            if global_step % 100 == 0:
                tb_writer.add_image(
                    f"train/img", make_grid(plot_img), global_step=global_step)
                tb_writer.add_image(
                    f"train/pred", make_grid(plot_output), global_step=global_step)

        """
            ------
            saving
            ------
        """
        # save D
        if e % 1 == 0:
            torch.save(D.state_dict(), Path(root_path, "logs",
                        experiment_id, f"model_epoch_D{e:0>3}.pth").as_posix())

        # save G
        if e % 1 == 0:
            torch.save(G.state_dict(), Path(root_path, "logs",
                        experiment_id, f"model_epoch_G{e:0>3}.pth").as_posix())


if __name__ == "__main__":
    typer.run(main)
