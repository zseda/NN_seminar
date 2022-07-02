import typer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from loguru import logger
from pathlib import Path
import uuid
from data_loader import get_dataloader
from model_dcgan import Discriminator, Generator
from torchvision.utils import make_grid
from model_dcgan import weights_init_normal
from torch.utils.tensorboard import SummaryWriter
import timm


def main(
    root_path: str = typer.Option('.'),
    epochs: int = typer.Option(20),
    batch_size: int = typer.Option(100),
    lr: float = typer.Option(1e-4),
    z_dim: int = typer.Option(100),
    experiment_id: str = typer.Option(f"debug-{uuid.uuid4()}"),

):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"batch_size: {batch_size}")
    tb_path = Path(root_path, "logs", experiment_id)
    tb_path.mkdir(parents=True, exist_ok=False)
    tb_writer = SummaryWriter(log_dir=tb_path.as_posix())
    logger.info(f"experiment id: {experiment_id}")

    # load data
    loader_train, loader_test, mnist_dim = get_dataloader(
        batch_size=batch_size)

    # classifier
    C = timm.create_model("efficientnet_b0", pretrained=True, num_classes=10, in_chans=1)
    C.to(device)

    # initialize G
    G = Generator(g_input_dim=z_dim)
    G.to(device)
    G.apply(weights_init_normal)

    # initialize D
    D = Discriminator()
    D.to(device)
    D.apply(weights_init_normal)

    # optimizer
    C_optimizer = optim.Adam(C.parameters(), lr=lr, betas=(0.5, 0.999))
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # loss
    criterion_classification = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    global_step = 0
    D_losses, G_losses = [], []

    # test labels
    labels_test = list()
    eye10 = torch.eye(10)
    for i in range(10):
        labels_test.append(eye10[i].repeat(8).view(8, 10))
    labels_test = torch.stack(labels_test).view(-1, 10).float().to(device)

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
            img, labels_real = batch
            img = img.to(device)
            labels_real = labels_real.to(device)
            labels_real_onehot = F.one_hot(labels_real, num_classes=10).float().to(device)

            actual_batch_size = img.shape[0]

            # reset gradients
            D.zero_grad()
            G.zero_grad()
            C.zero_grad()

            # create labels
            y_real = Variable(torch.ones(actual_batch_size, 1).to(device))
            y_fake = Variable(torch.zeros(actual_batch_size, 1).to(device))

            # generate random noise for G
            z = Variable(torch.randn(actual_batch_size, z_dim).to(device))

            # create class labels for generator - uniform distribution
            labels_fake = torch.randint(low=0, high=9, size=(actual_batch_size,))

            # one-hot encode class labels
            labels_fake_onehot = F.one_hot(labels_fake, num_classes=10).float().to(device)

            # generate images for D
            x_fake, x_fake_logits = G(z, labels_fake_onehot)

            """
                -------
                train C
                -------
            """
            C_out = C(img)
            C_loss = criterion_classification(C_out, labels_real_onehot)
            C_loss.backward()
            C_optimizer.step()

            """ 
                -------
                train D
                -------
            """
            D_out_real = D(img, labels_real_onehot)
            D_real_loss = criterion(D_out_real, y_real)

            D_out_fake = D(x_fake, labels_fake_onehot)
            D_fake_loss = criterion(D_out_fake, y_fake)

            # gradient backprop & optimize ONLY D's parameters
            D_loss = (D_real_loss + D_fake_loss) / 2.0
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
            C.zero_grad()

            # generate images via G
            # create labels for testing generator
            # convert to one hot encoding
            z = Variable(torch.randn(batch_size, z_dim).to(device))

            G_output, G_output_logits = G(z, labels_fake_onehot)
            D_out = D(G_output, labels_fake_onehot)
            G_disc_loss = criterion(D_out, y_real)

            # test generated images with classifier
            C_out = C(G_output)
            G_classification_loss = criterion_classification(C_out, labels_fake_onehot)

            G_loss = G_disc_loss + G_classification_loss

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

            # print every 50 steps
            if global_step % 50 == 0:
                tb_writer.add_scalar(
                    'train/disciriminator_loss', D_loss.item(), global_step=global_step)
                tb_writer.flush()
            D_losses.append(D_loss.data.item())
            if global_step % 50 == 0:
                tb_writer.add_scalar(
                    'train/C_loss', C_loss.item(), global_step=global_step)
                tb_writer.add_scalar(
                    'train/G_loss', G_loss.item(), global_step=global_step)
                tb_writer.add_scalar(
                    'train/G_disc_loss', G_disc_loss.item(), global_step=global_step)
                tb_writer.add_scalar(
                    'train/G_classification_loss', G_classification_loss.item(), global_step=global_step)
                tb_writer.flush()
            G_losses.append(G_loss.data.item())

            # print every 250 steps
            if global_step % 250 == 0:
                tb_writer.add_image(
                    f"train/img", make_grid(plot_img), global_step=global_step)
                tb_writer.add_image(
                    f"train/pred", make_grid(plot_output), global_step=global_step)

        """
            -------------------------------
            test class generation per epoch
            -------------------------------
        """
        # make prediction - we do not need gradients for that
        with torch.no_grad():
            # create random noise for G to generate images
            z = torch.randn(80, z_dim).to(device)
            # make prediction - use labels test => structured one-hot encoded labels
            G_test_output, G_test_output_logits = G(z, labels_test)
            # normalize images => output of G is tanh so we need to normalize to [0.0, 1.0]
            G_test_output = (G_test_output + 1.0) / 2.0
            # save to TensorBoard
            tb_writer.add_image(
                f"test/pred", make_grid(G_test_output), global_step=e)
            # flush cache in case anything is hanging in cache
            tb_writer.flush()
            

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
