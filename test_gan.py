import torch
from torchvision import transforms as transforms
from pathlib import Path, PurePath
from model_dcgan import Generator
from torch.autograd import Variable
import typer
import uuid
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision.io import read_image
import os
from pathlib import Path


def main(
    root_path: str = typer.Option('.'),
    batch_size: int = typer.Option(512),
    z_dim: int = typer.Option(100),
    experiment_id: str = typer.Option(f"debug-{uuid.uuid4()}"),
    dataset_size: int = typer.Option(60000),
    # full 80percent 60percent 40percent 20percent
    dataset_type: str = typer.Option("full")

):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"experiment id: {experiment_id}")
    logger.info(f"dataset_size: {dataset_size}")
    logger.info(f"dataset_type: {dataset_type}")

    to_pil = transforms.ToPILImage()
    #
    if dataset_type == "full":
        gan = Generator(g_input_dim=z_dim)
        gan.load_state_dict(torch.load(
            "logs/weightnorm/model_epoch_G500.pth", map_location=torch.device('cpu')))
        gan.eval()

        path_gan_predictions = Path("./full_gan_preds")
        path_gan_predictions.mkdir(parents=True, exist_ok=True)

    if dataset_type == "80percent":
        gan = Generator(g_input_dim=z_dim)
        gan.load_state_dict(torch.load(
            "logs/part80datatrain/model_epoch_G500.pth", map_location=torch.device('cpu')))
        gan.eval()

        path_gan_predictions = Path("./80percent_gan_preds")
        path_gan_predictions.mkdir(parents=True, exist_ok=True)

    if dataset_type == "60percent":
        gan = Generator(g_input_dim=z_dim)
        gan.load_state_dict(torch.load(
            "logs/part60datatrain/model_epoch_G500.pth", map_location=torch.device('cpu')))
        gan.eval()

        path_gan_predictions = Path("./60percent_gan_preds")
        path_gan_predictions.mkdir(parents=True, exist_ok=True)

    if dataset_type == "40percent":
        gan = Generator(g_input_dim=z_dim)
        gan.load_state_dict(torch.load(
            "logs/part40datatrain/model_epoch_G500.pth", map_location=torch.device('cpu')))
        gan.eval()

        path_gan_predictions = Path("./40percent_gan_preds")
        path_gan_predictions.mkdir(parents=True, exist_ok=True)

    if dataset_type == "20percent":
        gan = Generator(g_input_dim=z_dim)
        gan.load_state_dict(torch.load(
            "logs/part20datatrain/model_epoch_G500.pth", map_location=torch.device('cpu')))
        gan.eval()

        path_gan_predictions = Path("./40percent_gan_preds")
        path_gan_predictions.mkdir(parents=True, exist_ok=True)

    one_hot_labels = torch.eye(10)
    l = 0
    c = 1
    with torch.no_grad():
        for i in range(1, int(dataset_size/batch_size)):
            for label in one_hot_labels:
                # generate random noise for G
                z = Variable(torch.randn(
                    int(batch_size/one_hot_labels.size(dim=0))), z_dim).to(device)
                # dublicate label batchsize/one_hot_labels_size
                # every label directory has same amount of image
                label = label.repeat(
                    int(batch_size/one_hot_labels.size(dim=0))).view(int(batch_size/one_hot_labels.size(dim=0)), 10)
                # generate images
                G_out, G_out_logits = gan(
                    z, label)
                generated_samples = (G_out + 1)/2
                # create label directories
                label_dir = Path.joinpath(
                    path_gan_predictions, Path("./label{l}"))
                label_dir.mkdir(parents=True, exist_ok=True)
                l = l+1
                # save every generated images to corresponding directory
                for im in generated_samples:
                    im = to_pil(im)
                    im.save(Path(path_gan_predictions,
                                 f"prediction-{c}.png").as_posix())
                    c = c+1


if __name__ == "__main__":
    typer.run(main)
