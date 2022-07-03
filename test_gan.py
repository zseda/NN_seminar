import torch
from torchvision import transforms as transforms
from pathlib import Path
from model_dcgan import Generator
from torch.autograd import Variable
import typer
import uuid
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision.io import read_image


def main(
    root_path: str = typer.Option('.'),
    batch_size: int = typer.Option(100),
    z_dim: int = typer.Option(100),
    experiment_id: str = typer.Option(f"debug-{uuid.uuid4()}"),
    dataset_size: int = typer.Option(60000)

):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"batch_size: {batch_size}")
    tb_path = Path(root_path, "logs", experiment_id)
    tb_path.mkdir(parents=True, exist_ok=False)
    tb_writer = SummaryWriter(log_dir=tb_path.as_posix())
    logger.info(f"experiment id: {experiment_id}")

    full_gan = Generator(g_input_dim=z_dim)
    full_gan.load_state_dict(torch.load(
        "logs/weightnorm/model_epoch_G500.pth", map_location=torch.device('cpu')))
    full_gan.eval()

    path_full_gan_predictions = Path("./full_gan_predictions")
    path_full_gan_predictions.mkdir(parents=True, exist_ok=True)

    to_pil = transforms.ToPILImage()

    with torch.no_grad():
        for i in range(1, int(dataset_size/batch_size)):
            # generate random noise for G
            z = Variable(torch.randn(batch_size, z_dim).to(device))
            # create class labels for generator - uniform distribution
            labels_test = torch.randint(low=0, high=9, size=(batch_size,))
            # one-hot encode class labels
            labels_test_onehot = F.one_hot(
                labels_test, num_classes=10).float().to(device)
            G_out, G_out_logits = full_gan(
                z, labels_test_onehot)
            generated_samples = (G_out + 1)/2
            for im in generated_samples:
                im = to_pil(im)
                im.save(Path(path_full_gan_predictions,
                             f"prediction-{i}.png").as_posix())


if __name__ == "__main__":
    typer.run(main)
