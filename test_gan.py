from itertools import cycle
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
from enum import Enum
from dataclasses import dataclass, field
from tqdm import tqdm


class DatasetType(Enum):
    full = "full"
    percent80 = "percent80"
    percent60 = "percent60"
    percent40 = "percent40"
    percent20 = "percent20"


@dataclass
class ModelSpecs:
    dataset_type: DatasetType
    dataset_size: int
    model_path: Path
    pred_path: Path = field(init=False)

    def __post_init__(self):
        self.pred_path = Path("./predictions", self.dataset_type.value)


SPECS = {
    DatasetType.full: ModelSpecs(
        dataset_type=DatasetType.full,
        dataset_size=60000,
        model_path=Path("logs/weightnorm/model_epoch_G500.pth")
    ),
    DatasetType.percent80: ModelSpecs(
        dataset_type=DatasetType.percent80,
        dataset_size=48000,
        model_path=Path("logs/part80datatrain/model_epoch_G500.pth")
    ),
    DatasetType.percent60: ModelSpecs(
        dataset_type=DatasetType.percent60,
        dataset_size=36000,
        model_path=Path("logs/part60datatrain/model_epoch_G500.pth")
    ),
    DatasetType.percent40: ModelSpecs(
        dataset_type=DatasetType.percent40,
        dataset_size=24000,
        model_path=Path("logs/part40datatrain/model_epoch_G500.pth")
    ),
    DatasetType.percent20: ModelSpecs(
        dataset_type=DatasetType.percent20,
        dataset_size=12000,
        model_path=Path("logs/part20datatrain/model_epoch_G500.pth")
    ),
}


def main(
    root_path: str = typer.Option('.'),
    batch_size: int = typer.Option(512),
    z_dim: int = typer.Option(100),
    experiment_id: str = typer.Option(f"debug-{uuid.uuid4()}"),
    dataset_type: DatasetType = typer.Option(DatasetType.full)
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"experiment id: {experiment_id}")
    dataset_size = SPECS[dataset_type].dataset_size
    logger.info(f"dataset_size: {dataset_size}")
    logger.info(f"dataset_type: {dataset_type}")

    to_pil = transforms.ToPILImage()
    #
    gan = Generator(g_input_dim=z_dim)
    gan.load_state_dict(torch.load(SPECS[dataset_type].model_path.as_posix(), map_location=torch.device(device)))
    gan.eval()

    path_gan_predictions = SPECS[dataset_type].pred_path
    path_gan_predictions.mkdir(parents=True, exist_ok=True)
    
    # create label directories
    for label_id in range(10):
        label_dir = Path(path_gan_predictions, f"label{label_id}")
        label_dir.mkdir(parents=True, exist_ok=True)

    # one-hot label matrix
    one_hot_labels = torch.eye(10)

    # counter variable
    ds_iterator = zip(
        range(1, dataset_size + 1),
        cycle(one_hot_labels),
        cycle(range(10))
    )
    with torch.no_grad():
        for i, label, label_id in tqdm(ds_iterator, total=dataset_size):
            # generate random noise for G
            z = torch.randn(size=(1, z_dim)).to(device)

            # generate images
            G_out, G_out_logits = gan(z, label)
            generated_samples = (G_out + 1)/2

            # save every generated images to corresponding directory
            im = to_pil(generated_samples[0])
            im.save(Path(path_gan_predictions, f"label_{label_id}", f"prediction-{i}.png").as_posix())


if __name__ == "__main__":
    typer.run(main)
