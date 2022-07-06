from matplotlib import image
from torchvision import transforms as transforms
from pathlib import Path, PurePath
import typer
import uuid
from loguru import logger
import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class DatasetType(str, Enum):
    full = "full"
    percent80 = "percent80"
    percent60 = "percent60"
    percent40 = "percent40"
    percent20 = "percent20"


class labels_dir(str, Enum):
    Tshirt_Top = "label_0"
    Trouser = "label_1"
    Pullover = "label_2"
    Dress = "label_3"
    Coat = "label_4"
    Sandal = "label_5"
    Shirt = "label_6"
    Sneaker = "label_7"
    Bag = "label_8"
    Ankle_boot = "label_9"


@dataclass
class DataSpecs:
    dataset_type: DatasetType
    data_path: Path


SPECS = {
    DatasetType.full: DataSpecs(
        dataset_type=DatasetType.full,
        data_path=Path("predictions/full")
    ),
    DatasetType.percent80: DataSpecs(
        dataset_type=DatasetType.percent80,
        data_path=Path("predictions/percent80")
    ),
    DatasetType.percent60: DataSpecs(
        dataset_type=DatasetType.percent60,
        data_path=Path("predictions/percent60")
    ),
    DatasetType.percent40: DataSpecs(
        dataset_type=DatasetType.percent40,
        data_path=Path("predictions/percent40")
    ),
    DatasetType.percent20: DataSpecs(
        dataset_type=DatasetType.percent20,
        data_path=Path("predictions/percent20")
    ),
}


def main(
    experiment_id: str = typer.Option(f"debug-{uuid.uuid4()}"),
    dataset_type: DatasetType = typer.Option(DatasetType.full),
    img_size: int = typer.Option(28)

):

    logger.info(f"experiment id: {experiment_id}")
    data_path = SPECS[dataset_type].data_path
    logger.info(f"dataset_type: {dataset_type}")

    logger.info(f"image size: {img_size} * {img_size}")

    images = []
    labels = []

    def assign_label(img, label):
        return label

    def make_train_data(label, DIR):
        for img in tqdm(os.listdir(DIR)):
            path = os.path.join(DIR, img)
            img = cv2.imread(path, 0)
            images.append(np.array(img))
            labels.append(str(label))

    make_train_data('0', Path(data_path, "label_0"))
    make_train_data('1', Path(data_path, "label_1"))
    make_train_data('2', Path(data_path, "label_2"))
    make_train_data('3', Path(data_path, "label_3"))
    make_train_data('4', Path(data_path, "label_4"))
    make_train_data('5', Path(data_path, "label_5"))
    make_train_data('6', Path(data_path, "label_6"))
    make_train_data('7', Path(data_path, "label_7"))
    make_train_data('8', Path(data_path, "label_8"))
    make_train_data('9', Path(data_path, "label_9"))

    #labels = torch.from_numpy(labels)
    #onehot_labels = F.one_hot(labels, num_classes=10)

    # USE A LABELENCODER IF YOU WANT HERE

    holder = zip(images, labels)


if __name__ == "__main__":
    typer.run(main)
