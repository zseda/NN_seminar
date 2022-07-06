import imp
from unittest import loader
import torch
import typer
import timm
from torchinfo import summary
from loguru import logger
from pathlib import Path
from dataloader_real import get_dataloader
from dataloader_synthetic import get_dataset
from dataloader_synthetic import DatasetType
from torch.utils.tensorboard import SummaryWriter
import uuid
import torch


def train_test_classifier(loader_train, loader_test, device, epochs):
    # classifier
    C = timm.create_model("efficientnet_b0", pretrained=True,
                          num_classes=10, in_chans=1)
    summary(C)
    C.to(device)

    # training loop
    for e in range(epochs):
        for img, label in loader_train:

            # predict
            # loss
            # backward
            # logging

            # testing loop
            # TODO: container for metrics per batch
    e.g. accuracy_list = list()
    for img, label in loader_test:
        # predict
        ...
        # calc metrics
        accuracy_list.append(batch_accuracy)

    # calculate average metrics over all batches (single results in the container)
    average over all accuracy list

    # save final test metrics
    save metrics


def main(
    root_path: str = typer.Option('.'),
    epochs: int = typer.Option(500),
    batch_size: int = typer.Option(100),
    lr: float = typer.Option(1e-4),
    num_workers: int = typer.Option(16),
    experiment_id: str = typer.Option(f"debug-{uuid.uuid4()}"),
    # %100= 60.000 %80=48.000 %60=36.000 %40 =24.000 %20 = 12.000
    dataset_size: int = typer.Option(60000),
    dataset_type: DatasetType = typer.Option(DatasetType.full)


):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"experiment id: {experiment_id}")
    logger.info(f"batch_size: {batch_size}")

    # create log directory for classifier trained on real data
    real_tb_path = Path(root_path, "logs_classifier_real_data", experiment_id)
    real_tb_path.mkdir(parents=True, exist_ok=False)
    real_tb_writer = SummaryWriter(log_dir=real_tb_path.as_posix())

    # create log directory for classifier trained on synthetic data
    syn_tb_path = Path(root_path, "logs_classifier_syn_data", experiment_id)
    syn_tb_path.mkdir(parents=True, exist_ok=False)
    syn_tb_writer = SummaryWriter(log_dir=syn_tb_path.as_posix())

    # create log directory for classifier trained on real+synthetic data
    real_syn_tb_path = Path(
        root_path, "logs_classifier_real_syn_data", experiment_id)
    real_syn_tb_path.mkdir(parents=True, exist_ok=False)
    real_syn_tb_writer = SummaryWriter(log_dir=real_syn_tb_path.as_posix())

    # real fashionMNIST data loader
    loader_real, loader_test, _, real_dataset = get_dataloader(
        batch_size=batch_size, num_workers=num_workers, dataset_size=dataset_size)

    # synthetic fashionMNIST data loader
    synthetic_dataset = get_dataset(dataset_type=dataset_type)
    synthetic_dataset = synthetic_dataset.shuffle(buffer_size=100, seed=42)
    loader_synthetic = torch.utils.data.DataLoader(
        dataset=synthetic_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # concatenate real and synthetic datasets
    real_synthetic_dataset = torch.utils.data.ConcatDataset(
        real_dataset, synthetic_dataset)

    # real+synthetic FashionMNIST data loader
    loader_real_sythetic = torch.utils.data.DataLoader(
        dataset=real_synthetic_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    """
        -------------
        training real
        -------------
    """
    logger.info(f"Training real {dataset_type} FashionMNIST")
    train_test_classifier(loader_real, loader_test,
                          device=device, epochs=epochs, lr=lr)
    logger.info("Finished training real {dataset_type} FashionMNIST")

    """
        ------------------
        training synthetic
        ------------------
    """
    logger.info(f"Training synthetic {dataset_type} FashionMNIST")
    train_test_classifier(loader_synthetic, loader_test,
                          device=device, epochs=epochs, lr=lr)
    logger.info("Finished training synthetic {dataset_type} FashionMNIST")

    """
        -------------------------
        training real + synthetic
        -------------------------
    """
    logger.info(f"Training real + synthetic {dataset_type} FashionMNIST")
    train_test_classifier(loader_real_sythetic,
                          device=device, epochs=epochs, lr=lr)
    logger.info(
        f"Finished training real + synthetic {dataset_type} FashionMNIST")


if __name__ == "__main__":
    typer.run(main)
