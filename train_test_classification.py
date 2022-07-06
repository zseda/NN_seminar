import imp
import torch
import typer
import timm
from torchinfo import summary
from loguru import logger
from pathlib import Path
from dataloader_real import get_dataloader
from dataloader_synthetic import get_dataset
from dataloader_synthetic import DatasetType


def train_test_classifier(loader_train, loader_test, device, epochs,):
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
    epochs: int = typer.Option(100),
    batch_size: int = typer.Option(100),
    lr: float = typer.Option(1e-4),
    z_dim: int = typer.Option(100),
    start_c_after: int = typer.Option(15),
    num_workers: int = typer.Option(16),
    experiment_id: str = typer.Option(f"debug-{uuid.uuid4()}"),
    # %80=48.000 %60=36.000 %40 =24.000 %20 = 12.000
    dataset_size: int = typer.Option(48000),
    dataset_type: DatasetType = typer.Option(DatasetType.full)


):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader_real, loader_test, _ = get_dataloader(
        batch_size=batch_size, num_workers=num_workers, dataset_size=dataset_size)

    loader_synthetic = get_dataset(dataset_type=DatasetType.full)

    """
        -------------------------
        training real + synthetic
        -------------------------
    """
    train_test_classifier(with real+synthetic)

    """
        -------------
        training real
        -------------
    """
    train_test_classifier(with real)

    """
        ------------------
        training synthetic
        ------------------
    """
    train_test_classifier(with synthetic)


if __name__ == "__main__":
    typer.run(main)
