import torch
import typer
import timm
from torchinfo import summary
from loguru import logger
from pathlib import Path


def train_test_classifier():
    # classifier
    C = timm.create_model("efficientnet_b0", pretrained=True,
                          num_classes=10, in_chans=1)
    summary(C)
    C.to(device)

    # training loop
    for e in range(epochs):
        for img, label in train_ds: 

            # predict
            # loss
            # backward
            # logging

    # testing loop
    # TODO: container for metrics per batch
    e.g. accuracy_list = list()
    for img, label in test_ds:
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
    dataset_size: int = typer.Option(48000)

):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader_real =
    loader_synthetic =


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