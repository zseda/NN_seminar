from dataloader_synthetic import get_dataset
from dataloader_synthetic import DatasetType
import torch
import typer


def main(

    batch_size: int = typer.Option(100),

    num_workers: int = typer.Option(16),
    dataset_type: DatasetType = typer.Option(DatasetType.full)

):

    # synthetic fashionMNIST data loader
    synthetic_dataset = get_dataset(dataset_type=dataset_type)
    # synthetic_dataset = synthetic_dataset.shuffle(buffer_size=100, seed=42)
    loader_synthetic = torch.utils.data.DataLoader(
        dataset=synthetic_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


if __name__ == "__main__":
    typer.run(main)
