from torchmetrics import F1Score, Accuracy, Precision, Recall
import numpy as np
from dataloader_real import get_dataloader
from loguru import logger
import typer
from pathlib import Path
import torch


def main(
    root_path: str = typer.Option('.'),
    epochs: int = typer.Option(500),
    batch_size: int = typer.Option(100),
    lr: float = typer.Option(1e-4),
    num_workers: int = typer.Option(16),
    experiment_id: str = typer.Option(f"debug-{uuid.uuid4()}"),
    # %100= 60.000 %80=48.000 %60=36.000 %40 =24.000 %20 = 12.000
    dataset_size: int = typer.Option(60000),

):


logger.add(Path(root_path, 'train_test_classification.log'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
accuracy_list = []
f1_score_list = []
precision_list = []
recall_list = []

_, loader_test, _, _ = get_dataloader(
    batch_size=batch_size, num_workers=num_workers, dataset_size=dataset_size)
# testing loop
for img, label in loader_test:
    img = img.to(device)
    label = label.to(device)

    # predict
    C_out = C(img)

    # calculate metrics
    accuracy = (C_out, label)
    f1_score = (C_out, label)
    precision = (C_out, label)
    recall = (C_out, label)

    # append batch of metrics
    accuracy_list.append(accuracy)
    f1_score_list.append(f1_score)
    precision_list.append(precision)
    recall_list.append(recall)

# calculate average metrics over all batches (single results in the container)
avg_acc = np.mean(accuracy_list)
avg_f1 = np.mean(f1_score_list)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)

# save final test metrics
logger.info(f"Average accuracy {avg_acc}")
logger.info(f"Average f1_score {avg_f1}")
logger.info(f"Average precision {avg_precision}")
logger.info(f"Average recall {avg_recall}")

if __name__ == "__main__":
    typer.run(main)
