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
import torch.optim as optim
import torch.nn as nn
from torchmetrics import F1Score, Accuracy, Precision, Recall
import numpy as np
from tqdm import tqdm


def test_classifier(C, loader_test, tb_writer, device, current_epoch):
    accuracy = Accuracy().to(device)
    f1score = F1Score().to(device)
    precision = Precision().to(device)
    recall = Recall().to(device)

    accuracy_top3 = Accuracy(top_k=3).to(device)
    f1score_top3 = F1Score(top_k=3).to(device)
    precision_top3 = Precision(top_k=3).to(device)
    recall_top3 = Recall(top_k=3).to(device)

    # testing loop
    for img, label in loader_test:
        img = img.to(device)
        label = label.to(device)

        # predict
        C_out = C(img)
        # C_out = torch.argmax(C_out, dim=1)

        # update metrics
        accuracy.update(preds=C_out, target=label)
        f1score.update(preds=C_out, target=label)
        precision.update(preds=C_out, target=label)
        recall.update(preds=C_out, target=label)

        # update top3 metrics
        accuracy_top3.update(preds=C_out, target=label)
        f1score_top3.update(preds=C_out, target=label)
        precision_top3.update(preds=C_out, target=label)
        recall_top3.update(preds=C_out, target=label)

    # calculate average metrics over all batches (single results in the container)
    avg_acc = accuracy.compute()
    avg_f1 = f1score.compute()
    avg_precision = precision.compute()
    avg_recall = recall.compute()

    avg_acc_top3 = accuracy_top3.compute()
    avg_f1_top3 = f1score_top3.compute()
    avg_precision_top3 = precision_top3.compute()
    avg_recall_top3 = recall_top3.compute()

    # write to tensorboard 
    # accuracy
    tb_writer.add_scalar(
        'test/accuracy/top1',
        avg_acc,
        global_step=current_epoch
    )
    tb_writer.add_scalar(
        'test/accuracy/top3',
        avg_acc_top3,
        global_step=current_epoch
    )

    # f1
    tb_writer.add_scalar(
        'test/f1/top1',
        avg_f1,
        global_step=current_epoch
    )
    tb_writer.add_scalar(
        'test/f1/top3',
        avg_f1_top3,
        global_step=current_epoch
    )

    # precision
    tb_writer.add_scalar(
        'test/precision/top1',
        avg_precision,
        global_step=current_epoch
    )
    tb_writer.add_scalar(
        'test/precision/top3',
        avg_precision_top3,
        global_step=current_epoch
    )

    # recall
    tb_writer.add_scalar(
        'test/recall/top1',
        avg_recall,
        global_step=current_epoch
    )
    tb_writer.add_scalar(
        'test/recall/top3',
        avg_recall_top3,
        global_step=current_epoch
    )

    # flush tensorboard
    tb_writer.flush()


def train_test_classifier(
        loader_train,
        device,
        epochs,
        lr,
        tb_writer,
        log_dir,
        experiment_id,
        loader_test
    ):
    # classifier
    C = timm.create_model("efficientnet_b0", pretrained=True,
                          num_classes=10, in_chans=1)
    # summary(C)
    C.to(device)
    C_optimizer = optim.Adam(C.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion_classification = nn.CrossEntropyLoss()
    global_step = 0
    C_losses = []

    # training loop
    for e in tqdm(range(epochs)):
        for img_train, label_train in loader_train:
            # logger.info(img_train.shape)
            # logger.info(img_train.dtype)
            # logger.info(label_train.shape)
            # logger.info(label_train.dtype)
            # break
            img_train = img_train.to(device)
            label_train = label_train.to(device)
            # for logging
            global_step += 1
            # predict
            C_out_train = C(img_train)
            # loss
            C_loss = criterion_classification(C_out_train, label_train)
            # backward
            C_loss.backward()
            # optimize
            C_optimizer.step()
            # logging
            if global_step % 50 == 0:
                tb_writer.add_scalar(
                    'train/C_loss', C_loss.item(), global_step=global_step)
            tb_writer.flush()
            C_losses.append(C_loss.data.item())
        
        # end of epoch
        test_classifier(
            C=C,
            loader_test=loader_test,
            tb_writer=tb_writer,
            device=device,
            current_epoch=e
        )

    # save C
    torch.save(C.state_dict(), Path(log_dir, "classifier.pth"))


def main(
    root_path: str = typer.Option('.'),
    epochs: int = typer.Option(500),
    batch_size: int = typer.Option(100),
    lr: float = typer.Option(1e-4),
    num_workers: int = typer.Option(16),
    experiment_id: str = typer.Option(f"debug-{uuid.uuid4()}"),
    # %100= 60.000 %80=48.000 %60=36.000 %40 =24.000 %20 = 12.000
    dataset_size: int = typer.Option(60000),
    dataset_type: DatasetType = typer.Option(DatasetType.full),
    ankle_boots: bool = typer.Option(True)
):
    logger.add(Path(root_path, 'train_test_classification.log'))
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
    _, loader_test, _, real_dataset = get_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        dataset_size=dataset_size,
        with_target_transform=True,
        without_ankle_boots=not(ankle_boots)
    )
    loader_real = torch.utils.data.DataLoader(
        dataset=real_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # synthetic fashionMNIST data loader
    synthetic_dataset = get_dataset(dataset_type=dataset_type, with_ankle_boot=ankle_boots)
    # synthetic_dataset = synthetic_dataset.shuffle(buffer_size=100, seed=42)
    loader_synthetic = torch.utils.data.DataLoader(
        dataset=synthetic_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # concatenate real and synthetic datasets
    real_synthetic_dataset = torch.utils.data.ConcatDataset(
        [real_dataset, synthetic_dataset])

    # real+synthetic FashionMNIST data loader
    loader_real_sythetic = torch.utils.data.DataLoader(
        dataset=real_synthetic_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    """
        -------------
        training real
        -------------
    """
    logger.info(f"Training real {dataset_type} FashionMNIST")
    train_test_classifier(
        loader_real,
        device=device,
        epochs=epochs,
        lr=lr,
        tb_writer=real_tb_writer,
        log_dir=real_tb_path,
        experiment_id=experiment_id,
        loader_test=loader_test
    )
    logger.info(f"Finished training real {dataset_type} FashionMNIST")

    """
        ------------------
        training synthetic
        ------------------
    """
    logger.info(f"Training synthetic {dataset_type} FashionMNIST")
    train_test_classifier(
        loader_synthetic,
        device=device,
        epochs=epochs,
        lr=lr,
        tb_writer=syn_tb_writer,
        log_dir=syn_tb_path,
        experiment_id=experiment_id,
        loader_test=loader_test
    )
    logger.info(f"Finished training synthetic {dataset_type} FashionMNIST")

    """
        -------------------------
        training real + synthetic
        -------------------------
    """
    logger.info(f"Training real + synthetic {dataset_type} FashionMNIST")
    train_test_classifier(
        loader_real_sythetic,
        device=device,
        epochs=epochs,
        lr=lr,
        tb_writer=real_syn_tb_writer,
        log_dir=real_syn_tb_path,
        experiment_id=experiment_id,
        loader_test=loader_test
    )
    logger.info(f"Finished training real + synthetic {dataset_type} FashionMNIST")


if __name__ == "__main__":
    typer.run(main)
