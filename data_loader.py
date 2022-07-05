import torch
from torchvision import transforms, datasets


def get_dataloader(batch_size: int, num_workers: int, dataset_size: int):
    # MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])

    train_dataset = datasets.FashionMNIST(
        root='./fashion_mnist_data/', train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(
        root='./fashion_mnist_data/', train=False, transform=transform, download=False)
    # get partial dataset for different generators
    part_train_dataset = torch.utils.data.random_split(
        train_dataset, [dataset_size, len(train_dataset)-dataset_size], generator=torch.Generator().manual_seed(42))[0]

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(
        dataset=part_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False)
    mnist_dim = train_dataset.train_data.size(
        1) * train_dataset.train_data.size(2)
    return train_loader, test_loader, mnist_dim
