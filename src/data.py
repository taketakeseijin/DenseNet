
import torch
from torchvision import datasets, transforms

MEAN = dict(
    cifar10=[0.49139968, 0.48215827, 0.44653124]
)
STD = dict(
    cifar10=[0.24703233, 0.24348505, 0.26158768]
)

DATASETS = dict(
    cifar10=datasets.CIFAR10,
    cifar100=datasets.CIFAR100,
    SVHN=datasets.SVHN,
)

def get_transforms(config, normMean=None, normStd=None):
    dataset_name = config["dataset"]
    if config["data_augument"]:
        if normMean is None:
            normMean = MEAN[dataset_name]
        if normStd is None:
            normStd = STD[dataset_name]
        normTransform = transforms.Normalize(normMean, normStd)

        trainTransform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normTransform
        ])
        testTransform = transforms.Compose([
            transforms.ToTensor(),
            normTransform
        ])
    else:
        trainTransform = transforms.Compose([
            transforms.ToTensor(),
        ])
        testTransform = transforms.Compose([
            transforms.ToTensor(),
        ])
    return trainTransform, testTransform

def get_dataset(dataset_name, train, transform):
    return DATASETS[dataset_name](root="./dataset", train=train, transform=transform, download=True)

def get_dataloader(config, train, transform):
    dataset = get_dataset(config["dataset"], train, transform)
    return torch.utils.data.DataLoader(dataset, config["batch"], shuffle=True)
