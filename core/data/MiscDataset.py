import os
import shutil

from torchvision import datasets, transforms
import torchvision


class CIFARDataset(object):
    @staticmethod
    def get_cifar10_transform(name):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        if name == 'AutoAugment':
            policy = transforms.AutoAugmentPolicy.CIFAR10
            augmenter = transforms.AutoAugment(policy)
        elif name == 'RandAugment':
            augmenter = transforms.RandAugment()
        elif name == 'AugMix':
            augmenter = transforms.AugMix()
        else: raise f"Unknown augmentation method: {name}!"

        transform = transforms.Compose([
            augmenter,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        return transform

    @staticmethod
    def get_cifar10_train(path, transform=None, identity_transform=False):
        if transform is None:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2470, 0.2435, 0.2616]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
        return trainset

    @staticmethod
    def get_cifar10_test(path):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
        return testset

    @staticmethod
    def get_cifar100_train(path, transform=None, identity_transform=False):
        if transform is None:
            mean=[0.507, 0.487, 0.441]
            std=[0.267, 0.256, 0.276]
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean=[0.507, 0.487, 0.441]
            std=[0.267, 0.256, 0.276]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        trainset = torchvision.datasets.CIFAR100(root=path, train=True, download=True, transform=transform)
        return trainset

    @staticmethod
    def get_cifar100_test(path):
        mean=[0.507, 0.487, 0.441]
        std=[0.267, 0.256, 0.276]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        testset = torchvision.datasets.CIFAR100(root=path, train=False, download=True, transform=transform_test)
        return testset

class SVHNDataset(object):
    @staticmethod
    def get_svhn_train(path, transform=None):
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        trainset = torchvision.datasets.SVHN(root=path, split='train', download=True, transform=transform)
        return trainset

    @staticmethod
    def get_svhn_test(path):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.SVHN(root=path, split='test', download=True, transform=transform_test)
        return testset

class CINIC10Dataset(object):
    @staticmethod
    def get_cinic10_train(path, transform=None, identity_transform=False):
        if transform is None:
            mean = [0.47889522, 0.47227842, 0.43047404]
            std = [0.24205776, 0.23828046, 0.25874835]
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        if identity_transform:
            mean = [0.47889522, 0.47227842, 0.43047404]
            std = [0.24205776, 0.23828046, 0.25874835]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        path = os.path.join(path, 'train')
        trainset = torchvision.datasets.ImageFolder(root=path, transform=transform)
        return trainset

    @staticmethod
    def get_cinic10_test(path):
        mean = [0.47889522, 0.47227842, 0.43047404]
        std = [0.24205776, 0.23828046, 0.25874835]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        path = os.path.join(path, 'test')
        testset = torchvision.datasets.ImageFolder(root=path, transform=transform_test)
        return testset

class TinyImageNetDataset(object):
    @staticmethod
    def get_tinyimagenet_train(path, transform=None, identity_transform=False):
        if transform is None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

        if identity_transform:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        path = os.path.join(path, 'train')
        trainset = torchvision.datasets.ImageFolder(root=path, transform=transform)
        return trainset

    @staticmethod
    def get_tinyimagenet_test(path):
        mean = [0.47889522, 0.47227842, 0.43047404]
        std = [0.24205776, 0.23828046, 0.25874835]
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        path = os.path.join(path, 'val')
        testset = torchvision.datasets.ImageFolder(root=path, transform=transform_test)
        return testset

class ImageNetDataset(object):
    @staticmethod
    def get_ImageNet_train(path, transform=None):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        trainset = datasets.ImageFolder(
            path,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(
                #     brightness=0.4,
                #     contrast=0.4,
                #     saturation=0.4),
                transforms.ToTensor(),
                normalize,
            ]))


        return trainset

    @staticmethod
    def get_ImageNet_test(path):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        testset = datasets.ImageFolder(
            path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
        ]))
        return testset