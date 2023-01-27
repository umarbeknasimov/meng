import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_train_test_loaders(train_shuffle=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128, shuffle=train_shuffle)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False)

  
    return train_loader, test_loader