'''Contains the train/val dataloaders.'''
from pathlib import Path
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_dataloaders(data_root: str, batch_size: int, num_workers: int, data_augmentation: bool=False):
    '''Returns the train and validation dataloaders.

    Each loader yields a tensor of shape [batch_size, 3, 224, 224]. Of note is that the training 
    uses RandomResizedCrop : it randomly crop the image, then this crop is resized (to 224).

    Args:
        data_root : dataset directory root.
        batch_size : number of images inside of a batch.
        num_workders : 
        data_augmentation : whether to add data augmentation during training.
    '''

    # train/val split
    training_split = str(Path(data_root, "training"))
    validation_split = str(Path(data_root, "validation"))

    # from torchvision docs
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)

    # training 
    if data_augmentation:
        data_aug_tf = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(0.5),
        ]) # in case we add an augmentation
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            data_aug_tf,
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    train_ds = datasets.ImageFolder(training_split, transform=train_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    # validation
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    val_ds = datasets.ImageFolder(validation_split, transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader
