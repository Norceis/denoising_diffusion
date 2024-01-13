import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from torch import Tensor

IMG_SIZE = 32
BATCH_SIZE = 128


def load_transformed_dataset():
    """
    Returns data after applying appropriate transformations,
    to work with diffusion models.
    """
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.MNIST(
        root="data", train=True, download=True, transform=data_transform
    )

    test = torchvision.datasets.MNIST(
        root="data", train=False, download=True, transform=data_transform
    )

    return torch.utils.data.ConcatDataset([train, test])


def show_tensor_image(image: Tensor) -> None:
    """
    Plots image after applying reverse transformations.
    """

    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )

    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))


# dataloader = DataLoader(
#     load_transformed_dataset(), batch_size=BATCH_SIZE, shuffle=True, drop_last=True
# )
