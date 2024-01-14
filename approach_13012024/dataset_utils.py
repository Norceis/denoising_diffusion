from typing import Union

from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from torch import Tensor
from torchvision.datasets import VisionDataset

IMG_SIZE = 32
BATCH_SIZE = 128


def load_transformed_dataset() -> Union[VisionDataset, ConcatDataset]:
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


def convert_tensor_to_image(image: Tensor) -> Image:
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

    return reverse_transforms(image)


# dataloader = DataLoader(
#     load_transformed_dataset(), batch_size=BATCH_SIZE, shuffle=True, drop_last=True
# )
