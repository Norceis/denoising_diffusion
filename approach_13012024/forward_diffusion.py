from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import Tensor
from dataclasses import dataclass
import seaborn_image as isns
from dataset_utils import convert_tensor_to_image
from unet import Unet
from utils import get_index_from_list, get_device


@dataclass
class ForwardDiffusion:
    timesteps: int = 300
    start: float = 0.0001
    end: float = 0.02
    img_size: int = 32
    device = get_device()

    betas = torch.linspace(start, end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def sample(self, x_0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = get_index_from_list(
            self.sqrt_alphas_cumprod, t, x_0.shape
        )
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        return sqrt_alphas_cumprod_t.to(self.device) * x_0.to(
            self.device
        ) + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(
            self.device
        ), noise.to(
            self.device
        )

    def get_loss(self, model: Unet, x_0: Tensor, t: Tensor) -> Tensor:
        x_noisy, noise = self.sample(x_0, t)
        noise_pred = model(x_noisy, t)
        return F.l1_loss(noise, noise_pred)

    @torch.no_grad()
    def sample_timestep(self, model: Unet, x: Tensor, t: Tensor) -> Tensor:
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        """
        betas_t = get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def plot_image_generation(
        self,
        model: Unet,
        save_path: Path,
        num_images: int = 10,
        epoch_number: int = 0,
    ) -> None :
        # Sample noise
        img = torch.randn((1, 1, self.img_size, self.img_size), device=self.device)

        images = []
        stepsize = int(self.timesteps / num_images)

        for i in range(0, self.timesteps)[::-1]:
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            img = self.sample_timestep(model, img, t)
            if i % stepsize == 0:
                output_img = convert_tensor_to_image(img.detach().cpu())
                images.append(np.array(output_img, dtype=np.int32))

        isns.ImageGrid(
            images, cmap="binary", col_wrap=num_images, showticks=False, cbar=False
        )
        plt.savefig(save_path / f"epoch_{epoch_number}_pic.png")
        # plt.show()
