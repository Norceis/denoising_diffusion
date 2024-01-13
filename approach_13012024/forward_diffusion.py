import torch
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass

from unet import Unet
from utils import get_index_from_list, get_device


@dataclass
class ForwardDiffusion:
    timesteps: int = 300
    start: float = 0.0001
    end: float = 0.02
    device = get_device()

    betas = torch.linspace(start, end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def sample(self, x_0: Tensor, t: Tensor):
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

    def get_loss(self, model: Unet, x_0: Tensor, t: Tensor):
        x_noisy, noise = self.sample(x_0, t)
        noise_pred = model(x_noisy, t)
        return F.l1_loss(noise, noise_pred)

    @torch.no_grad()
    def sample_timestep(self, model: Unet, x: Tensor, t: Tensor):
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
