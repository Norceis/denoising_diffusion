from pathlib import Path

import torch
from torch import Tensor
from torch.optim import Optimizer
from datetime import datetime
from unet import Unet


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_index_from_list(vals: Tensor, t: Tensor, x_shape: tuple) -> Tensor:
    batch_size = t.shape[0]
    output = vals.gather(-1, t.cpu())
    return output.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def save_model(
    model: Unet, optimizer: Optimizer, epoch: int, elapsed_time: float, path: Path
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "elapsed_time": elapsed_time,
    }

    torch.save(
        checkpoint,
        path / f"e{epoch}_{elapsed_time}min_ckpt.pth",
    )


def get_run_path(time: float) -> Path:
    current_date = datetime.fromtimestamp(time).strftime("%Y-%m-%d-%H-%M-%S")
    run_path = Path(f"{current_date}")
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path
