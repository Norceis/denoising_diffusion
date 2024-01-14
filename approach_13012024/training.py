import time
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm

from forward_diffusion import ForwardDiffusion
from dataset_utils import load_transformed_dataset
from unet import Unet
from utils import save_model, get_run_path

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 2e-4
SAVE_INTERVAL = 30  # minutes

if __name__ == "__main__":
    model = Unet()
    dataset = load_transformed_dataset()

    # dataloader = DataLoader(
    #     Subset(dataset, range(int(len(dataset) * 0.1))),
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     drop_last=True,
    # )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    fd = ForwardDiffusion()

    model.to(fd.device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    beginning_time = time.time()
    current_time = time.time()
    run_path = get_run_path(beginning_time)

    for epoch in range(EPOCHS):
        epoch_progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            t = torch.randint(0, fd.timesteps, (BATCH_SIZE,), device=fd.device).long()
            loss = fd.get_loss(model, batch[0], t)
            loss.backward()
            optimizer.step()

            epoch_progress_bar.set_postfix(loss=loss.item(), refresh=True)
            epoch_progress_bar.update(1)

            elapsed_time = time.time() - current_time
            if elapsed_time >= SAVE_INTERVAL * 60:
                total_time_lapsed = int((time.time() - beginning_time) / 60)
                save_model(model, optimizer, epoch, total_time_lapsed, run_path)
                current_time = time.time()
                print(
                    f"Model saved at epoch {epoch}, step {step}, elapsed time {total_time_lapsed} min"
                )
        epoch_progress_bar.close()
        fd.plot_image_generation(model, run_path, num_images=10, epoch_number=epoch)

    total_time_lapsed = int((time.time() - beginning_time) / 60)
    save_model(model, optimizer, epoch, total_time_lapsed, run_path)
    print(
        f"Model saved at epoch {epoch}, step {step}, elapsed time {total_time_lapsed} min"
    )
