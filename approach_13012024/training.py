import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from forward_diffusion import ForwardDiffusion
from dataset_utils import load_transformed_dataset
from unet import Unet

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 3e-4

if __name__ == "__main__":
    model = Unet()
    dataloader = DataLoader(
        load_transformed_dataset(), batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    fd = ForwardDiffusion()

    model.to(fd.device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            t = torch.randint(0, fd.timesteps, (BATCH_SIZE,), device=fd.device).long()
            loss = fd.get_loss(model, batch[0], t)
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        # sample_plot_image()
