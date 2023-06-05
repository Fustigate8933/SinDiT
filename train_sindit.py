import torch
import torchvision
import functools
from torchvision import transforms
from diffusion.dit import DiT
import logging
import numpy as np
from tqdm import tqdm
from diffusion.resample import UniformSampler
from diffusion.respace import create_gaussian_diffusion
import os

def random_crop_and_pad(image, crop_ratio=0.77):
    batch_size, channels, height, width = image.shape
    crop_height = int(height * crop_ratio)
    crop_width = int(width * crop_ratio)
    top = np.random.randint(0, height - crop_height)
    left = np.random.randint(0, width - crop_width)
    bottom = top + crop_height
    right = left + crop_width
    cropped_image = image[:, :, top:bottom, left:right]
    padded_image = torch.nn.functional.pad(cropped_image, (0, width - crop_width, 0, height - crop_height), mode='constant', value=0)
    return padded_image


def train_sindit(data_path="./data", checkpoint_interval=epochs, resume_checkpoint=None):
    logging.basicConfig(filename="./training_log.txt", level=logging.DEBUG, filemode="a",
                        format="[%(asctime)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())

    if torch.cuda.is_available():
        device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    # device = "cpu"  # edit line 199 in gaussian_diffusion.py

    logging.info(f"Starting program on {device}")
    seed = 42
    torch.manual_seed(seed)

    # input_size = 32

    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize(input_size),
        # transforms.CenterCrop(input_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    data = torchvision.datasets.ImageFolder(data_path, transform=img_transforms)
    dataloader = iter(torch.utils.data.DataLoader(data, num_workers=0, batch_size=1, shuffle=True))
    # class_names = os.listdir(data_path)
    logging.info(f"Data created with {len(data)} images")

    logging.info("Starting Training...")
    batch, cond = next(dataloader)
    cond = torch.tensor([1], device=device)
    batch = batch.to(device)
    cond = cond.to(device)

    input_size = 128
    batch = batch[:, :, :input_size, :input_size]
    num_heads = 6
    hidden_size = 384
    depth = 12

    model = DiT(
        input_size=input_size,
        patch_size=4,
        in_channels=3,
        hidden_size=hidden_size,
        depth=depth,  # number of DiT blocks
        num_heads=num_heads,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1,
        learn_sigma=False
    ).to(device)
    logging.info("Model created")

    diffusion = create_gaussian_diffusion()
    logging.info(f"Diffusion created with loss: {diffusion.loss_type}")

    schedule_sampler = UniformSampler(diffusion)
    logging.info("Schedule sampler created")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.9999)

    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["opt"])
        logging.info(f"Resumed training from checkpoint: {resume_checkpoint}")

    epochs = 100
    with tqdm(total=epochs) as tm:
        for iteration in range(epochs):
            t, weights = schedule_sampler.sample(batch.shape[0], device)
            running_loss = 0
            for _ in range(epochs):
                patch = random_crop_and_pad(batch, 0.77)
                model_kwargs = dict(y=cond)
                compute_losses = functools.partial(
                    diffusion.training_losses,
                    model,
                    patch,
                    t,
                    model_kwargs=model_kwargs
                )
                loss_dict = compute_losses()
                loss = loss_dict["loss"]
                opt.zero_grad()
                loss.backward()
                opt.step()
                running_loss += loss.item()

            if iteration % checkpoint_interval == 0 and iteration != 0:
                save_params = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                }
                if not os.path.exists("./checkpoints"):
                    os.makedirs("./checkpoints")
                save_path = f"./checkpoints/checkpoint_{iteration}.pt"
                torch.save(save_params, save_path)
                logging.info(f"\nCheckpoint saved at iteration {iteration}")

            tm.set_postfix(loss=running_loss / (iteration + 1))

        save_params = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
        }
        if not os.path.exists("./models"):
            os.makedirs("./models")
        save_path = f"./models/final_model.pt"
        torch.save(save_params, save_path)
        logging.info(f"Training complete! Final model saved to {save_path}.")


if __name__ == "__main__":
    # To resume training from a checkpoint, provide the path to the checkpoint file as the `resume_checkpoint` argument.
    # If starting training from scratch, set `resume_checkpoint` to None.
    resume_checkpoint = "./models/final_model.pt"
    # resume_checkpoint = None
    train_sindit(data_path="./data", resume_checkpoint=resume_checkpoint)
