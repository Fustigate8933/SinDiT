from diffusion.respace import create_gaussian_diffusion
from tqdm import tqdm
import torch
from diffusion.dit import DiT
from torchvision import transforms
import torchvision
from PIL import Image
from diffusion.resample import UniformSampler
import logging
import os
import cv2
import albumentations as A
# from diffusers.models import AutoencoderKL
from copy import deepcopy
# from diffusion.showimgs import showimgs, showrgb
# import matplotlib.pyplot as plt
# import numpy as np
from collections import OrderedDict


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def train(resume_checkpoint=False, checkpoint_dir="", output_interval=2000, epochs=2000000, model_save_dir="", colab=False, image_path=""):
    logging.basicConfig(filename="./training_log.txt", level=logging.DEBUG, filemode="a", format="[%(asctime)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())

    if torch.cuda.is_available():
        device = "cuda:0"
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
    else:
        device = "cpu"
    logging.info(f"Starting program on {device}")

    args = {
        "image_size": 128,
        "num_heads": 6,
        "hidden_size": 768,  # default 768
        "patch_size": 4,
        "depth": 12  # default 12
    }
    model = DiT(
        input_size=args["image_size"],
        patch_size=args["patch_size"],
        in_channels=3,
        hidden_size=args["hidden_size"],
        depth=args["depth"],  # number of DiT blocks
        num_heads=args["num_heads"],
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1,
        learn_sigma=False
    ).to(device)

    ema = deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.9999)

    if resume_checkpoint is not False:
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["opt"])

    image = cv2.imread(image_path)
    image = cv2.resize(image, (args["image_size"], args["image_size"]), interpolation=cv2.INTER_AREA)
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = image_transforms(image)[None, :, :, :].to(device)  # (c, w, h) -> (1, c, w, h)

    diffusion = create_gaussian_diffusion(
        steps=1000,
        learn_sigma=False,
        sigma_small=False,
        noise_schedule="cosine",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
    )

    schedule_sampler = UniformSampler(diffusion)

    logging.info(f"Starting training")

    with tqdm(total=epochs) as tdm:
        for epoch in range(1, epochs + 1):
            transform = A.Compose([
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(p=1, shift_limit_x=(-0.01, 0.01), shift_limit_y=0.0, rotate_limit=7),
            ])
            im = transform(image=torch.squeeze(image.cpu(), dim=0).numpy())["image"]
            augmented_image = torch.tensor(im[None, :, :, :]).to(device)
            t, _ = schedule_sampler.sample(augmented_image.shape[0], device)
            loss = diffusion.training_losses(model, augmented_image, t, model_kwargs={})["loss"].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(ema, model)

            tdm.set_postfix(MSE=loss.item(), epoch=epoch)

            if epoch % output_interval == 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": optimizer.state_dict(),
                    "args": args
                }
                checkpoint_path = os.path.join(model_save_dir, f"epoch-{epoch}.pt") if not colab else os.path.join(model_save_dir, f"model.pt")
                logging.info(f"Saving checkpoint model at {checkpoint_path}")
                torch.save(checkpoint, checkpoint_path)
        model.eval()


if __name__ == "__main__":
    train(
        resume_checkpoint=False,
        checkpoint_dir="",
        output_interval=5000,
        epochs=200000,
        model_save_dir="./models/balloon",
        colab=False,
        image_path="./data/balloon/balloon.png"
    )


"""
image augmentation
change patch size
unnormalize image at sampling
"""
