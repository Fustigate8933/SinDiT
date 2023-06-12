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
import matplotlib.pyplot as plt
import numpy as np


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

    image_size = 128
    num_heads = 6  # default 12
    hidden_size = 768  # default 768
    patch_size = 4
    depth = 12  # default 12
    model = DiT(
        input_size=image_size,
        patch_size=patch_size,
        in_channels=3,
        hidden_size=hidden_size,
        depth=depth,  # number of DiT blocks
        num_heads=num_heads,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1,
        learn_sigma=False
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.9999)
    mse = torch.nn.MSELoss()

    if resume_checkpoint is not False:
        logging.info(f"Loading model and optimizer state from {checkpoint_dir}")
        model.load_state_dict(torch.load(checkpoint_dir, map_location=device))
        optimizer.load_state_dict(torch.load(checkpoint_dir.replace(".pt", "_optimizer.pt"), map_location=device))
        optimizer.zero_grad()

    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize(image_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = image_transforms(image)[None, :, :, :].to(device)  # (c, w, h) -> (1, c, w, h)
    # plt.imshow(cv2.cvtColor(np.moveaxis(torch.squeeze(image.cpu(), dim=0).numpy(), 0, -1), cv2.COLOR_BGR2RGB))
    # plt.show()
    # return

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
                A.VerticalFlip(p=0.2),
                A.GaussianBlur(p=0.5, sigma_limit=(0, 3)),
                A.RandomBrightnessContrast(p=0.4),
                A.ShiftScaleRotate(p=1, shift_limit=0.1, rotate_limit=20, scale_limit=0.3),
            ])
            augmented_image = torch.tensor(transform(image=torch.squeeze(image.cpu(), dim=0).numpy())["image"][None, :, :, :]).to(device)

            t, _ = schedule_sampler.sample(augmented_image.shape[0], device)
            noise = torch.randn_like(augmented_image).to(device)
            x_t = diffusion.q_sample(augmented_image, t, noise=noise)  # noised image

            predicted_noise = x_t - model(x_t, t)

            loss = mse(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            tdm.set_postfix(MSE=loss.item(), epoch=epoch)

            if epoch % output_interval == 0:
                checkpoint_path = os.path.join(model_save_dir, f"wave-model-numhead6-epoch-{epoch}.pt") if not colab else os.path.join(model_save_dir, f"model.pt")
                logging.info(f"Saving checkpoint model at {checkpoint_path}")
                torch.save(model.state_dict(), checkpoint_path)
                torch.save(optimizer.state_dict(), checkpoint_path.replace(".pt", "_optimizer.pt"))



if __name__ == "__main__":
    train(
        resume_checkpoint=False,
        checkpoint_dir="",
        output_interval=5000,
        epochs=200000,
        model_save_dir="./models",
        colab=False,
        image_path="./data/wave/wave.jpg"
    )


"""
image augmentation
change patch size
unnormalize image at sampling
"""
