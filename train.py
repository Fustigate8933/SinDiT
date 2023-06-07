from diffusion.respace import create_gaussian_diffusion
from tqdm import tqdm
import torch
from diffusion.dit import DiT
from torchvision import transforms
import torchvision
from PIL import Image
from diffusion.resample import UniformSampler
import logging


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def train(resume_checkpoint=False, checkpoint_dir=""):
    logging.basicConfig(filename="./training_log.txt", level=logging.DEBUG, filemode="a",
                        format="[%(asctime)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())

    if torch.cuda.is_available():
        device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    logging.info(f"Starting program on {device}")

    image_size = 128
    num_heads = 12
    hidden_size = 768
    patch_size = 4
    depth = 12
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

    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    data = torchvision.datasets.ImageFolder("./data", transform=img_transforms)
    dataloader = iter(torch.utils.data.DataLoader(data, num_workers=0, batch_size=1, shuffle=True))
    image, _ = next(dataloader)
    image = image.to(device)

    epochs = 2000000 - 67000
    output_interval = 500
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.9999)
    mse = torch.nn.MSELoss()

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

    if resume_checkpoint is not False:
        logging.info(f"Loading checkpoint from {checkpoint_dir}")
        model.load_state_dict(torch.load(checkpoint_dir))

    schedule_sampler = UniformSampler(diffusion)

    logging.info(f"Starting training")

    with tqdm(total=epochs) as tdm:
        for epoch in range(1, epochs + 1):
            t, weights = schedule_sampler.sample(image.shape[0], device)
            noise = torch.randn_like(image).to(device)
            x_t = diffusion.q_sample(image, t, noise=noise) # noised image
            predicted_noise = x_t - model(x_t, t)

            loss = mse(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tdm.set_postfix(MSE=loss.item(), epoch=epoch)

            if epoch % output_interval == 0:
                checkpoint_path = f"./models/model-epoch-{epoch}.pt"
                logging.info(f"Saving checkpoint at {checkpoint_path}")
                torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    train(True, "./models/model-epoch-67000.pt")