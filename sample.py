import torch
from diffusion.dit import DiT
import logging
from diffusion.respace import create_gaussian_diffusion
import torchvision
from torchvision.transforms import Normalize

def sample(model_path, num_samples=1):
    logging.basicConfig(filename="./training_log.txt", level=logging.DEBUG, filemode="a",
                        format="[%(asctime)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())

    if torch.cuda.is_available():
        device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    logging.info(f"Starting program on {device}")

    checkpoint = torch.load(model_path)
    args = checkpoint["args"]

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
    model.load_state_dict(checkpoint["ema"])
    model.eval()

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

    image_untransform = Normalize(mean=[-1, -1, -1], std=[2, 2, 2])

    logging.info("Starting sampling")
    for i in range(1, num_samples + 1):
        _sample = diffusion.p_sample_loop(
            model,
            (1, 3, args["image_size"], args["image_size"]),
            model_kwargs={},
            device=device,
            progress=True
        )
        unnormalized = image_untransform(_sample[0])
        torchvision.utils.save_image(unnormalized, f"./results/balloon/epoch-{i}.jpg")


if __name__ == "__main__":
    sample(model_path="./models/balloon/epoch-.pt", num_samples=1)
