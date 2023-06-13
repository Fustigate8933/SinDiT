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
    model.load_state_dict(torch.load(model_path, map_location=device))
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
            (1, 3, image_size, image_size),
            model_kwargs={},
            device=device,
            progress=True
        )
        unnormalized = image_untransform(_sample[0])
        torchvision.utils.save_image(unnormalized, f"./results/wave-model-numhead6-epoch-100000-{i}.jpg")


if __name__ == "__main__":
    sample(model_path="./models/model1.pt", num_samples=1)
