import torch
import torchvision
from torchvision import transforms
from diffusion.dit import DiT
import logging
import math
from diffusion.respace import create_gaussian_diffusion
import os


def sample_sindit():
    NUM_SAMPLES = 10000
    input_size = 32
    FULL_SIZE = (input_size, input_size)
    MIN_SIZE = 25
    MAX_SIZE = 250
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 10000
    STOP_SCALE = 16
    RESUME_CHECKPOINT = ""
    MODELS_PATH = "./models"
    RESULTS_PATH = "./results"
    SCALE_FACTOR = 0.75

    logging.basicConfig(filename="./cleandit/training_log.txt", level=logging.DEBUG, filemode="a",
                        format="[%(asctime)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())

    if torch.cuda.is_available():
        device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    # device = "cpu"  # also edit like 268 on gaussian_diffusion.py
    logging.info(f"Starting program on {device}")
    seed = 42
    torch.manual_seed(seed)

    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    data_path = "./data"
    data = torchvision.datasets.ImageFolder(data_path, transform=img_transforms)
    dataloader = iter(torch.utils.data.DataLoader(data, num_workers=0, batch_size=1, shuffle=True))
    class_names = os.listdir(data_path)
    logging.info(f"Data created with {len(data)} images")

    patch_size = 4
    in_channels = 3
    hidden_size = 1152
    depth = 28
    num_heads = 16
    mlp_ratio = 4.0
    class_dropout_prob = 0.1
    num_classes = len(data)
    learn_sigma = False

    models = []
    diffusions = []

    for _ in range(STOP_SCALE + 1)[-1:]:
        model_path = os.path.join(MODELS_PATH, "img1.pt")
        model = DiT(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,  # number of DiT blocks
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
            device=device
        )
        logging.info("Model created")
        model.load_state_dict(torch.load(model_path, map_location=device)["model"])
        model.to(device)
        model.eval()

        diffusion = create_gaussian_diffusion()
        logging.info(f"Diffusion created with loss: {diffusion.loss_type}")

        models.append(model)
        diffusions.append(diffusion)
    logging.info(f"{len(models)} models created")

    logging.info("Sampling images")
    for current_scale in range(STOP_SCALE + 1)[-1:]:
        model, diffusion = models[0], diffusions[0]
        current_factor = math.pow(SCALE_FACTOR, STOP_SCALE - current_scale)
        curr_h, curr_w = round(FULL_SIZE[0] * current_factor), round(FULL_SIZE[1] * current_factor)
        curr_h_pad, curr_w_pad = math.ceil(curr_h / 8) * 8, math.ceil(curr_w / 8) * 8
        pad_size = (0, curr_w_pad - curr_w, 0, curr_h_pad - curr_h)

        model_kwargs = {}

        if any(pad_size):
            model_kwargs["pad_size"] = pad_size

        sample = diffusion.p_sample_loop(
            model,
            (1, 3, curr_h, curr_w),
            model_kwargs=model_kwargs,
            device=device,
            progress=True
        )

        for i in range(sample.shape[0]):
            torchvision.utils.save_image(sample[i] * 0.5 + 0.5, f"{RESULTS_PATH}/1.png")
    logging.info("Sampling complete")


if __name__ == "__main__":
    sample_sindit()
