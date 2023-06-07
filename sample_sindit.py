import torch
import torchvision
from torchvision import transforms
from diffusion.dit import DiT
import logging
import math
from diffusion.respace import create_gaussian_diffusion
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def sample_sindit(model_name=""):
    logging.basicConfig(filename="./training_log.txt", level=logging.DEBUG, filemode="a",
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

    input_size = 180
    STOP_SCALE = 16
    MODELS_PATH = "./models"
    RESULTS_PATH = "./results"
    patch_size = 4
    in_channels = 3
    hidden_size = 384
    depth = 12
    num_heads = 6
    mlp_ratio = 4.0
    class_dropout_prob = 0.1
    learn_sigma = False

    model_path = os.path.join(MODELS_PATH, f"{model_name}.pt")
    model = DiT(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=depth,  # number of DiT blocks
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        class_dropout_prob=class_dropout_prob,
        num_classes=1,
        learn_sigma=learn_sigma,
        device=device
    )
    logging.info("Model created")
    model.load_state_dict(torch.load(model_path, map_location=device)["model"])
    model.to(device)
    model.eval()

    diffusion = create_gaussian_diffusion()
    logging.info(f"Diffusion created")

    logging.info("Sampling images")
    # for current_scale in range(STOP_SCALE + 1)[-1:]:
    #     model, diffusion = model, diffusion
    #     # current_factor = math.pow(SCALE_FACTOR, STOP_SCALE - current_scale)
    #     # curr_h, curr_w = round(FULL_SIZE[0] * current_factor), round(FULL_SIZE[1] * current_factor)
    #     # curr_h_pad, curr_w_pad = math.ceil(curr_h / 8) * 8, math.ceil(curr_w / 8) * 8
    #     # pad_size = (0, curr_w_pad - curr_w, 0, curr_h_pad - curr_h)
    #     #
    #     # model_kwargs = {}
    #     #
    #     # if any(pad_size):
    #     #     model_kwargs["pad_size"] = pad_size
    #
    #     sample = diffusion.p_sample_loop(
    #         model,
    #         (1, 3, input_size, input_size),
    #         model_kwargs={},
    #         device=device,
    #         progress=True
    #     )
    #     for i in range(sample.shape[0]):
    #         torchvision.utils.save_image(sample[i] * 0.5 + 0.5, f"{RESULTS_PATH}/1.jpg")
    # logging.info("Sampling complete")

    with torch.no_grad():
        x = torch.randn((1, 3, 180, 180)).to(device)
        last_sample = None
        for i in tqdm(range(0, 1000)[::-1]):
            t = torch.tensor([i]).to(device)
            if last_sample is not None:
                predicted_sample = diffusion.p_sample(
                    model,
                    last_sample,
                    t,
                    clip_denoised=True,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs={}
                )["sample"]
            else:
                predicted_sample = diffusion.p_sample(
                    model,
                    x,
                    t,
                    clip_denoised=True,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs={}
                )["sample"]
            last_sample = predicted_sample
            predicted_image = transforms.ToPILImage()(last_sample.squeeze().cpu())
            plt.imshow(predicted_image)
            plt.title(i)
            plt.show()

        # predicted_image = diffusion.p_sample(
        #             model,
        #             x,
        #             torch.tensor([0]).to(device),
        #             clip_denoised=True,
        #             denoised_fn=None,
        #             cond_fn=None,
        #             model_kwargs={}
        #         )["sample"]

if __name__ == "__main__":
    sample_sindit(model_name="balloon")
