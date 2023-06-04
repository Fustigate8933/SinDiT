import torch
import torchvision
import functools
from torchvision import transforms
from diffusion.dit import DiT
import logging
from time import time
import random
from diffusion.resample import UniformSampler
from diffusion.respace import create_gaussian_diffusion
import os

logging.basicConfig(filename="./training_log.txt", level=logging.DEBUG, filemode="a", format="[%(asctime)s] %(message)s")
logging.getLogger().addHandler(logging.StreamHandler())

if torch.cuda.is_available():
    device = "cuda:0"
    torch.cuda.set_device(device)
else:
    device = "cpu"
device = "cpu" # edit line 199 in gaussian_diffusion.py

logging.info(f"Starting program on {device}")
seed = 42
torch.manual_seed(seed)

input_size = 64

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

num_heads = 16
hidden_size = 1152
depth = 28

model = DiT(
    input_size=input_size,
    patch_size=4,
    in_channels=3,
    hidden_size=hidden_size,
    depth=depth,  # number of DiT blocks
    num_heads=num_heads,
    mlp_ratio=4.0,
    class_dropout_prob=0.1,
    num_classes=len(data),
    learn_sigma=False
).to(device)
logging.info("Model created")

diffusion = create_gaussian_diffusion()
logging.info(f"Diffusion created with loss: {diffusion.loss_type}")

schedule_sampler = UniformSampler(diffusion)
logging.info("Schedule sampler created")

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.9999)

train_steps = 0
log_steps = 0
running_loss = 0
start_time = time()

for i in range(len(data)):
    batch, cond = next(dataloader)  # batch: because only training on 1 image a time, shape is [1, 3, input_size, input_size]
    cond = torch.tensor([1], device=device)
    logging.info(f"Training on image {i + 1} with shape {batch.shape}")
    batch = batch.to(device)
    cond = cond.to(device)
    t, weights = schedule_sampler.sample(batch.shape[0], device)
    # curr_h = round(batch.shape[2] * random.uniform(0.75, 1.25))
    # curr_w = round(batch.shape[3] * random.uniform(0.75, 1.25))
    # curr_h, curr_w = 4 * (curr_h // 4), 4 * (curr_h // 4)
    # batch = F.interpolate(batch, (curr_h, curr_h), mode="bicubic")
    # print(batch.shape)
    model_kwargs = dict(y=cond)
    compute_losses = functools.partial(
        diffusion.training_losses,
        model,
        batch,
        t,
        model_kwargs=model_kwargs
    )
    loss_dict = compute_losses()
    loss = loss_dict["loss"]
    opt.zero_grad()
    loss.backward()
    opt.step()

    save_params = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
    }
    save_path = f"./models/{class_names[i]}.pt"
    torch.save(save_params, save_path)

logging.info("Training complete!")
