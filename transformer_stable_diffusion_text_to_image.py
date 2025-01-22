
!pip install --upgrade diffusers transformers -q

from pathlib import Path
import tqdm
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2

import torch

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    image_gen_steps = 10
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (200, 200)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float32,
    revision="fp16", use_auth_token='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', guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)

def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image

generate_image("a boy with a dog", image_gen_model)

