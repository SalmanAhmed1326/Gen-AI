
!pip install googletrans==3.1.0a0
!pip install --upgrade diffusers transformers -q

from googletrans import Translator
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

def get_translation(text,dest_lang):
  translator = Translator()
  translated_text = translator.translate(text, dest=dest_lang)
  return translated_text.text

import torch

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    image_gen_steps = 10
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (200, 200)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt3"
    prompt_dataset_size = 6
    prompt_max_length = 12

image_gen_model = StableDiffusionPipeline.from_pretrained( CFG.image_gen_model_id, torch_dtype=torch.float32, revision="fp16", use_auth_token='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', guidance_scale=9 )
image_gen_model = image_gen_model.to(CFG.device)

def generate_image(prompt, model):
  image = model( prompt, num_inference_steps=CFG.image_gen_steps, generator=CFG.generator, guidance_scale=CFG.image_gen_guidance_scale ).images[0]
  image = image.resize(CFG.image_gen_size)
  return image

translation = get_translation("ప్రజలు హోలీ జరుపుకుంటున్నారు","en")
generate_image(translation, image_gen_model)

translation = get_translation("La gente está celebrando Holi","en")
generate_image(translation, image_gen_model)

