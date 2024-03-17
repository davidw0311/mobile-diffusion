import sys
from diffusers import StableDiffusionPipeline
import torch

device = "cuda:0"
model_id = "sd-pokemon-model-from-scratch"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None, revision="fp16")

pipe = pipe.to(device)

prompt = "a blue pokemon"
image = pipe(prompt).images[0]
    
image.save("pokemon2.png")
