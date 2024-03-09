import sys
import os
print(os.getcwd())
print(os.listdir())
from diffusers import StableDiffusionPipeline
import torch


device = "mps"
model_id = "runwayml/stable-diffusion-v1-5"
# model_id = "/Users/davidw/Desktop/David/NUS/_Classes/CS5260_deep_learning_2/project/mobile-diffusion/sd-pokemon-model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16", from_scratch=True, safety_checker=None)
pipe = pipe.to(device)

prompt = "a colorful bird flying through the air"
image = pipe(prompt).images[0]
    
image.save("pokemon_bird.png")