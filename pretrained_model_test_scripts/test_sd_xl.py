from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
generator = torch.Generator("cuda").manual_seed(88)
image = pipeline("A photorealistic image of a cozy living room, with a Shiba Inu puppy.", generator=generator).images[0]
image.save("shiba.png")