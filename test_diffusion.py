import sys
import os
print(os.getcwd())
print(os.listdir())
from diffusers import StableDiffusionPipeline
import torch


device = "mps"
model_id = "runwayml/stable-diffusion-v1-5"
# model_id = "/Users/davidw/Desktop/David/NUS/_Classes/CS5260_deep_learning_2/project/mobile-diffusion/sd-pokemon-model"

# unet_dict = {'_class_name': 'UNet2DConditionModel', '_diffusers_version': '0.6.0',
#                            'act_fn': 'gelu',
#                            'attention_head_dim': 8,
#                            'block_out_channels': [320, 128, 64, 32],
#                            'center_input_sample': False, 'cross_attention_dim': 768,
#                            'down_block_types': ['DownBlock2D_64x64_mdCustom',
#                                                 'DownBlock2D_32x32_mdCustom',
#                                                 'DownBlock2D_16x16_mdCustom',
#                                                 'DownBlock2D_mdCustom'], 
#                            'downsample_padding': 1, 'flip_sin_to_cos': True, 'freq_shift': 0, 'in_channels': 4, #'layers_per_block': 2,
#                            'mid_block_scale_factor': 1, 'norm_eps': 1e-05, 'norm_num_groups': 32, 'out_channels': 4, 'sample_size': 64, 
#                            'up_block_types': ['UpBlock2D_16x16_mdCustom',
#                                               'UpBlock2D_32x32_mdCustom',
#                                               'UpBlock2D_64x64_mdCustom',
#                                               'UpBlock2D_mdCustom'], 
#                            'transformer_layers_per_block': [0, 1, 2, 3],  # make transformer layers increase as resolution is smaller
#                            'only_cross_attention': [True, True, False, False],
#                            'layers_per_block': [1, 1, 1, 2],                           
#                            }

# unet_dict = {'_class_name': 'UNet2DConditionModel', '_diffusers_version': '0.6.0', '_name_or_path': '/home/patrick/stable-diffusion-v1-5/unet', 
#              'act_fn': 'silu', 'attention_head_dim': 8, 'block_out_channels': [320, 640, 1280, 1280], 'center_input_sample': False, 
#              'cross_attention_dim': 768, 
#              'down_block_types': ['DownBlock2D_64x64_mdCustom',
#                                                 'DownBlock2D_32x32_mdCustom',
#                                                 'DownBlock2D_16x16_mdCustom',
#                                                 'DownBlock2D_mdCustom'], 
#              'downsample_padding': 1, 'flip_sin_to_cos': True, 'freq_shift': 0, 'in_channels': 4, 'layers_per_block': 2, 
#              'mid_block_scale_factor': 1, 'norm_eps': 1e-05, 'norm_num_groups': 32, 'out_channels': 4, 'sample_size': 64, 
#              'up_block_types': ['UpBlock2D_16x16_mdCustom',
#                                               'UpBlock2D_32x32_mdCustom',
#                                               'UpBlock2D_64x64_mdCustom',
#                                               'UpBlock2D_mdCustom'], 
#              }

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16", from_scratch=False, safety_checker=None)
pipe = pipe.to(device)

prompt = "a colorful bird flying through the air"
image = pipe(prompt, num_inference_steps = 20).images[0]
    
image.save("pokemon_bird.png")