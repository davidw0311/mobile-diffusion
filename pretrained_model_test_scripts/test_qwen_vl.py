from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, max_length=512).eval()

# Specify hyperparameters for generation (No need to do this if you are using transformers>4.32.0)
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)


### Example 1: Generate a description for an image
image_link = "https://www.wallpaperflare.com/static/81/1023/637/the-garden-of-words-detailed-rest-summer-wallpaper.jpg"
query = tokenizer.from_list_format([
    {'image': image_link}, # Either a local path or an url
    {'text': 'Describe this image: '},
])
inputs = tokenizer(query, return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
print(response)

### Example 2: Annotate with object detection
# image_link = "https://www.wallpaperflare.com/static/81/1023/637/the-garden-of-words-detailed-rest-summer-wallpaper.jpg"
# query = tokenizer.from_list_format([
#     {'image': image_link}, # Either a local path or an url
#     {'text': 'Generate the caption in English with grounding:'},
# ])
# inputs = tokenizer(query, return_tensors='pt')
# inputs = inputs.to(model.device)
# pred = model.generate(**inputs)
# response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
# print(response)
# # <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>Generate the caption in English with grounding:<ref> Woman</ref><box>(451,379),(731,806)</box> and<ref> her dog</ref><box>(219,424),(576,896)</box> playing on the beach<|endoftext|>
# image = tokenizer.draw_bbox_on_latest_picture(response)
# if image:
#   image.save('3.jpg')
# else:
#   print("no box")