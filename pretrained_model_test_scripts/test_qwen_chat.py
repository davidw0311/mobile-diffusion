from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

### Example 1: Generate a description for an image
image_link = "https://www.wallpaperflare.com/static/81/1023/637/the-garden-of-words-detailed-rest-summer-wallpaper.jpg"
response, history = model.chat(tokenizer, query=f'<img>{image_link}</img>这是什么', history=None)
print(response)