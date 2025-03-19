from share import *
import config
import json
import cv2
import einops
import numpy as np
import torch
import random
from random import randint
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from PIL import Image

apply_uniformer = UniformerDetector()

model = create_model('/teamspace/studios/this_studio/ControlNet/models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict('/teamspace/studios/this_studio/ControlNet/checkpoints/model2-epoch=04.ckpt', location='cuda'))
model = model.cuda()

def process(input_image, prompt, a_prompt='best quality, extremely detailed', n_prompt='lowres, cropped, worst quality, low quality', num_samples=4, image_resolution=512, detect_resolution=512, ddim_steps=20, guess_mode=False, strength=1.0, scale=9.0, seed=-1, eta=0.0):
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape
        ## DELETE APPLY_UNIFORMER
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results

ddim_sampler = DDIMSampler(model)

def save_array_as_image(array, output_path):
    try:
        if not isinstance(array, np.ndarray):
            raise ValueError("Input must be a NumPy array.")
        if array.ndim not in (2, 3):
            raise ValueError("Array must be 2D (grayscale) or 3D (RGB).")
        
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        image = Image.fromarray(array)

        image.save(output_path)
        print(f"Image saved successfully at {output_path}")
    except Exception as e:
        raise Exception(f"Error saving image: {e}")

def load_image_with_numpy(image_path):
    img = Image.open(image_path)

    img_array = np.array(img)
    print(image_path)
    return img_array

def read_text_file(file_path, line_number):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

            if line_number < 0 or line_number > len(lines):
                raise ValueError(f"Line number {line_number} is out of range. The file has {len(lines)} lines.")
            
            line_content = lines[line_number].strip()

            content_dict = json.loads(line_content)
            print(content_dict["prompt"])
            return content_dict["prompt"]
    except :
        raise Exception("File error")

def write_string_to_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
    except:
        raise Exception()

def init_prompt(line_nr=randint(0,500)):
    img = load_image_with_numpy(f"/teamspace/studios/this_studio/ControlNet/training/fill50k/source/{line_nr}.png")
    prompt = read_text_file("/teamspace/studios/this_studio/ControlNet/training/fill50k/prompt.json",line_nr)
    write_string_to_file("/teamspace/studios/this_studio/ControlNet/new_predict/prompt.txt",prompt)
    save_array_as_image(img,"/teamspace/studios/this_studio/ControlNet/new_predict/input_image.png")
    return img,prompt

img,prompt = init_prompt()

ceva = process(img,prompt)

for i,el in enumerate(ceva):
    save_array_as_image(el,f"/teamspace/studios/this_studio/ControlNet/predict/{i}.png")
