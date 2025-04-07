from share import *
import config
import json
import cv2
import einops
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
#from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from PIL import Image
import json
import sys
import os

assert len(sys.argv) == 5, 'Args are wrong. There should be 4 args: channels, prompt_path, model_path, nr_of_samples.'

channels = sys.argv[1]
assert channels in ['1', '3', '4'], 'Input channels must be 1, 3 or 4.'

prompt_path = sys.argv[2]
assert os.path.exists(prompt_path), f'Prompt path {prompt_path} does not exist.'

model_path = sys.argv[3]
assert os.path.exists(model_path), f'Model path {model_path} does not exist.'

nr_of_samples = int(sys.argv[4])
assert nr_of_samples > 0, f'Number of samples {nr_of_samples} must be greater than 0.'

#apply_uniformer = UniformerDetector()
model = create_model(f'./ControlNet/models/cldm_v21_{channels}.yaml').cpu()
model.load_state_dict(load_state_dict(model_path, location='cuda'))
model = model.cuda()

def process(input_image, prompt, a_prompt='best quality, extremely detailed', n_prompt='lowres, cropped, worst quality, low quality', num_samples=5, image_resolution=512, detect_resolution=512, ddim_steps=50, guess_mode=False, strength=1.2, scale=7.0, seed=-1, eta=0.0):
    with torch.no_grad():
        
        if channels == '3':
            input_image = HWC3(input_image)
            
        #M detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
        img = resize_image(input_image, image_resolution)
        if channels == '1':
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)  # restore shape [H, W, 1]
        
        H, W, C = img.shape
        #M detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        detected_map = img
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
            raise ValueError("Array must be 2D (grayscale) or 3D (RGB/RGBA).")

        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        if array.ndim == 2:
            mode = 'L'  # Grayscale
            array_to_save = array
        elif array.ndim == 3:
            if array.shape[2] == 1:
                mode = 'L'
                array_to_save = np.squeeze(array, axis=2)  # Don't modify original
            elif array.shape[2] == 3:
                mode = 'RGB'
                array_to_save = array
            elif array.shape[2] == 4:
                mode = 'RGBA'
                array_to_save = array
            else:
                raise ValueError("3D array must have 1 (grayscale), 3 (RGB), or 4 (RGBA) channels.")

        image = Image.fromarray(array_to_save, mode)
        image.save(output_path)
        print(f"Image saved successfully at {output_path}")
    except Exception as e:
        raise Exception(f"Error saving image: {e}")


def load_image_with_numpy(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    return img_array

def load_grayscale_image_with_numpy(image_path):
    img = Image.open(image_path).convert("L")  # Convert to grayscale (L mode)
    img_array = np.expand_dims(np.array(img), axis=-1)  # Expand the dimensions to include the channel (depth of 1)
    return img_array

def read_sample_paths(file_path, n=100):
    filenames = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if i >= n:
                break
            try:
                data = json.loads(line.strip())
                source_path = data.get("source", "")  # Get the "source" field
                filename = os.path.basename(source_path)  # Extract only the filename
                filenames.append(filename)
            except json.JSONDecodeError as e:
                print(f"Error decoding line {i+1}: {e}")
    return filenames

samples = read_sample_paths(prompt_path, nr_of_samples)

if channels == '1':
    samples_folder = 'generated_depths'
elif channels == '3':
    samples_folder = 'masks_out'
elif channels == '4':
    samples_folder = 'rgba_masks'

for i,sample in enumerate(samples):
    if channels == '1':
        img = load_grayscale_image_with_numpy(f"./testing/{samples_folder}/{sample}")
    else:
        img = load_image_with_numpy(f"./testing/{samples_folder}/{sample}")
    save_array_as_image(img,f"./output/input_{i}.png")
    ceva = process(img,"")
    for j,el in enumerate(ceva):
        save_array_as_image(el,f"./output/predicted_{i}_{j}.png")