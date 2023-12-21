# Version 2.5

import gradio as gr
import requests
import os
import base64
import json
import cv2
from PIL.ExifTags import TAGS

# genai_env\Scripts\activate
# A1111 API
# url: EKO 1 = "http://10.2.5.35:7860/?__theme=dark" / EKO 2 = "http://10.2.4.15:7860/?__theme=dark"
url = "http://10.2.4.15:7860"
# Doc http://10.2.4.15:7860/docs#/default
# Gradio UI: http://127.0.0.1:7860/?__theme=dark

# Find the directory where the script is running
cwd = os.getcwd()

# Model
model_checkpoint = "dreamshaper_8.safetensors [879db523c3]"

# Hidden prompts
default_pp = "highly detailed, masterpiece, 8k, uhd"
default_np = "watermark, (text:1.2), naked, nsfw, deformed, bad anatomy, disfigured, mutated, fused fingers, extra fingers"
chara_sheet = "Character sheet concept art, full body length, (plain background:1.2)"

def encode_image_to_base64(image_data):
    _, buffer = cv2.imencode('.png', image_data)
    encoded_string = base64.b64encode(buffer)
    return encoded_string.decode('utf-8')

def step_1_txt2img_controlnet(prompt_input, negative_prompt_input):
    """
    The 1st step of the character workflow generates 4 images.
    The parameters are set for fast generation but low quality.
    ControlNet Open Pose is used to get a character sheet reference, the reference image has 4 T-poses.
    Users only have to write a short prompt, for example: "an astronaut wearing a backpack".
    """

    # Positive prompt
    inputs_pp = [chara_sheet, prompt_input, default_pp]
    combined_pp = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_pp]))
    # Negative prompt
    inputs_np = [negative_prompt_input, default_np]
    combined_np = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_np]))
    
    # Fetch the openpose reference image from the working directory
    image_path = os.path.join(os.getcwd(), "image_input\character_sheet_model.jpg")
    image_input = cv2.imread(image_path)
    image_data = encode_image_to_base64(image_input)
    
    controlnet_payload = {
        "prompt": combined_pp,
        "negative_prompt": combined_np,
        "batch_size": 4,
        "seed": -1,
        "steps": 20,
        "width": 1024,
        "height": 476,
        "sampler_name": "Euler a",
        "save_images": True,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "input_image": image_data,
                        "model" :"control_v11p_sd15_openpose [cab727d4]",
                        "module" : "openpose_full",
                        "weight": 1,
                        "resize_mode": "Scale to Fit (Inner Fit)",
                        "control_mode": "Balanced",
                        "pixel_perfect": True
                    },
                ]
            }
        }
    }

    # For the script to override the model chosen on A1111    
    override_settings = {
        "sd_model_checkpoint": model_checkpoint
    }
    override_payload = {
        "override_settings": override_settings
    }
    controlnet_payload.update(override_payload)
    
    txt2img_controlnet_response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=controlnet_payload)
    r = txt2img_controlnet_response.json()

    # Extract and save images locally
    output_folder = "image_output"
    image_paths = []
    for idx, image in enumerate(r.get("images", [])):
        image_path = os.path.join(output_folder, f"generated_image_{idx}.jpg")
        with open(image_path, "wb") as img_file:
            img_file.write(base64.b64decode(image))
        image_paths.append(image_path)
        
    # Extract and print infotexts
    if 'info' in r:
        jsoninfo = json.loads(r['info'])
        print("________________Info Text________________")
        print(f"Positive prompt: {jsoninfo['infotexts'][0]}")
    else:
        print("Info key not found in response:", r)
    
    return image_paths

def step_2_img2img(prompt_input, negative_prompt_input):
    """
    2nd step is to upscale the image chosen from the 1st step.
    """

    # Positive prompt
    inputs_pp = [chara_sheet, prompt_input, default_pp]
    combined_pp = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_pp]))
    # Negative prompt
    inputs_np = [negative_prompt_input, default_np]
    combined_np = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_np]))
    
    # Fetch the openpose reference image from the working directory
    image_path = os.path.join(os.getcwd(), "image_input\step_1.jpg")
    image_input = cv2.imread(image_path)
    image_data = encode_image_to_base64(image_input)
    
    img2img_payload = {
        "init_images": [image_data],
        "denoising_strength":0.45,
        "prompt": combined_pp,
        "negative_prompt": combined_np,
        "batch_size": 2,
        "seed": -1,
        "steps": 30,
        "width": 1433,
        "height": 660,
        "sampler_name": "DPM++ 2M Karras",
        "save_images": True
    }

    # For the script to override the model chosen on A1111    
    override_settings = {
        "sd_model_checkpoint": model_checkpoint
    }
    override_payload = {
        "override_settings": override_settings
    }
    img2img_payload.update(override_payload)
    
    img2img_response = requests.post(url=f'{url}/sdapi/v1/img2img', json=img2img_payload)
    r = img2img_response.json()

    # Extract and save images locally
    output_folder = "image_output"
    image_paths = []
    for idx, image in enumerate(r.get("images", [])):
        image_path = os.path.join(output_folder, f"generated_image_{idx}.jpg")
        with open(image_path, "wb") as img_file:
            img_file.write(base64.b64decode(image))
        image_paths.append(image_path)
        
    # Extract and print infotexts
    if 'info' in r:
        jsoninfo = json.loads(r['info'])
        print("________________Info Text________________")
        print(f"Positive prompt: {jsoninfo['infotexts'][0]}")
    else:
        print("Info key not found in response:", r)
    
    return image_paths

# Step 3 is 50% in photoshop and 50% inpainting the sketch area (don't know how to automate the inpainting part)
def step_3_img2img(prompt_input, negative_prompt_input):
    """
    2nd step is to upscale the image chosen from the 1st step.
    """

    # Positive prompt
    inputs_pp = [chara_sheet, prompt_input, default_pp]
    combined_pp = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_pp]))
    # Negative prompt
    inputs_np = [negative_prompt_input, default_np]
    combined_np = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_np]))
    
    # Fetch the openpose reference image from the working directory
    image_path = os.path.join(os.getcwd(), "image_input\step_1.jpg")
    image_input = cv2.imread(image_path)
    image_data = encode_image_to_base64(image_input)
    
    img2img_payload = {
        "init_images": [image_data],
        "denoising_strength":0.45,
        "prompt": combined_pp,
        "negative_prompt": combined_np,
        "batch_size": 2,
        "seed": -1,
        "steps": 30,
        "width": 1433,
        "height": 660,
        "sampler_name": "DPM++ 2M Karras",
        "save_images": True
    }

    # For the script to override the model chosen on A1111    
    override_settings = {
        "sd_model_checkpoint": model_checkpoint
    }
    override_payload = {
        "override_settings": override_settings
    }
    img2img_payload.update(override_payload)
    
    img2img_response = requests.post(url=f'{url}/sdapi/v1/img2img', json=img2img_payload)
    r = img2img_response.json()

    # Extract and save images locally
    output_folder = "image_output"
    image_paths = []
    for idx, image in enumerate(r.get("images", [])):
        image_path = os.path.join(output_folder, f"generated_image_{idx}.jpg")
        with open(image_path, "wb") as img_file:
            img_file.write(base64.b64decode(image))
        image_paths.append(image_path)
        
    # Extract and print infotexts
    if 'info' in r:
        jsoninfo = json.loads(r['info'])
        print("________________Info Text________________")
        print(f"Positive prompt: {jsoninfo['infotexts'][0]}")
    else:
        print("Info key not found in response:", r)
    
    return image_paths

def step_4_img2img(prompt_input, negative_prompt_input, adetailer_prompt_input):
    """
    This is the final upscale part, with the use of the aDetailer extension to regerate the face of the character.
    """
    # Positive prompt
    inputs_pp = [chara_sheet, prompt_input, default_pp]
    combined_pp = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_pp]))
    # Negative prompt
    inputs_np = [negative_prompt_input, default_np]
    combined_np = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_np]))
    
    # Fetch the openpose reference image from the working directory
    image_path = os.path.join(os.getcwd(), "image_input\step_3-2.jpg")
    image_input = cv2.imread(image_path)
    image_data = encode_image_to_base64(image_input)
     
    img2img_payload = {
        "init_images": [image_data],
        "denoising_strength":0.3,
        "prompt": combined_pp,
        "negative_prompt": combined_np,
        "batch_size": 1,
        "seed": -1,
        "steps": 30,
        "width": 1718,
        "height": 787,
        "sampler_name": "DPM++ 2M Karras",
        "save_images": True,
        "alwayson_scripts": {
            "ADetailer": {
                "args": [
                    {
                        "0": True,
                        "1": False,
                        "2": {
                        "ad_cfg_scale" : 7,
                        "ad_checkpoint" : "Use same checkpoint",
                        "ad_clip_skip" : 1,
                        "ad_confidence" : 0.3,
                        "ad_controlnet_guidance_end" : 1,
                        "ad_controlnet_guidance_start" : 0,
                        "ad_controlnet_model" : "None",
                        "ad_controlnet_module" : "None",
                        "ad_controlnet_weight" : 1,
                        "ad_denoising_strength" : 0.4,
                        "ad_dilate_erode" : 4,
                        "ad_inpaint_height" : 512,
                        "ad_inpaint_only_masked" : True,
                        "ad_inpaint_only_masked_padding" : 32,
                        "ad_inpaint_width" : 512,
                        "ad_mask_blur" : 4,
                        "ad_mask_k_largest" : 0,
                        "ad_mask_max_ratio" : 1,
                        "ad_mask_merge_invert" : "None",
                        "ad_mask_min_ratio" : 0,
                        "ad_model" : "face_yolov8n.pt",
                        "ad_negative_prompt" : "",
                        "ad_noise_multiplier" : 1,
                        "ad_prompt" : adetailer_prompt_input,
                        "ad_restore_face" : False,
                        "ad_sampler" : "DPM++ 2M Karras",
                        "ad_steps" : 28,
                        "ad_use_cfg_scale" : False,
                        "ad_use_checkpoint" : False,
                        "ad_use_clip_skip" : False,
                        "ad_use_inpaint_width_height" : False,
                        "ad_use_noise_multiplier" : False,
                        "ad_use_sampler" : False,
                        "ad_use_steps" : False,
                        "ad_use_vae" : False,
                        "ad_vae" : "Use same VAE",
                        "ad_x_offset" : 0,
                        "ad_y_offset" : 0,
                        "is_api" : [ ]
                        },
                        "3" : {
                        "ad_cfg_scale" : 7,
                        "ad_checkpoint" : "Use same checkpoint",
                        "ad_clip_skip" : 1,
                        "ad_confidence" : 0.3,
                        "ad_controlnet_guidance_end" : 1,
                        "ad_controlnet_guidance_start" : 0,
                        "ad_controlnet_model" : "None",
                        "ad_controlnet_module" : "None",
                        "ad_controlnet_weight" : 1,
                        "ad_denoising_strength" : 0.4,
                        "ad_dilate_erode" : 4,
                        "ad_inpaint_height" : 512,
                        "ad_inpaint_only_masked" : True,
                        "ad_inpaint_only_masked_padding" : 32,
                        "ad_inpaint_width" : 512,
                        "ad_mask_blur" : 4,
                        "ad_mask_k_largest" : 0,
                        "ad_mask_max_ratio" : 1,
                        "ad_mask_merge_invert" : "None",
                        "ad_mask_min_ratio" : 0,
                        "ad_model" : "None",
                        "ad_negative_prompt" : "",
                        "ad_noise_multiplier" : 1,
                        "ad_prompt" : "",
                        "ad_restore_face" : False,
                        "ad_sampler" : "DPM++ 2M Karras",
                        "ad_steps" : 28,
                        "ad_use_cfg_scale" : False,
                        "ad_use_checkpoint" : False,
                        "ad_use_clip_skip" : False,
                        "ad_use_inpaint_width_height" : False,
                        "ad_use_noise_multiplier" : False,
                        "ad_use_sampler" : False,
                        "ad_use_steps" : False,
                        "ad_use_vae" : False,
                        "ad_vae" : "Use same VAE",
                        "ad_x_offset" : 0,
                        "ad_y_offset" : 0,
                        "is_api" : [ ]
                        }
                     }    
                ]
            }
        }
    }

    # For the script to override the model chosen on A1111    
    override_settings = {
        "sd_model_checkpoint": model_checkpoint
    }
    override_payload = {
        "override_settings": override_settings
    }
    img2img_payload.update(override_payload)
    
    img2img_response = requests.post(url=f'{url}/sdapi/v1/img2img', json=img2img_payload)
    r = img2img_response.json()

    # Extract and save images locally
    output_folder = "image_output"
    image_paths = []
    for idx, image in enumerate(r.get("images", [])):
        image_path = os.path.join(output_folder, f"generated_image_{idx}.jpg")
        with open(image_path, "wb") as img_file:
            img_file.write(base64.b64decode(image))
        image_paths.append(image_path)
        
    # Extract and print infotexts
    if 'info' in r:
        jsoninfo = json.loads(r['info'])
        print("________________Info Text________________")
        print(f"Positive prompt: {jsoninfo['infotexts'][0]}")
    else:
        print("Info key not found in response:", r)
    
    return image_paths

# Not needed now:
def txt2img_test_dynamic_prompt():
    """
        "alwayson_scripts": {
            "Dynamic Prompts v2.17.1": {
                "args": {
                    "0": True, # Dynamic Prompts enabled
                    "1": False, # Combinatorial generation
                    "2": 1, # Combinatorial batches
                    "3": False, # Magic prompt
                    "4": False, # I'm feelinf lucky
                    "5": False, # Attention Grabber
                    "6": 1.1, # Min attention
                    "7": 1.5, # Max attention
                    "8": 100, # Max magic prompt length
                    "9": 0.7, # Magic prompt creativity
                    "10": False, # Fixed seed
                    "11": False, # Unlink seed from prompt
                    "12": True, # Don't apply to negative prompts
                    "13": False, # Enable Jinja2 templates
                    "14": False, # Don't generate images
                    "15": 0, # Max generations (0 = all combinations - the batch count value is ignored)
                    "16": "Gustavosta/MagicPrompt-Stable-Diffusion", # Magic prompt model
                    "17": "" # Magic prompt blocklist regex
                }
            }
        }
    }
    """

# Inputs
prompt_input = gr.components.Textbox(lines=2, placeholder="Enter what you'd like to see here", label="Prompt")
negative_prompt_input = gr.components.Textbox(lines=2, placeholder="Enter what you don't want here", label="Negative prompt")
adetailer_prompt_input = gr.components.Textbox(lines=2, placeholder="Enter the description of the face here", label="Prompt for the face")
#adetailer_negative_prompt_input = gr.components.Textbox(lines=2, placeholder="Enter what you don't want for the face here", label="Negative prompt for the face")
# Outputs
generated_image = gr.Gallery(elem_id="generated_image", label="Generated Image")

# Create the ui
ui = gr.Interface(
    fn = step_4_img2img,
    inputs = [
        prompt_input,
        negative_prompt_input,
        #adetailer_prompt_input,
        #adetailer_negative_prompt_input
        ],
    outputs = [
        generated_image,
    ],
    title = "Stable Diffusion Artist UI Pipe",
    allow_flagging = "never"
)

# Run the ui
ui.launch()