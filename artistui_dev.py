# Version 2.1

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
url = "http://10.2.5.35:7860"
# Doc http://10.2.4.15:7860/docs#/default
# Gradio UI: http://127.0.0.1:7860/?__theme=dark

# Find the directory where the script is running
cwd = os.getcwd()

# default prompts, hidden from users
default_pp = "highly detailed, masterpiece, 8k, uhd"
default_np = "watermark, text, censored, deformed, bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, worst quality, low quality, mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, Abnormal hands, abnormal legs, abnormal feet, abnormal fingers"

#__________________Config____________________
model_checkpoint = "realisticVisionV20_v20NoVAE.safetensors [c0d1994c73]"

def encode_image_to_base64(image_data):
    _, buffer = cv2.imencode('.png', image_data)
    encoded_string = base64.b64encode(buffer)
    return encoded_string.decode('utf-8')

def step_1_controlnet(prompt_input, negative_prompt_input):
    """
    """
    # Positive prompt
    inputs_pp = [prompt_input, default_pp]
    combined_pp = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_pp]))
    # Negative prompt
    inputs_np = [negative_prompt_input, default_np]
    combined_np = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_np]))
    
    # Fetch the openpose reference image from the working directory
    image_path = os.path.join(os.getcwd(), "character_sheet_model.jpg")
    image_input = cv2.imread(image_path)
    image_data = encode_image_to_base64(image_input)
    
    controlnet_payload = {
        "prompt": combined_pp,
        "negative_prompt": combined_np,
        "batch_size": 1,
        "seed": -1,
        "steps": 25,
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

    # Save the image locally
    image_data = r.get("images", [])
    output_folder = "image_output"
    image_path = os.path.join(output_folder, "generated_image.jpg")
    with open(image_path, "wb") as img_file:
        img_file.write(base64.b64decode(r["images"][0]))
        
    # Extract and print infotexts
    jsoninfo = json.loads(r['info'])
    print("\n________________Info Text________________")
    print(f"Positive prompt: {jsoninfo['infotexts'][0]}")
    
    return image_path

# Inputs
prompt_input = gr.components.Textbox(lines=2, placeholder="Enter what you'd like to see here", label="Prompt")
negative_prompt_input = gr.components.Textbox(lines=2, placeholder="Enter what you don't want here", label="Negative prompt")
# Outputs
generated_image = gr.Image(elem_id="generated_image", label="Generated Image")

# Create the ui
ui = gr.Interface(
    fn = step_1_controlnet,
    inputs = [
        prompt_input,
        negative_prompt_input
        ],
    outputs = [
        generated_image,
    ],
    title = "Stable Diffusion Artist UI Pipe",
    allow_flagging = "never"
)

# Run the ui
ui.launch()