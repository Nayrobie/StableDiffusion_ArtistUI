import gradio as gr
import requests
import os
import base64
import json
import cv2
from PIL.ExifTags import TAGS

# A1111 API
# url: EKO 1 = "http://10.2.5.35:7860/?__theme=dark" / EKO 2 = "http://10.2.4.15:7860/?__theme=dark"
url = "http://10.2.5.35:7860"
# Doc http://10.2.4.15:7860/docs#/default
# Gradio UI: http://127.0.0.1:7860/?__theme=dark

#__________________Config____________________
model_checkpoint = "revAnimated_v122.safetensors [4199bcdd14]"
controlnet_0 = "canny"
controlnet_1 = "depth_midas"
#____________________________________________

# Maps the controlnet modules above to the right models
controlnet_mapping = {
    "canny": "control_v11p_sd15_canny [d14c016b]",
    "depth_midas": "control_v11f1p_sd15_depth [cfd03158]",
    "lineart": "control_v11p_sd15_lineart [43d4be0d]",
    "openpose_full": "control_v11p_sd15_openpose [cab727d4]",
    "scribble_pidinet": "control_v11p_sd15_scribble [d4ba51ff]",
}
# default prompts, hidden from users
default_pp = "highly detailed, masterpiece, 8k, uhd"
default_np = "watermark, text, censored, deformed, bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, worst quality, low quality, mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, Abnormal hands, abnormal legs, abnormal feet, abnormal fingers"

def encode_image_to_base64(image_data):
    _, buffer = cv2.imencode('.png', image_data)
    encoded_string = base64.b64encode(buffer)
    return encoded_string.decode('utf-8')

def txt2img(prompt_input, negative_prompt_input, image_input):
    # Positive prompt
    inputs_pp = [prompt_input, default_pp]
    combined_pp = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_pp]))
    # Negative prompt
    inputs_np = [negative_prompt_input, default_np]
    combined_np = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_np]))

    txt2img_payload = {
        "prompt": combined_pp,
        "negative_prompt": combined_np,
        "batch_size": 1,
        "seed": -1,
        "steps": 25,
        "width": 768,
        "height": 768,
        "sampler_name": "Euler a",
        "save_images": True,
        "alwayson_scripts": {
            "Dynamic Prompts v2.17.1": {
                "args": {
                    "0": True, # Dynamic Prompts enabled
                    "1": True, # Combinatorial generation
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

    # For the script to override the model chosen on A1111    
    override_settings = {
        "sd_model_checkpoint": model_checkpoint
    }
    override_payload = {
        "override_settings": override_settings
    }
    txt2img_payload.update(override_payload)
    
    txt2img_response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=txt2img_payload)
    r = txt2img_response.json()

    # Save the image locally
    image_data = r.get("images", [])
    output_folder = "image_output"
    image_path = os.path.join(output_folder, "generated_image.jpg")
    with open(image_path, "wb") as img_file:
        img_file.write(base64.b64decode(r["images"][0]))
        
    # Extract and print infotexts
    jsoninfo = json.loads(r['info'])
    print("Positive prompt:", jsoninfo["infotexts"][0]) # Debug info on the terminal
    return image_path

def img2img(prompt_input, negative_prompt_input, image_input):
    # Positive prompt
    inputs_pp = [prompt_input, default_pp]
    combined_pp = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_pp]))
    # Negative prompt
    inputs_np = [negative_prompt_input, default_np]
    combined_np = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_np]))

    image_data = encode_image_to_base64(image_input)

    img2img_payload = {
        "init_images": [image_data],
        "denoising_strength":0.5,
        "prompt": combined_pp,
        "negative_prompt": combined_np,
        "batch_size": 1,
        "seed": -1,
        "steps": 25,
        "width": 768,
        "height": 768,
        "sampler_name": "Euler a",
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

    # Save the image locally
    image_data = r.get("images", [])
    output_folder = "image_output"
    image_path = os.path.join(output_folder, "generated_image.jpg")
    with open(image_path, "wb") as img_file:
        img_file.write(base64.b64decode(r["images"][0]))
        
    # Extract and print infotexts
    jsoninfo = json.loads(r['info'])
    print("Positive prompt:", jsoninfo["infotexts"][0]) # Debug info on the terminal
    return image_path

def txt2img_controlnet(prompt_input, negative_prompt_input, image_input):
    # Positive prompt
    inputs_pp = [prompt_input, default_pp]
    combined_pp = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_pp]))
    # Negative prompt
    inputs_np = [negative_prompt_input, default_np]
    combined_np = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_np]))
    # input image
    image_data = encode_image_to_base64(image_input)
    # Controlnet mapping to the right model
    if controlnet_0 not in controlnet_mapping:
        raise ValueError(f"Unsupported controlnet module: {controlnet_0}")
    controlnet_model_0 = controlnet_mapping[controlnet_0]
    if controlnet_1 not in controlnet_mapping:
        raise ValueError(f"Unsupported controlnet module: {controlnet_1}")
    controlnet_model_1 = controlnet_mapping[controlnet_1]
    
    controlnet_payload = {
        "prompt": combined_pp,
        "negative_prompt": combined_np,
        "batch_size": 1,
        "seed": -1,
        "steps": 25,
        "width": 768,
        "height": 768,
        "sampler_name": "Euler a",
        "save_images": True,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "input_image": image_data,
                        "module": controlnet_0,
                        "model": controlnet_model_0,
                        "weight": 1,
                        "resize_mode": "Scale to Fit (Inner Fit)",
                        "control_mode": "Balanced",
                        "pixel_perfect": True
                    },
                    {
                        "input_image": image_data,
                        "module": controlnet_1,
                        "model": controlnet_model_1,
                        "weight": 1,
                        "resize_mode": "Scale to Fit (Inner Fit)",
                        "control_mode": "Balanced",
                        "pixel_perfect": True
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
    print("Positive prompt:", jsoninfo["infotexts"][0]) # Debug info on the terminal
    return image_path
    
# Inputs
image_input =  gr.Image(sources="upload", elem_id="input_image", label="Input Image")
prompt_input = gr.components.Textbox(lines=2, placeholder="Enter what you'd like to see here", label="Prompt")
negative_prompt_input = gr.components.Textbox(lines=2, placeholder="Enter what you don't want here", label="Negative prompt")
# Outputs
generated_image = gr.Image(elem_id="generated_image", label="Generated Image")

# Create the ui
iface = gr.Interface(
    fn=txt2img,
    inputs=[
        prompt_input,
        negative_prompt_input,
        image_input,
    ],
    outputs=[
        generated_image,
    ],
    title="Stable Diffusion Artist UI",
)

# Display the generated image above the output components
def update_image(image_path):
    if image_path:
        with open(image_path, "rb") as img_file:
            return img_file.read()

iface.test_command = update_image

# Run the ui
iface.launch()
