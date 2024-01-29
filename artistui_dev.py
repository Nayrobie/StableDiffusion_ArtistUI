# Version 2.8.5

import gradio as gr
import requests
import os
import base64
import json
import cv2
from PIL.ExifTags import TAGS
import numpy as np

# genai_env\Scripts\activate
# url: EKO 1 = "http://10.2.5.35:7860" / EKO 2 = "http://10.2.4.15:7860"
url = "http://10.2.4.15:7860"

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

def get_select_index(evt: gr.SelectData):
    # Save selected image index
    index = int(evt.index)
    return index

def save_image_to_dir(step_number, index, image, r):
    # Define the output dir
    output_directory = os.path.join(os.getcwd(), "image_output")
    os.makedirs(output_directory, exist_ok=True)  # Ensure the directory exists
    
    # Save the image to the output dir with the step number in the name
    image_path = os.path.join(output_directory, f"output_image_step_{step_number}_{index}.jpg")
    with open(image_path, "wb") as img_file:
        img_file.write(base64.b64decode(image))

    # Extract and print infotexts
    if 'info' in r:
        jsoninfo = json.loads(r['info'])
        print("________________Info Text________________")
        print(f"Positive prompt: {jsoninfo['infotexts'][0]}")
    else:
        print("Info key not found in response:", r)

    return image_path

def step_1_txt2img_controlnet(prompt_input):
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
    inputs_np = [default_np]
    combined_np = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_np]))
    
    # Fetch the openpose reference image from the working directory
    image_path = os.path.join(os.getcwd(), "image_input\character_sheet_model.jpg")
    image_input = cv2.imread(image_path)
    image_data = encode_image_to_base64(image_input)
    
    controlnet_payload = {
        "prompt": combined_pp,
        "negative_prompt": combined_np,
        "batch_size": 2,
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

    # Save the image using the save_image_to_dir function with step number 1
    image_paths = []
    for idx, image in enumerate(r.get("images", [])):
        image_path = save_image_to_dir(1, idx, image, r)
        image_paths.append(image_path)
    
    return image_paths

def step_2_img2img(selected_index, generated_images_step_1):
    """
    2nd step is to upscale the image chosen from the 1st step.
    """

    if generated_images_step_1 and selected_index is not None:
        # Construct the file path for the selected image based on the index
        image_path = os.path.join(os.getcwd(), f"image_output\\output_image_step_1_{int(selected_index)}.jpg")
        # Check if the file exists
        if os.path.exists(image_path):
            # Read the image
            image_input = cv2.imread(image_path)
            # Encode the image to base64
            image_data = encode_image_to_base64(image_input)
        else:
            print(f"Error: Image file not found at {image_path}")
            return []        
        
        # Read the image
        image_input = cv2.imread(image_path)
        # Encode the image to base64
        image_data = encode_image_to_base64(image_input)
    
    img2img_payload = {
        "init_images": [image_data],
        "denoising_strength":0.45,
        "prompt": "",
        "negative_prompt": "",
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

    # Save the image using the save_image_to_dir function with step number 2
    image_paths = []
    for idx, image in enumerate(r.get("images", [])):
        image_path = save_image_to_dir(2, idx, image, r)
        image_paths.append(image_path)

    return image_paths

with gr.Blocks() as ui:
    gr.Markdown("ArtistUI")

    # Step 1
    with gr.Tab("Step 1"):
        selected = gr.Number(label="Index number") # Debug
        # Input
        prompt_input_step_1 = gr.Textbox(lines=2, placeholder="Enter what you'd like to see here", label="Prompt")
        negative_prompt_input_step_1 = gr.Textbox(lines=2, placeholder="Enter what you don't want here", label="Negative prompt")
        # Output    
        generated_image_step_1 = gr.Gallery(elem_id="generated_image_step_1", label="Generated Image")
        # Button
        generate_button_step_1 = gr.Button("Generate")
        send_to_step_2_button = gr.Button("Send to next step")

    # Button to initate step 1
    generate_button_step_1.click(step_1_txt2img_controlnet,
                                 inputs=prompt_input_step_1,
                                 outputs=generated_image_step_1)

    
    # Update the selected variable in response to gallery selection
    generated_image_step_1.select(get_select_index, None, selected)
    
    # Step 2
    with gr.Tab("Step 2"):
        # Output  
        generated_image_step_2 = gr.Gallery(elem_id="generated_image_step_2", label="Generated Image") #, show_download_button=False)
        # Button
        generate_button_step_2 = gr.Button("Generate")
        
    # Button to initiate step 2
    send_to_step_2_button.click(step_2_img2img,
                            inputs=[selected, generated_image_step_1],
                            outputs=generated_image_step_2)
# Run the ArtistUI    
if __name__ == "__main__":
    ui.launch()