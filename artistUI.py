# Version 1.0

import gradio as gr
import requests
import os
import base64
import json
import cv2
from PIL.ExifTags import TAGS

url = "http://10.2.5.35:7860"

#__________________Config____________________
model_checkpoint = "realisticVisionV20_v20NoVAE.safetensors [c0d1994c73]"
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
    """Encodes the given image data to base64."""
    _, buffer = cv2.imencode('.png', image_data)
    encoded_string = base64.b64encode(buffer)
    return encoded_string.decode('utf-8')

def txt2img(prompt_input, negative_prompt_input, image_input):
    """Generates images based on the given prompts and inputs.
    The model used is hardcoded, it can be modified in the config at the begining of this script.
    This function generated 4 images from the prompt input of the user in the UI,
    there are 2 default prompts (positive and negative) to enhance the output image in addition
    to the user prompts."""
    # Positive prompt
    inputs_pp = [prompt_input, default_pp]
    combined_pp = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_pp]))
    # Negative prompt
    inputs_np = [negative_prompt_input, default_np]
    combined_np = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_np]))

    txt2img_payload = {
        "prompt": combined_pp,
        "negative_prompt": combined_np,
        "batch_size": 4,
        "seed": -1,
        "steps": 25,
        "width": 768,
        "height": 768,
        "sampler_name": "Euler a",
        "save_images": True,
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

     # Save the images locally (modification for >1 image)
    output_folder = "image_output"
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists
    
    image_paths = []
    for i, img_data in enumerate(r.get("images", [])):
        image_path = os.path.join(output_folder, f"generated_image_{i}.jpg")
        with open(image_path, "wb") as img_file:
            img_file.write(base64.b64decode(img_data))
        image_paths.append(image_path)
        
        if i == 0:  # Print infotext for the first image only
            jsoninfo = json.loads(r['info'])
            print("\n________________Info Text________________")
            print(f"Positive prompt: {jsoninfo['infotexts'][0]}")

    return image_paths
 
# Inputs
image_input =  gr.Image(sources="upload", elem_id="input_image", label="Input Image")
prompt_input = gr.components.Textbox(lines=2, placeholder="Enter what you'd like to see here", label="Prompt")
negative_prompt_input = gr.components.Textbox(lines=2, placeholder="Enter what you don't want here", label="Negative prompt")
# Outputs
generated_image = gr.Gallery(elem_id="generated_image", label="Generated Image")

# Create the ui
ui = gr.Interface(
    fn = txt2img,
    inputs = [
        prompt_input,
        negative_prompt_input,
        image_input
    ],
    outputs = [
        generated_image,
    ],
    title ="Stable Diffusion Artist UI Pipe",
    description ="The current use of this UI is to do a simple text to image where 4 images are generated from the Stable Diffusion API, enter the prompt input and press the submit button",
    allow_flagging = "never"
)

# Display the generated image above the output components
def update_image(image_path):
    """Checks and reads the generated image file, if it exists."""
    # test if file exists
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return img_file.read()
    else:
        # Handle case when file does not exist
        return None  # Or any other appropriate action or value

# Run the ui
ui.launch()