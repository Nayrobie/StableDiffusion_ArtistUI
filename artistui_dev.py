# Version 2.10

import gradio as gr
import requests
import os
import base64
import json
import cv2
import socket
import win32com.client
import pythoncom

# genai_env\Scripts\activate
# url: EKO 2 = "http://10.2.4.15:7860" / EKO 1 = "http://10.2.5.35:7860"

#def get_ip_by_hostname(hostname):
#    try:
#        ip_address = socket.gethostbyname(hostname)
#        return ip_address
#    except socket.gaierror as e:
#        return f"Erreur lors de la récupération de l'IP pour {hostname}: {e}"

# Get ip adress
#hostname = 'EKO2'
#url = f"http://{get_ip_by_hostname(hostname)}:7860"
#print(url)

url = "http://10.2.4.15:7860"

# Model
model_checkpoint = "spybgsToolkitFor_v50NoiseOffset.safetensors [690cb24a47]"

# Directories
cwd = os.getcwd()
output_directory = os.path.join(os.getcwd(), "image_output")

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
    # Check if output dir exists
    os.makedirs(output_directory, exist_ok=True)  
    
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

def step_1_txt2img_controlnet(prompt_input_step_1, negative_prompt_input_step_1):
    """
    The 1st step of the character workflow generates 4 images.
    The parameters are set for fast generation but low quality.
    ControlNet Open Pose is used to get a character sheet reference, the reference image has 4 T-poses.
    Users only have to write a short prompt, for example: "an astronaut wearing a backpack".
    """

    # Positive prompt
    inputs_pp = [chara_sheet, prompt_input_step_1, default_pp]
    combined_pp = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_pp]))
    # Negative prompt
    inputs_np = [negative_prompt_input_step_1, default_np]
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
        "width": 1229, # initial (ref image) resolution *1.2
        "height": 571,
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
                        "pixel_perfect": True,
                        "save_detected_map": False,
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

    try:    
        txt2img_controlnet_response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=controlnet_payload)
        r = txt2img_controlnet_response.json()

        # Save the image using the save_image_to_dir function with step number 1
        image_paths = []
        for idx, image in enumerate(r.get("images", [])):
            image_path = save_image_to_dir(1, idx, image, r)
            image_paths.append(image_path)
        
        return image_paths
    
    except requests.exceptions.ConnectionError as e:
        # Print your custom message for the specific error
        print("Error: Connection to the server failed. Please run webui-user_ListenAPI.bat on EKO1")
    except Exception as e:
        # Handle other exceptions if needed
        print("An unexpected error occurred:", e)

def step_2_img2img(selected_index_step_1):
    """
    2nd step is to upscale the image chosen from the 1st step.
    """

    if selected_index_step_1 is not None:
        # Construct the file path for the selected image based on the index
        image_path = os.path.join(os.getcwd(), f"image_output\\output_image_step_1_{int(selected_index_step_1)}.jpg")
        # Check if the file exists
        if os.path.exists(image_path):
            # Read the image
            image_input = cv2.imread(image_path)
            # Encode the image to base64
            image_data = encode_image_to_base64(image_input)
        else:
            print(f"Error: Image file not found at {image_path}")
            return []
    
    img2img_payload = {
        "init_images": [image_data],
        "denoising_strength":0.45,
        "prompt": "",
        "negative_prompt": "",
        "batch_size": 2,
        "seed": -1,
        "steps": 30,
        "width": 1434, # initial (ref image) resolution *1.4
        "height": 666,
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
    
    try:
        img2img_response = requests.post(url=f'{url}/sdapi/v1/img2img', json=img2img_payload)
        r = img2img_response.json()

        # Save the image using the save_image_to_dir function with step number 2
        image_paths = []
        for idx, image in enumerate(r.get("images", [])):
            image_path = save_image_to_dir(2, idx, image, r)
            image_paths.append(image_path)

        return image_paths
    except requests.exceptions.ConnectionError as e:
        # Print your custom message for the specific error
        print("Error: Connection to the server failed. Please run webui-user_ListenAPI.bat on EKO1")
    except Exception as e:
        # Handle other exceptions if needed
        print("An unexpected error occurred:", e)

def send_to_photoshop(selected_index_step_2):
    # Check if selected_index_step_2 is not None and is an integer
    if selected_index_step_2 is not None:

        # Check if output dir exists
        os.makedirs(output_directory, exist_ok=True)  
        # Construct the file path for the selected image based on the index
        image_path = os.path.join(output_directory, f"output_image_step_2_{int(selected_index_step_2)}.jpg")
        # Check if the file exists
        if os.path.exists(image_path):

            # Need to co initialize to avoid "CoInitialize has not been called" exception
            pythoncom.CoInitialize()

            # Open Photoshop
            psApp = win32com.client.Dispatch("Photoshop.Application")

            # Open selected image from Step 2
            psApp.Open(image_path)

            # Reference the active document
            doc = psApp.Application.ActiveDocument
            # Add a new blank layer
            layer = doc.ArtLayers.Add()
            layer.name = "PaintOver"

            print(f"Image opened successfully in Photoshop: {image_path}")
        else:
            print(f"Error: Image file not found at {image_path}")
    else:
        print("Please select an image in Step 2")

def step_3_img2img(prompt_input_step_3, negative_prompt_input_step_3): # input_image_step_3
    """
    3rd step is like 2nd step but without the upscale. It's to generate the image again and render the sketch from Photoshop (from 3rd step)
    """

    # Positive prompt
    inputs_pp = [chara_sheet, prompt_input_step_3]
    combined_pp = ", ".join(filter(None, [str(i) if i != "None" else "" for i in inputs_pp]))

    # Check if an image was uploaded by the user
    if input_image_step_3 is not None:
        # Convert the uploaded image to base64
        # image_data = encode_image_to_base64(input_image_step_3) # To fix: coloration is wrong: check by importing from path instead of this
        image_path = os.path.join(os.getcwd(), f"01.jpg")
        image_data = encode_image_to_base64(cv2.imread(image_path))
    else:
        print("Error: No image uploaded in Step 3")
    
    img2img_payload = {
        "init_images": [image_data],
        "denoising_strength":0.45,
        "prompt": combined_pp,
        "negative_prompt": negative_prompt_input_step_3,
        "batch_size": 2,
        "seed": -1,
        "steps": 30,
        "width": 1434, # initial (ref image) resolution *1.4
        "height": 666,
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
    
    try:
        img2img_response = requests.post(url=f'{url}/sdapi/v1/img2img', json=img2img_payload)
        r = img2img_response.json()

        # Save the image using the save_image_to_dir function with step number 3
        image_paths = []
        for idx, image in enumerate(r.get("images", [])):
            image_path = save_image_to_dir(3, idx, image, r)
            image_paths.append(image_path)

        return image_paths
    
    except requests.exceptions.ConnectionError as e:
        # Print your custom message for the specific error
        print("Error: Connection to the server failed. Please run webui-user_ListenAPI.bat on EKO1")
    except Exception as e:
        # Handle other exceptions if needed
        print("An unexpected error occurred:", e)

def step_4_img2img(selected_index_step_3):
    """
    This is the final upscale part, with the use of the aDetailer extension to regerate the face of the character.
    """

    if selected_index_step_3 is not None:
        # Construct the file path for the selected image based on the index
        image_path = os.path.join(os.getcwd(), f"image_output\\output_image_step_1_{int(selected_index_step_3)}.jpg")
        # Check if the file exists
        if os.path.exists(image_path):
            # Read the image
            image_input = cv2.imread(image_path)
            # Encode the image to base64
            image_data = encode_image_to_base64(image_input)
        else:
            print(f"Error: Image file not found at {image_path}")
            return []

    img2img_payload = {
        "init_images": [image_data],
        "denoising_strength":0.3,
        "prompt": "",
        "negative_prompt": "",
        "batch_size": 1,
        "seed": -1,
        "steps": 30,
        "width": 1843, # initial (ref image) resolution *1.8
        "height": 857,
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
                        "ad_denoising_strength" : 0.4,
                        "ad_inpaint_height" : 512,
                        "ad_inpaint_only_masked" : True,
                        "ad_inpaint_only_masked_padding" : 32,
                        "ad_inpaint_width" : 512,
                        "ad_mask_blur" : 4,
                        "ad_model" : "face_yolov8n.pt",
                        "ad_negative_prompt" : "",
                        "ad_prompt" : "",
                        "ad_restore_face" : False,
                        "ad_sampler" : "DPM++ 2M Karras",
                        "ad_steps" : 28,
                        },
                        "3" : {
                        "ad_cfg_scale" : 7,
                        "ad_confidence" : 0.3,
                        "ad_controlnet_guidance_end" : 1,
                        "ad_controlnet_weight" : 1,
                        "ad_denoising_strength" : 0.4,
                        "ad_dilate_erode" : 4,
                        "ad_inpaint_height" : 512,
                        "ad_inpaint_only_masked" : True,
                        "ad_inpaint_only_masked_padding" : 32,
                        "ad_inpaint_width" : 512,
                        "ad_mask_blur" : 4,
                        "ad_negative_prompt" : "",
                        "ad_prompt" : "",
                        "ad_restore_face" : False,
                        "ad_sampler" : "DPM++ 2M Karras",
                        "ad_steps" : 28,
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

    # Save the image using the save_image_to_dir function with step number 3
    image_paths = []
    for idx, image in enumerate(r.get("images", [])):
        image_path = save_image_to_dir(4, idx, image, r)
        image_paths.append(image_path)

    return image_paths
        
with gr.Blocks() as ui:
    gr.Markdown("ArtistUI")

    # _________ Step 1 _________
    with gr.Tab("First Image Generation"):
        selected_index_step_1 = gr.Number(label="Index number", visible=False) # Debug
        # Input
        prompt_input_step_1 = gr.Textbox(lines=2, placeholder="Enter what you'd like to see here", label="Prompt")
        negative_prompt_input_step_1 = gr.Textbox(lines=2, placeholder="Enter what you don't want here", label="Negative prompt")
        # Output    
        generated_image_step_1 = gr.Gallery(elem_id="generated_image_step_1", label="Generated Image", show_download_button=False)
        # Button
        generate_button_step_1 = gr.Button("Generate")
        send_to_step_2_button = gr.Button("Send to next step")

    # Button to initate step 1
    generate_button_step_1.click(step_1_txt2img_controlnet,
                                 inputs=[prompt_input_step_1,negative_prompt_input_step_1],
                                 outputs=generated_image_step_1)
    
    # Update the selected variable in response to gallery selection
    generated_image_step_1.select(get_select_index, None, selected_index_step_1)
    
    # _________ Step 2 _________
    with gr.Tab("Image upscale"):
        selected_index_step_2 = gr.Number(label="Index number", visible=False) # Debug
        # Output  
        generated_image_step_2 = gr.Gallery(label="Generated Image", show_download_button=False)
        # Button
        send_to_photoshop_button = gr.Button("Send to Photoshop")

    # Update the selected variable in response to gallery selection
    generated_image_step_2.select(get_select_index, None, selected_index_step_2)
        
    # Button to initiate step 2 (in Step 1 tab)
    send_to_step_2_button.click(step_2_img2img,
                                inputs=[selected_index_step_1],
                                outputs=generated_image_step_2)

    # Button to send the selected image to Photoshop
    send_to_photoshop_button.click(send_to_photoshop,
                                   inputs=[selected_index_step_2])    
    # _________ Step 3 _________
    with gr.Tab("Photoshop Sketch"):
        selected_index_step_3 = gr.Number(label="Index number", visible=False) # Debug
        # Input
        input_image_step_3 = gr.Image(sources="upload", label="Drop your Photoshop sketch here as a JPG file")
        prompt_input_step_3 = gr.Textbox(placeholder="Enter what you'd like to see here", label="Prompt for the sketch")
        negative_prompt_input_step_3 = gr.Textbox(placeholder="Enter what you don't want here", label="Negative prompt for the sketch")
        # Output    
        generated_image_step_3 = gr.Gallery(label="Generated Image", show_download_button=False)
        # Button
        generate_button_step_3 = gr.Button("Generate")
        send_to_step_4_button = gr.Button("Send to next step")
        # Button to initate step 1
        generate_button_step_3.click(step_3_img2img,
                                    inputs=[prompt_input_step_3,negative_prompt_input_step_3],
                                    outputs=generated_image_step_3)
   
        # Update the selected variable in response to gallery selection
        generated_image_step_3.select(get_select_index, None, selected_index_step_3)
        

    # _________ Step 4 _________
    with gr.Tab("Final Upscale"):
        selected_index_step_4 = gr.Number(label="Index number", visible=False)  # Debug
        # Output
        generated_image_step_4 = gr.Gallery(label="Generated Image", show_download_button=False)
        # Button
        generate_button_step_4 = gr.Button("Generate")

        # Button to initiate step 4
        generate_button_step_4.click(step_4_img2img,
                                    inputs=[selected_index_step_3],
                                    outputs=generated_image_step_4)

        # Update the selected variable in response to gallery selection
        generated_image_step_4.select(get_select_index, None, selected_index_step_4)

# Run the ArtistUI    
if __name__ == "__main__":
    ui.launch()