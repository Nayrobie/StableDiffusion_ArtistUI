# StableDiffusion_ArtistUI

Released versions: 
* v1.0 a simplified web ui working with stable diffusion models. Users can enter a prompt and a negative prompt, press submit and get 4 images generated. The model, sampler, and other parameters can't be chosen by the users, and a default positive and a default negative prompt are added in the background to enhance the generated image and so that users don't have to write a long prompt.

Developpement versions:
* v2.0 focuses on the first step of a character workflow using ControlNet openpose model. User should uploada character sheet image and enter a prompt to generate 1 image.
* v2.1 the reference image is no longer in the web UI inputs, it's harcoded so that users only have to enter a prompt to get a character sheet.
* v2.2 now 4 character sheet images get generated. There are displayed as a gallery in the web UI for users to choose their favourite one.
* v2.3 adds the 2nd step of the character workflow to upscale the image chosen from the 1st step
* v2.4 adds the 4th step to the character workflow with the final upscale and the face upscale with the aDetailer extension
* v2.5 adds step 3 which is another img2img step like step 2 but without the upscaling, this step is to render the PhotoShop sketch (it replaces the workflow inpainting part as I can't find a way to automate the layering part)
* v2.6 a first draft that adds a tab for each step of the workflow
* v2.7 tabs and UI cleanup, photoshop upload to fix
* v2.8 changes the gradio interface to gradio blocks, and adds a send to next step button for step 2 using index numbers for selected image from the gallery

## Description
This project involves creating a simplified user interface powered by Stable Diffusion, an AI-based system that generates images. The interface allows users to input prompts, generating multiple images that fulfill the provided criteria and restrictions. It utilizes Gradio for the user interface and integrates with Stable Diffusion APIs to generate and display images based on user inputs. 

The goal is to provide an interactive platform for generating and exploring AI-generated imagery that is easy and fast to use, and does not require previous experience in generating images.

## How to use
* Prompt Inputs: Users can enter their preferred prompt and a negative prompt in the provided text boxes. The Positive prompt should describe the image to generate, the negative prompt specifies what you don't want in your image.
* Submit button: Upon submission, the system processes the prompts and generates a single image based on the provided inputs and the default prompts running in the background.
* Generated Image: The output displays the generated image.

## Installation
To install, pull and create an origin with git (enter link...)

This directory directly provides a structure in which it will be easy to to create and run the python script in a virtual environement. It contains :
* autoinstall.py that will install all the requirement needed for the pythonscript to run correctly.
it will install the requierement in the virtual envinronment folder.
* it add all the dependecies in the requirement.txt file so it can be processes.
* the Scriptlauncher ensure to activate the virtual environement, lauch the script, and the deactivate it once the script is closed.

As some specific library are not available for all python version, Shebang is more than recommended.

## Authors and acknowledgment
* Yonah Bole, main contributor
* Frederic Saclier, manager in charge

## Project status
In progress