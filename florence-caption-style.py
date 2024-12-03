import os
import glob
import requests
import torch
import openai
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# openai.api_key = 
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the Florence-2 model and processor
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    torch_dtype=torch_dtype,
    trust_remote_code=True
).to(device)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True
)

def generate_caption(image, task_prompt="<MORE_DETAILED_CAPTION>"):
    """
    Generate a caption for the given image using the Florence-2 model.
    """
    try:
        inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        return parsed_answer.get(task_prompt, generated_text)  # Safeguard in case key is missing
    except Exception as e:
        logging.error(f"Error generating caption: {e}")
        return ""

def refine_caption_with_openai(caption, gpt_prompt):
    """
    Refine the given caption to focus on the watch and replace background/style descriptions with 'STRSTY'.
    """
    try:
        refinement_prompt = (
            f"{gpt_prompt}\n\n"
            f"Caption:\n{caption}"
        )

        completion = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant helping revise captions for images for fine-tuning a text-to-image model."
                },
                {
                    "role": "user",
                    "content": refinement_prompt
                }
            ]
        )
        refined_caption = completion.choices[0].message.content.strip()
        return refined_caption
    except Exception as e:
        logging.error(f"Error refining caption: {e}")
        return ""

def run_inference_on_image(image_path, output_folder, base_name, task_prompt="<MORE_DETAILED_CAPTION>", gpt_prompt=""):
    """
    Generate and refine captions for a single image, with caching.
    """
    # Define paths for initial and refined captions
    initial_caption_path = os.path.join(output_folder, f"{base_name}_initial.txt")
    refined_caption_path = os.path.join(output_folder, f"{base_name}.txt")

    # Check if initial caption exists
    if os.path.exists(initial_caption_path):
        with open(initial_caption_path, 'r', encoding='utf-8') as f:
            initial_caption = f.read()
        logging.info(f"Loaded existing initial caption for {base_name}.")
    else:
        try:
            # Generate initial caption and save it
            image = Image.open(image_path).convert("RGB")
            initial_caption = generate_caption(image, task_prompt)
            with open(initial_caption_path, 'w', encoding='utf-8') as f:
                f.write(initial_caption)
            logging.info(f"Generated and saved initial caption for {base_name}.")
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            return

    try:
        refined_caption = refine_caption_with_openai(initial_caption, gpt_prompt)
        with open(refined_caption_path, 'w', encoding='utf-8') as f:
            f.write(refined_caption)
        logging.info(f"Saved caption for {base_name}.")
    except Exception as e:
        logging.error(f"Error refining caption for {base_name}: {e}")
        return

    # Print the refined caption
    print(refined_caption)
    print("###########################")

def main(input_folder, output_base_folder):
    """
    Main function to process all images in the input folder.
    """
    # Define the GPT prompt
    gpt_prompt = (
        "Please rewrite the following caption to focus on describing the watch, "
        "and remove any descriptions of the background. "
        "Ensure the caption reads naturally and coherently."
    )

    # Set the task prompt for Florence-2
    task_prompt = "<MORE_DETAILED_CAPTION>"

    # Create output folder
    output_folder = os.path.join(output_base_folder, "captions_with_STRSTY")
    os.makedirs(output_folder, exist_ok=True)

    # Define supported image extensions
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))

    if not image_files:
        logging.warning("No image files found in the specified folder.")
        return

    # Process each image
    for image_path in image_files:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        run_inference_on_image(image_path, output_folder, base_name, task_prompt=task_prompt, gpt_prompt=gpt_prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process captions to focus on the watch and use 'STRSTY' for style.")

    # Positional arguments
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing images.")
    parser.add_argument("output_base_folder", type=str, help="Path to the folder for saving output captions.")

    # Parse arguments
    args = parser.parse_args()

    input_folder = args.input_folder
    output_base_folder = args.output_base_folder

    main(input_folder, output_base_folder)
