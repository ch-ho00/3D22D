import os
import glob
import requests
import torch
import openai
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from openai import OpenAI
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
client = OpenAI(api_key="")

# Set device and torch dtype
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

def generate_caption(image, task_prompt="<CAPTION>"):
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
    Refine the given caption using OpenAI's GPT model.
    """
    try:
        refinement_prompt = (
            f"{gpt_prompt}:\n\n"
            f"{caption}"
        )

        completion = client.chat.completions.create(
            model="gpt-4",  # Ensure the model name is correct. It was "gpt-4o" previously, which might be a typo.
            messages=[
                {"role": "system", "content": "You are an assistant helping revise caption for an image for finetuning a visual-language model."},
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

def run_inference_on_image(image_path, output_folder, base_name, task_prompt="<CAPTION>", gpt_prompt=""):
    """
    Generate and refine captions for a single image, with caching.
    """
    # Define paths for initial and refined captions
    initial_caption_path = os.path.join(output_folder, f"{base_name}_initial.txt")
    refined_caption_path = os.path.join(output_folder, f"{base_name}_refined.txt")

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

def main(input_folder, output_base_folder, prompt_type, gpt_prompt):
    """
    Main function to process all images in the input folder with the specified prompt type.
    """
    # Create a separate subfolder for each prompt type
    prompt_folder_name = prompt_type.strip('<>').replace('>', '').replace('<', '')
    output_folder = os.path.join(output_base_folder, prompt_folder_name)
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
        run_inference_on_image(image_path, output_folder, base_name, task_prompt=prompt_type, gpt_prompt=gpt_prompt)

if __name__ == "__main__":
    input_folder = "/root/Richmont_caption/Richmont_Loras_Renders"
    output_base_folder = "/root/captioned" 
    
    # Define prompt types and their corresponding GPT prompts
    prompt_configs = {
        "<CAPTION>": (
            "Maintaining all original information and ensuring caption reads concisely and naturally and coherently, revise the caption "
            "Also Replace every occurrence of the word 'watch' or watch model name like Jaeger-LeCoultre Master Ultra Thin and watch model name with 'RYMDWTH'."
            "Do not start sentence with a 'the image'"
        ),
        "<DETAILED_CAPTION>": (
            "Maintaining all original information and ensuring caption reads concisely and naturally and coherently, revise the caption "
            "Also Replace every occurrence of the word 'watch' or watch model name like Jaeger-LeCoultre Master Ultra Thin and watch model name with 'RYMDWTH'."
            "Do not start sentence with a 'the image'"
        ),
        "<MORE_DETAILED_CAPTION>": (
            "Maintaining all original information and ensuring caption reads concisely and naturally and coherently, revise the caption "
            "Also Replace every occurrence of the word 'watch' or watch model name like Jaeger-LeCoultre Master Ultra Thin and watch model name with 'RYMDWTH'."
            "Do not start sentence with a 'the image'."
        )
    }
    
    # List of prompt types to process
    prompt_types = ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]
    # prompt_types = ["<MORE_DETAILED_CAPTION>"]
    
    for prompt_type in prompt_types:
        gpt_prompt = prompt_configs.get(prompt_type, "Refine the following caption to make it more natural and human-like.")
        main(input_folder, output_base_folder, prompt_type, gpt_prompt)
