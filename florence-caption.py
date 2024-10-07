import os
import glob
import requests
import torch
import openai
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from openai import OpenAI

client = OpenAI()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

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
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        # max_new_tokens=1024,
        # num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    return parsed_answer[task_prompt]

def refine_caption_with_openai(caption, gpt_prompt):

    refinement_prompt = (
        f"{gpt_prompt}:\n\n"
        f"{caption}"
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant helping revise caption for an image for finetuning a visual-language model."},
            {
                "role": "user",
                "content": refinement_prompt
            }
        ]
    )
    refined_caption = completion.choices[0].message.content
    return refined_caption

def run_inference_on_image(image_path, output_path, task_prompt="<MORE_DETAILED_CAPTION>"):
    image = Image.open(image_path).convert("RGB")
    
    initial_caption = generate_caption(image, task_prompt)
    print(initial_caption)

    gpt_prompt = "Refine the following caption by replacing every occurrence of the word 'watch' with 'WTCHCRT' in a cohesive manner"
    refined_caption = refine_caption_with_openai(initial_caption, gpt_prompt)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(refined_caption)
    
    print(refined_caption)
    print("###########################")

def main(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not image_files:
        print("No image files found in the specified folder.")
        return
    
    for image_path in image_files:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_folder, f"{base_name}.txt")
        run_inference_on_image(image_path, output_path)


if __name__ == "__main__":
    input_folder = "/root/WSSA0018"
    output_folder = "/root/WSSA0018"
    
    main(input_folder, output_folder)
