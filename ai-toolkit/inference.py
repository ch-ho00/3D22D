import os
import torch
from diffusers import FluxPipeline

# Initialize the model pipeline
model_id = "black-forest-labs/FLUX.1-dev"

# Input for the directory containing LoRA weights
lora_dir = input("""Enter the path to a directory containing LoRA weights (saved with 'safetensors' extension):""").strip()
if not os.path.isdir(lora_dir):
    raise ValueError(f"The provided path '{lora_dir}' is not a valid directory.")

# Collect all .safetensors files in the given directory
weight_files = [f for f in os.listdir(lora_dir) if f.endswith(".safetensors")]
if not weight_files:
    raise ValueError(f"No '.safetensors' files found in the directory '{lora_dir}'.")

print(f"Found {len(weight_files)} weight files: {weight_files}")

# Define a set of prompts for inference
prompts = [
    "Close-up view of RMTWTH in a forest.",
    # "RMTWTH watch resting on the beach.",
]

# Configure output directory
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Inference settings
seed = 42
num_inference_steps = 2

# Run inference for each weight file with each prompt
for weight_file in weight_files:
    weight_path = os.path.join(lora_dir, weight_file)
    
    # Extract experiment prefix and step from the weight file
    if "_" in weight_file and weight_file.endswith(".safetensors"):
        experiment_prefix = weight_file.split("_")[0]
        step = weight_file.split("_")[-1].replace(".safetensors", "")
    else:
        raise ValueError(f"Invalid weight file format: {weight_file}")
    
    print(f"Loading LoRA weights from {weight_path}")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='balanced')
    pipe.load_lora_weights(lora_dir, weight_name=weight_file, adapter_name='lora')
    pipe.set_adapters('lora')
    pipe.fuse_lora(adapter_names=['lora'])

    print(pipe.transformer.proj_out.weight.device)
    # import pdb; pdb.set_trace()
    for prompt in prompts:
        print(f"Using weight: {weight_file}, Prompt: {prompt}")
        image = pipe(
            prompt,
            output_type="pil",
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cuda").manual_seed(seed),
            height=2048,
            width=2048,
        ).images[0]
        # Save the image with filename as {experimentPrefix}-{step}-{caption}.png
        sanitized_prompt = prompt.replace(" ", "_").replace("/", "_")
        output_file = os.path.join(output_dir, f"{experiment_prefix}-{step}-{sanitized_prompt}.png")
        image.save(output_file)
        print(f"Image saved to {output_file}")
    del pipe
    torch.cuda.empty_cache()

print("Inference completed for all weight files and prompts.")


