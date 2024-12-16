# Tested on 48 GB VRAM RTX A6000
# reference to https://github.com/black-forest-labs/flux?tab=readme-ov-file#diffusers-integration
import torch
from diffusers import FluxPipeline

# Choose mode
model_id = "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='balanced')
lora_dir = input("""Enter either:
                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict)""").strip()
weight_name = input("Weight file with the extension name 'safetensors':").strip() or None
print(f"Loading lora weights from {lora_dir} with weight name {weight_name}")
pipe.load_lora_weights(lora_dir,
                       weight_name=weight_name,
                       adapter_name='lora')
pipe.set_adapters('lora')
pipe.fuse_lora(adapter_names=['lora'])
# pipe.enable_model_cpu_offload()

prompt = "undefined"
seed = 42
num_inference_steps = 28
output_file = "test.png"
while True:
    prompt = input(f"Prompt(default: {prompt}):").strip() or prompt
    print(
        f"Prompt: {prompt}\nRandom seed: {seed}\nNumber of inference steps: {num_inference_steps}\nOutput file name: {output_file}")
    image = pipe(
        prompt,
        output_type="pil",
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(seed),
        height=2048,
        width=2048,        
    ).images[0]
    output_file = prompt.replace(" ", "_") + ".png"
    image.save(output_file)
    print(f"Image saved to {output_file}")
