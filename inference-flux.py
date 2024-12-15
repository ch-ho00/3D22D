# reference to https://github.com/black-forest-labs/flux?tab=readme-ov-file#diffusers-integration
import torch
from diffusers import FluxPipeline

# Choose mode
model_id = input(
    "Choose mode: 0(black-forest-labs/FLUX.1-dev. You are supposed to have done the third step of 'https://github.com/monk-after-90s/ai-toolkit?tab=readme-ov-file#flux1-dev') \nor\n 1(black-forest-labs/FLUX.1-schnell)").strip()
if model_id == "0":
    model_id = "black-forest-labs/FLUX.1-dev"
elif model_id == "1":
    model_id = "black-forest-labs/FLUX.1-schnell"
else:
    raise ValueError("Invalid model id")

pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
lora_dir = input("""Enter either:
                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict)""").strip()
weight_name = input("Weight file with the extension name 'safetensors':").strip() or None

pipe.load_lora_weights(lora_dir,
                       weight_name=weight_name)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "undefined"
seed = "undefined"
num_inference_steps = "undefined"
output_file = "undefined"
while True:
    prompt = input(f"Prompt(default: {prompt}):").strip() or prompt
    seed = int(input(f"Random seed(default: {seed})(int):").strip() or seed)
    num_inference_steps = int(input(
        f"Number of inference steps(default: {num_inference_steps})(int, 28 is recommend):").strip() or num_inference_steps)
    output_file = input(f"Output file name(default: {output_file}):").strip() or output_file
    print(
        f"Prompt: {prompt}\nRandom seed: {seed}\nNumber of inference steps: {num_inference_steps}\nOutput file name: {output_file}")
    image = pipe(
        prompt,
        output_type="pil",
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(seed)
    ).images[0]
    image.save(output_file)
    print(f"Image saved to {output_file}")
