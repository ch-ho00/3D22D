# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import torch
from diffusers import FluxPipeline
from PIL import Image


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        model_id = "black-forest-labs/FLUX.1-dev"
        self.pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='balanced')
        
        lora_dir = "/root/RMTWTH_64rank_2res_4ksteps_lr4e-4_batch3"
        weight_name = "RMTWTH_64rank_2res_4ksteps_lr4e-4_batch3_000002500.safetensors"
        self.pipe.load_lora_weights(lora_dir, weight_name=weight_name, adapter_name='lora')
        self.pipe.set_adapters('lora')
        self.pipe.fuse_lora(adapter_names=['lora'])

    def predict(
        self,
        prompt: str = Input(description="Text prompt for image generation"),
        seed: int = Input(description="Random seed for reproducibility", default=42),
        num_inference_steps: int = Input(description="Number of inference steps", default=28),
        height: int = Input(description="Height of the generated image", default=2048),
        width: int = Input(description="Width of the generated image", default=2048),
    ) -> Path:
        """Run a single prediction on the model"""
        print(f"Prompt: {prompt}\nRandom seed: {seed}\nNumber of inference steps: {num_inference_steps}")
        
        image = self.pipe(
            prompt,
            output_type="pil",
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(seed),
            height=height,
            width=width,
        ).images[0]
        
        output_file = f"{prompt.replace(' ', '_')}.png"
        image.save(output_file)
        print(f"Image saved to {output_file}")
        
        return Path(output_file)

