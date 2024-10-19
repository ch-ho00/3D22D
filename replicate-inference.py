import replicate

prompt = "a silver metal WTHCTR"
output = replicate.run(
    "ch-ho00/cartier-model2-ft2:2a18f8c55504f8cecd9230142b1d2f2579d49c2018aeb65ad0426b0b266574f9",
    input={
        "prompt" : prompt,
        "model": "dev",
        "lora_scale": 1,
        "num_outputs": 1,
        "aspect_ratio": "1:1",
        "output_format": "webp",
        "guidance_scale": 3.5,
        "output_quality": 90,
        "prompt_strength": 0.8,
        "extra_lora_scale": 1,
        "num_inference_steps": 28
    }
)
print(output)

import pdb; pdb.set_trace()
print()