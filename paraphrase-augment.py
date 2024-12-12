import os
import shutil
import random
from pathlib import Path
import openai

# Set your OpenAI API key before running:
# export OPENAI_API_KEY=
openai.api_key = os.getenv("OPENAI_API_KEY")

# Adjust paths as necessary
BASE_DIR = Path("/root/RMTWTH")
ORIGINAL_DIR = BASE_DIR / "1"
MASK_DIR = BASE_DIR / "mask"
LOGO_MASK_DIR = BASE_DIR / "logo-mask"
OUTPUT_DIR = BASE_DIR / "augmented"

# Create OUTPUT_DIR if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "mask").mkdir(exist_ok=True)
(OUTPUT_DIR / "logo-mask").mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

def is_image(file_path: Path) -> bool:
    return file_path.suffix.lower() in IMAGE_EXTENSIONS

def paraphrase_caption(original_caption: str, iteration: int) -> str:
    """
    Use the OpenAI ChatCompletion endpoint to paraphrase the given caption.
    Add iteration info and slight randomness to encourage different outputs.
    """
    # Add iteration info to prompt
    # Also add a random word to the prompt to further encourage variation
    random_word = str(random.randint(10000, 99999))
    prompt = (
        "Paraphrase the following caption using simple terms while preserving its meaning. "
        f"This is attempt number {iteration}, try to phrase it differently than before. "
        f"Random token: {random_word}\n\n{original_caption}"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that paraphrases captions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=100
    )
    paraphrased = response.choices[0].message.content.strip()
    return paraphrased

def main():
    # Iterate multiple times to create multiple augmented versions
    for iter_ in range(3):
        for file_name in os.listdir(ORIGINAL_DIR):
            if file_name.endswith(".png"):
                stem = Path(file_name).stem  # e.g. RMTWTH_12
                txt_file = ORIGINAL_DIR / f"{stem}.txt"

                if txt_file.exists():
                    # Read original caption
                    with open(txt_file, "r") as f:
                        original_caption = f.read().strip()

                    # Paraphrase the caption with iteration info
                    paraphrased_caption = paraphrase_caption(original_caption, iter_)

                    # Construct new filenames
                    new_stem = f"{stem}_aug{iter_}"
                    new_png = f"{new_stem}.png"
                    new_txt = f"{new_stem}.txt"

                    # Copy and rename original image to OUTPUT_DIR
                    shutil.copy(ORIGINAL_DIR / file_name, OUTPUT_DIR / new_png)

                    # Write the new paraphrased caption in OUTPUT_DIR
                    with open(OUTPUT_DIR / new_txt, "w") as f:
                        f.write(paraphrased_caption)

                    # Copy and rename mask image if it exists
                    original_mask_path = MASK_DIR / file_name
                    if original_mask_path.exists():
                        mask_new_name = f"{new_stem}.png"
                        shutil.copy(original_mask_path, OUTPUT_DIR / 'mask' / mask_new_name)

                    # Copy and rename logo-mask image if it exists
                    original_logo_mask_path = LOGO_MASK_DIR / file_name
                    if original_logo_mask_path.exists():
                        logo_mask_new_name = f"{new_stem}.png"
                        shutil.copy(original_logo_mask_path, OUTPUT_DIR / 'logo-mask' / logo_mask_new_name)

                    print(f"Augmented {stem} -> {new_stem}")
                else:
                    print(f"No caption file found for {file_name}. Skipping.")

if __name__ == "__main__":
    main()
