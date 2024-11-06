import argparse
import os
import replicate
from pathlib import Path
from tqdm import tqdm

def generate_caption(image_path, model_version):
    with open(image_path, "rb") as image_file:
        output = replicate.run(
            model_version,
            input={"image": image_file}
        )
    return output

def process_image(image_path, input_folder, output_base_folder, model_version):
    relative_path = os.path.relpath(image_path, input_folder)
    output_path = os.path.splitext(os.path.join(output_base_folder, relative_path))[0] + ".txt"
    
    if os.path.exists(output_path):
        print(f"Skipping: {relative_path} already captioned.")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    caption = generate_caption(image_path, model_version)
    if caption:
        with open(output_path, "w") as f:
            f.write(caption)
        print(f"Captioned: {relative_path}")
    else:
        print(f"Failed to caption: {relative_path}")

def is_image_file(filename):
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS

def main(input_folder, output_base_folder):
    model_version = "lucataco/joy-caption-pre-alpha:31665fdccd897d20cbda1fa305e64f1b94a181e0350409ed2a40df7a243830a5"
    image_files = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(input_folder)
        for file in files if is_image_file(file)
    ]
    
    print(f"Found {len(image_files)} image(s) to process.")
    
    for image_path in tqdm(image_files):
        process_image(image_path, input_folder, output_base_folder, model_version)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for all images in a nested directory structure.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing images.")
    parser.add_argument("output_base_folder", type=str, help="Path to the base folder for saving output captions.")
    args = parser.parse_args()
    main(args.input_folder, args.output_base_folder)
