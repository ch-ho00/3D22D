import os
import replicate
import boto3
from pathlib import Path
import requests
from PIL import Image

# Configuration
s3_client = boto3.client('s3')
BUCKET_NAME = 'flux-finetune-bucket'
replicate.api_token = os.getenv("REPLICATE_API_TOKEN")

ROOT_DIR = Path("/root/RichemontRenders10Vars/1")
OUTPUT_DIR = Path("/root/masked_images")
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

def is_image(file_path):
    return file_path.suffix.lower() in IMAGE_EXTENSIONS

def upload_to_s3(file_path):
    try:
        s3_client.upload_file(str(file_path), BUCKET_NAME, file_path.name, ExtraArgs={'ACL': 'public-read'})
        return f"https://{BUCKET_NAME}.s3.amazonaws.com/{file_path.name}"
    except Exception as e:
        print(f"Failed to upload {file_path} to S3: {e}")
        return None

def process_image(image_path, output_path):
    # Upload image and get URL
    image_url = upload_to_s3(image_path)
    if not image_url:
        print(f"Skipping {image_path} due to upload failure.")
        return

    # Run the Replicate model to get the segmented image (output[0])
    output = replicate.run(
        "logerzhu/ad-inpaint:b1c17d148455c1fda435ababe9ab1e03bc0d917cc3cf4251916f22c45c83c7df",
        input={
            "pixel": "1024 * 1024",
            "scale": 3,
            "prompt": "watch",
            "image_num": 1,
            "image_path": image_url,
            "manual_seed": -1,
            "product_size": "0.5 * width",
            "guidance_scale": 7.5,
            "negative_prompt": "illustration, 3d, sepia, painting, cartoons, sketch, (worst quality:2)",
            "num_inference_steps": 1
        }
    )

    # Download the segmented image
    response = requests.get(output[0])
    if response.status_code != 200:
        print(f"Failed to download segmented image for {image_path}")
        return
    
    # Load segmented image and resize to the original image's dimensions
    with open(output_path, "wb") as f:
        f.write(response.content)

    # Resize the mask to match the original image size
    original_img = Image.open(image_path)
    segmented_img = Image.open(output_path).convert("RGBA") 
    segmented_resized = segmented_img.resize(original_img.size, Image.ANTIALIAS)
    alpha = segmented_resized.split()[-1]
    alpha.save(output_path)
    print(f"Processed and saved mask: {output_path}")

def main():
    for subdir, _, files in os.walk(ROOT_DIR):
        for file in files:
            file_path = Path(subdir) / file
            if is_image(file_path):
                relative_path = file_path.relative_to(ROOT_DIR)
                output_file_path = OUTPUT_DIR / relative_path.parent / f"mask_{relative_path.name}"
                output_file_path.parent.mkdir(parents=True, exist_ok=True)
                process_image(file_path, output_file_path)

if __name__ == "__main__":
    main()
