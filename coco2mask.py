import json
import os
from PIL import Image, ImageDraw
import numpy as np

def create_binary_masks(
    annotations_file='annotations.json',
    images_dir='.',
    output_dir='masks',
    category_ids=None
):
    os.makedirs(output_dir, exist_ok=True)
    with open(annotations_file, 'r') as f:
        coco = json.load(f)

    images = coco.get('images', [])
    image_id_to_info = {image['id']: image for image in images}
    annotations = coco.get('annotations', [])
    image_id_to_annotations = {}
    for ann in annotations:
        img_id = ann['image_id']
        if category_ids is not None and ann['category_id'] not in category_ids:
            continue  # Skip annotations not in the specified categories
        if img_id not in image_id_to_annotations:
            image_id_to_annotations[img_id] = []
        image_id_to_annotations[img_id].append(ann)

    for image_id, image_info in image_id_to_info.items():
        file_name = image_info['file_name']
        width = image_info['width']
        height = image_info['height']

        image_path = os.path.join(images_dir, file_name)
        if not os.path.exists(image_path):
            print(f"Image file {file_name} not found in {images_dir}. Skipping.")
            continue

        # Initialize a blank mask
        mask = Image.new('L', (width, height), 0)  # 'L' mode for (8-bit pixels, black and white)
        draw = ImageDraw.Draw(mask)

        anns = image_id_to_annotations.get(image_id, [])
        for ann in anns:
            segmentation = ann.get('segmentation', [])
            if not segmentation:
                continue  # Skip if no segmentation

            if isinstance(segmentation, list):
                for seg in segmentation:
                    if len(seg) < 6:
                        continue  # Not a valid polygon
                    polygon = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                    draw.polygon(polygon, outline=255, fill=255)
            else:
                print(f"RLE segmentation found in annotation {ann['id']}. Skipping.")
                continue

        # Save the mask
        base_name, ext = os.path.splitext(file_name)
        mask_filename = f"{base_name}.png"
        mask_path = os.path.join(output_dir, mask_filename)
        mask.save(mask_path)
        print(f"Saved mask for {file_name} as {mask_filename}")

if __name__ == "__main__":
    create_binary_masks(
        annotations_file='/root/RMTWTH/logo-mask.json',
        images_dir='/root/RMTWTH/1',
        output_dir='/root/RMTWTH/logo-mask',
        category_ids=[4]
    )
