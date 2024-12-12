import os
import shutil
from pathlib import Path

# Base directories
BASE_DIR = Path("/root")
SOURCE_DIR = BASE_DIR / "RMTWTH"
CAPTION_SOURCE_DIR = BASE_DIR / "caption_"
MERGED_DIR = BASE_DIR / "merged_folder"

# Create the merged directory structure
(MERGED_DIR / "1").mkdir(parents=True, exist_ok=True)
(MERGED_DIR / "mask").mkdir(exist_ok=True)
(MERGED_DIR / "logo-mask").mkdir(exist_ok=True)

# These are the sets of additional captions we want to integrate
caption_sets = [
    ("CAPTION", "_cap"),
    ("DETAILED_CAPTION", "_det"),
    ("MORE_DETAILED_CAPTION", "_more")
]

# Copy original images and text from RMTWTH/1 to merged_folder/1
# Also copy corresponding masks and logo-masks
original_dir = SOURCE_DIR / "1"
mask_dir = SOURCE_DIR / "mask"
logo_mask_dir = SOURCE_DIR / "logo-mask"

for file_name in os.listdir(original_dir):
    if file_name.endswith(".png"):
        stem = Path(file_name).stem  # e.g. RMTWTH_2
        # Copy the original image to merged_folder/1
        shutil.copy(original_dir / file_name, MERGED_DIR / "1" / file_name)
        
        # Copy the corresponding mask if exists
        mask_file = mask_dir / file_name
        if mask_file.exists():
            shutil.copy(mask_file, MERGED_DIR / "mask" / file_name)
        
        # Copy the corresponding logo-mask if exists
        logo_mask_file = logo_mask_dir / file_name
        if logo_mask_file.exists():
            shutil.copy(logo_mask_file, MERGED_DIR / "logo-mask" / file_name)
        
        # Copy the corresponding original caption if exists
        txt_file = original_dir / f"{stem}.txt"
        if txt_file.exists():
            shutil.copy(txt_file, MERGED_DIR / "1" / f"{stem}.txt")
        else:
            print(f"No original caption found for {stem}")

        # Now create the augmented sets (_cap, _det, _more)
        for (caption_subdir, suffix) in caption_sets:
            # Find the corresponding caption file in each caption set
            caption_file = CAPTION_SOURCE_DIR / caption_subdir / f"{stem}.txt"
            if caption_file.exists():
                # Copy image again with suffix
                new_image_name = f"{stem}{suffix}.png"
                shutil.copy(original_dir / file_name, MERGED_DIR / "1" / new_image_name)
                
                # Copy mask with suffix
                if mask_file.exists():
                    mask_image_name = f"{stem}{suffix}.png"
                    shutil.copy(mask_file, MERGED_DIR / "mask" / mask_image_name)
                
                # Copy logo-mask with suffix
                if logo_mask_file.exists():
                    logo_mask_image_name = f"{stem}{suffix}.png"
                    shutil.copy(logo_mask_file, MERGED_DIR / "logo-mask" / logo_mask_image_name)
                
                # Copy caption with suffix
                new_caption_name = f"{stem}{suffix}.txt"
                shutil.copy(caption_file, MERGED_DIR / "1" / new_caption_name)
            else:
                print(f"No {suffix} caption found for {stem}")

print("Merging and augmentation completed successfully.")
