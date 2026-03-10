import cv2
import numpy as np
import os
import glob

def augment_retinal_images(input_folder="masks", output_folder="masked"):
    # --- 1. Create Output Directory ---
    os.makedirs(output_folder, exist_ok=True)
    
    # --- 2. Find all images ---
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        
    if not image_paths:
        print(f"No images found in '{input_folder}'.")
        return

    print(f"🚀 Found {len(image_paths)} images. Starting augmentation...\n")
    
    total_saved = 0

    # --- 3. Loop through and augment ---
    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        filename, ext = os.path.splitext(base_name)
        
        img = cv2.imread(image_path)
        if img is None:
            continue
            
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)

        # --- AUGMENTATION 1: Save Original ---
        cv2.imwrite(os.path.join(output_folder, f"{filename}_orig{ext}"), img)
        total_saved += 1
        
        # --- AUGMENTATION 2: Horizontal Flip ---
        h_flip = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(output_folder, f"{filename}_hflip{ext}"), h_flip)
        total_saved += 1
        
        # --- AUGMENTATION 3: Vertical Flip ---
        v_flip = cv2.flip(img, 0)
        cv2.imwrite(os.path.join(output_folder, f"{filename}_vflip{ext}"), v_flip)
        total_saved += 1
        
        # --- AUGMENTATION 4: Rotate +15 Degrees ---
        M_rot_pos = cv2.getRotationMatrix2D(center, 15, 1.0)
        rot_pos = cv2.warpAffine(img, M_rot_pos, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        cv2.imwrite(os.path.join(output_folder, f"{filename}_rot15{ext}"), rot_pos)
        total_saved += 1
        
        # --- AUGMENTATION 5: Rotate -15 Degrees ---
        M_rot_neg = cv2.getRotationMatrix2D(center, -15, 1.0)
        rot_neg = cv2.warpAffine(img, M_rot_neg, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        cv2.imwrite(os.path.join(output_folder, f"{filename}_rot_neg15{ext}"), rot_neg)
        total_saved += 1

    print(f"✅ Augmentation Complete!")
    print(f"📈 Your dataset grew from {len(image_paths)} to {total_saved} images.")
    print(f"📍 Images saved in: {os.path.abspath(output_folder)}")

# Execute
augment_retinal_images(input_folder="masks", output_folder="masked")