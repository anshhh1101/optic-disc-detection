import json
import cv2
import numpy as np
import os
import math

# -------- RELATIVE PATHS --------
# Make sure your original 181-350 images are inside the 'processed' folder
IMAGE_DIR = "processed"       
JSON_PATH = "via_project_4Mar2026_21h53m (2).json"     
SAVE_DIR = "masks"            

os.makedirs(SAVE_DIR, exist_ok=True)

# -------- LOAD JSON --------
if not os.path.exists(JSON_PATH):
    print(f"❌ Error: JSON file not found at '{JSON_PATH}'")
    exit()

with open(JSON_PATH, "r") as f:
    data = json.load(f)

# --- AUTO-DETECT JSON FORMAT ---
# This safely bypasses the "_via_settings" and targets the images directly
if "_via_img_metadata" in data:
    metadata = data["_via_img_metadata"]
else:
    metadata = data

print("Total annotated images in JSON:", len(metadata))

created = 0
skipped = 0

for key, item in metadata.items():
    # Safely skip any stray project settings if they slipped through
    if "filename" not in item:
        continue
        
    filename = item["filename"]
    regions = item.get("regions", [])

    # Check if the image has any annotations
    if len(regions) == 0:
        skipped += 1
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(image_path)

    if img is None:
        print(f"⚠️ Image not found or unreadable: {filename}")
        skipped += 1
        continue

    h, w = img.shape[:2]

    # Create empty black mask
    mask = np.zeros((h, w), dtype=np.uint8)

    for region in regions:
        shape = region["shape_attributes"]

        if shape["name"] == "ellipse":
            # Using round() to gracefully handle decimal pixels from VIA
            cx = int(round(shape["cx"]))
            cy = int(round(shape["cy"]))
            rx = int(round(shape["rx"]))
            ry = int(round(shape["ry"]))
            
            theta_rad = float(shape.get("theta", 0))
            theta_deg = math.degrees(theta_rad)

            cv2.ellipse(mask, (cx, cy), (rx, ry), theta_deg, 0, 360, 255, -1)
            
        elif shape["name"] == "circle":
            cx = int(round(shape["cx"]))
            cy = int(round(shape["cy"]))
            r  = int(round(shape["r"]))
            cv2.circle(mask, (cx, cy), r, 255, -1)

    # Save format matches the U-Net training script expectations
    mask_name = filename.replace(".jpg", "_mask.png")
    save_path = os.path.join(SAVE_DIR, mask_name)

    cv2.imwrite(save_path, mask)
    created += 1

print("-" * 30)
print(f"✅ Masks successfully created: {created}")
print(f"⚠️ Images skipped (not found/no regions): {skipped}")
print(f"📍 Saved in folder: {SAVE_DIR}")
print("Process Complete.")