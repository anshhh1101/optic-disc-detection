import cv2
import numpy as np
import os
import glob

def batch_preprocess_vgg(input_folder="original", output_folder="processed"):
    # --- 1. Create Output Directory ---
    os.makedirs(output_folder, exist_ok=True)
    
    # --- 2. Find all images in the input folder ---
    image_paths = []
    # Search for common image formats
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        
    if not image_paths:
        print(f"No images found in the '{input_folder}' folder.")
        return

    print(f"🚀 Found {len(image_paths)} images. Starting VGG preprocessing...\n")

    # --- 3. Loop through each image ---
    success_count = 0
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        print(f"Processing: {filename}...")
        
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"  -> Error: Could not read {filename}. Skipping.")
            continue
            
        # FOV Masking (to keep the background perfectly black later)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, fov_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # STEP 1: Extract Red Channel
        b, g, r = cv2.split(img)

        # STEP 2: Gamma Correction (Applied directly to the Red channel)
        # Gamma < 1.0 brightens midtones. Gamma > 1.0 darkens them.
        gamma = 0.9
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(r, table)

        # STEP 3: Morphology (Vessel Suppression)
        # 9x9 kernel to erase thin vessels without melting the optic disc
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        vessels_removed = cv2.morphologyEx(gamma_corrected, cv2.MORPH_CLOSE, kernel)
        
        # Apply the FOV mask to clean up the edges
        vessels_removed = cv2.bitwise_and(vessels_removed, vessels_removed, mask=fov_mask)

        # STEP 4: CLAHE (Contrast Enhancement)
        # 2.5 clip limit keeps contrast high without amplifying static
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_img = clahe.apply(vessels_removed)

        # STEP 5: Median Blur 
        # 3x3 kernel smooths micro-noise while keeping edges sharp
        blurred = cv2.medianBlur(enhanced_img, 21)

        # Final Formatting: Convert to 3-Channel for VGG Input
        vgg_ready = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

        # --- Save the Output ---
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, vgg_ready)
        success_count += 1

    print(f"\n✅ Finished! Preprocessed {success_count} images successfully.")
    print(f"📍 Images are ready for your CNN in: {os.path.abspath(output_folder)}")

# Example usage:
batch_preprocess_vgg(input_folder="original", output_folder="processed")