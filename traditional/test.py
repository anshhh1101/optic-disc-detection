import cv2
import numpy as np
import os

def generate_disc_masks(input_folder="original", mask_folder="disc_mask", processed_folder="processed"):
    # 1. Setup Folders
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)

    # Get list of images
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"No images found in '{input_folder}'")
        return

    print(f"🚀 Generating Optic Disc masks for {len(image_files)} images...")
    success_count = 0

    for filename in image_files:
        image_path = os.path.join(input_folder, filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"  -> Error: Could not read {filename}. Skipping.")
            continue
            
        # Make a copy of the original image to draw the green outline on
        img_display = img.copy()
            
        # 2. Extract Red Channel
        b, g, r = cv2.split(img)
        
        # 3. FOV Masking
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, fov_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # 4. Detailed Morphology (Vessel Suppression)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        dilated = cv2.dilate(r, kernel, iterations=1)
        vessels_removed = cv2.erode(dilated, kernel, iterations=1)
        vessels_removed = cv2.bitwise_and(vessels_removed, vessels_removed, mask=fov_mask)
        
        # 5. Enhancement & Smoothing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized_img = clahe.apply(vessels_removed)
        blurred = cv2.medianBlur(normalized_img, 199)
        
        # 6. Percentile Thresholding
        valid_pixels = blurred[fov_mask > 0]
        if len(valid_pixels) == 0:
            continue
            
        threshold_value = np.percentile(valid_pixels, 98) 
        if threshold_value >= 254:
            threshold_value = 250 
        
        _, binary_mask = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
        
        # 7. Isolate Largest Contour for the Final Mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize a clean black image for the mask
        final_mask = np.zeros_like(binary_mask)

        if contours:
            valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                
                # --- NEW: Use fitEllipse to make it a perfect oval ---
                if len(largest_contour) >= 5:
                    ellipse = cv2.fitEllipse(largest_contour)
                    
                    # 1. Draw the filled white ellipse on the black mask
                    cv2.ellipse(final_mask, ellipse, 255, thickness=-1)
                    
                    # 2. Draw the green outline on our display image
                    cv2.ellipse(img_display, ellipse, (0, 255, 0), 4)
                    
                # 8. SAVE THE OUTPUTS
                mask_output_path = os.path.join(mask_folder, f"mask_{filename}")
                processed_output_path = os.path.join(processed_folder, f"annotated_{filename}")
                
                # Save the solid binary mask
                cv2.imwrite(mask_output_path, final_mask)
                # Save the original image with the green circle
                cv2.imwrite(processed_output_path, img_display)
                
                success_count += 1

    print(f"\nFinished! Generated {success_count} masks successfully.")
    print(f"Masks are in: {os.path.abspath(mask_folder)}")
    print(f"Annotated images are in: {os.path.abspath(processed_folder)}")

# Execute
generate_disc_masks(input_folder="original", mask_folder="disc_mask", processed_folder="processed")