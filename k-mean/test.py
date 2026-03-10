import cv2
import numpy as np
import os
import glob

def batch_ellipse_roi_kmeans(input_folder="original", mask_folder="disc_masked", processed_folder="processed", roi_size=700):
    # --- 1. Create Output Directories ---
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)
    
    # --- 2. Find all images in the input folder ---
    image_paths = []
    # Search for common image formats
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        
    if not image_paths:
        print(f"No images found in the '{input_folder}' folder.")
        return

    print(f"Found {len(image_paths)} images. Starting batch processing...\n")

    # --- 3. Loop through each image ---
    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        filename, ext = os.path.splitext(base_name)
        print(f"Processing: {base_name}...")
        
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"  -> Error: Could not read {image_path}. Skipping.")
            continue
            
        b, g, r = cv2.split(img)
        img_display = img.copy()
        
        # =========================================================
        # --- STEP 1: ROUGH LOCALIZATION (Grayscale) ---
        # =========================================================
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1a. Strong CLAHE to find the brightest region
        clahe_loc = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        norm_gray = clahe_loc.apply(gray)
        
        # 1b. Gamma Correction to crush the background brightness (Localization)
        gamma_loc = 2.5 
        table_loc = np.array([((i / 255.0) ** gamma_loc) * 255 for i in np.arange(0, 256)]).astype("uint8")
        norm_gray = cv2.LUT(norm_gray, table_loc)
        
        blurred_for_loc = cv2.GaussianBlur(norm_gray, (151, 151), 0)
        _, _, _, max_loc = cv2.minMaxLoc(blurred_for_loc)
        
        center_x, center_y = max_loc
        half_size = roi_size // 2
        
        y1 = max(0, center_y - half_size)
        y2 = min(img.shape[0], center_y + half_size)
        x1 = max(0, center_x - half_size)
        x2 = min(img.shape[1], center_x + half_size)
        
        # NOTE: The cv2.rectangle line has been completely removed to keep the output clean!
        
        # =========================================================
        # --- STEP 2: SEGMENTATION (Red Channel) ---
        # =========================================================
        roi_r = r[y1:y2, x1:x2]
        
        # 2a. Remove blood vessels FIRST so CLAHE doesn't amplify them
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        vessels_removed_roi = cv2.morphologyEx(roi_r, cv2.MORPH_CLOSE, kernel)
        
        # 2b. Apply CLAHE specifically to the ROI to stretch Optic Disc vs Retina
        clahe_roi = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced_roi = clahe_roi.apply(vessels_removed_roi)
        
        # 2c. Apply Gamma to the ROI to darken the retina back down
        gamma_roi = 4.0 
        table_roi = np.array([((i / 255.0) ** gamma_roi) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced_roi = cv2.LUT(enhanced_roi, table_roi)
        
        # 2d. Final blur before clustering
        blurred_roi = cv2.medianBlur(enhanced_roi, 51)
        
        # =========================================================
        # --- STEP 3: K-MEANS (K=3) ---
        # =========================================================
        pixel_values = blurred_roi.reshape((-1, 1))
        pixel_values = np.float32(pixel_values)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers_uint8 = np.uint8(centers)
        clustered_roi = centers_uint8[labels.flatten()].reshape(blurred_roi.shape)
        
        # =========================================================
        # --- STEP 4: Mask Generation (Using fitEllipse) ---
        # =========================================================
        brightest_cluster_idx = np.argmax(centers)
        roi_binary_mask = (labels.reshape(blurred_roi.shape) == brightest_cluster_idx).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(roi_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_final_mask = np.zeros_like(roi_binary_mask)
        full_final_mask = np.zeros_like(gray)
        
        if contours:
            valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
            
            if valid_contours:
                largest_contour_roi = max(valid_contours, key=cv2.contourArea)
                
                if len(largest_contour_roi) >= 5:
                    ellipse_roi = cv2.fitEllipse(largest_contour_roi)
                    
                    # Draw the PERFECT, filled ellipse onto our ROI mask
                    cv2.ellipse(roi_final_mask, ellipse_roi, 255, thickness=-1)
                    
                    # --- Map the Ellipse Back to the Full Image ---
                    (cx, cy), (w, h), angle = ellipse_roi
                    ellipse_full = ((cx + x1, cy + y1), (w, h), angle)
                    
                    # Draw the green outline on the display image (NO rectangle)
                    cv2.ellipse(img_display, ellipse_full, (0, 255, 0), 4)
                    
                full_final_mask[y1:y2, x1:x2] = roi_final_mask

        # =========================================================
        # --- 5. Save the Outputs directly to folders ---
        # =========================================================
        # Save the pure binary mask to disc_masked folder
        mask_output_path = os.path.join(mask_folder, f"{filename}_mask.png")
        cv2.imwrite(mask_output_path, full_final_mask)
        
        # Save the annotated full image to processed folder
        processed_output_path = os.path.join(processed_folder, f"{filename}_processed.jpg")
        cv2.imwrite(processed_output_path, img_display)
        
    print("\nBatch processing is complete!")

# Example usage to run the batch script:
# Make sure you have a folder named 'original' with your images in the same directory!
batch_ellipse_roi_kmeans(input_folder="original", mask_folder="disc_masked", processed_folder="processed", roi_size=700)