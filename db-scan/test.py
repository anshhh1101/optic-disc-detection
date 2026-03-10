import cv2
import numpy as np
import os
import glob
from sklearn.cluster import DBSCAN

def batch_ellipse_roi_dbscan(input_folder="original", mask_folder="disc_masked", processed_folder="processed", roi_size=700):
    # --- 1. Create Output Directories ---
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(processed_folder, exist_ok=True)
    
    # --- 2. Find all images in the input folder ---
    image_paths = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        
    if not image_paths:
        print(f"No images found in the '{input_folder}' folder.")
        return

    print(f"Found {len(image_paths)} images. Starting DBSCAN batch processing...\n")

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
        
        clahe_loc = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        norm_gray = clahe_loc.apply(gray)
        
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
        
        # =========================================================
        # --- STEP 2: PREPROCESSING (Red Channel) ---
        # =========================================================
        roi_r = r[y1:y2, x1:x2]
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        vessels_removed_roi = cv2.morphologyEx(roi_r, cv2.MORPH_CLOSE, kernel)
        
        clahe_roi = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced_roi = clahe_roi.apply(vessels_removed_roi)
        
        gamma_roi = 4.0 
        table_roi = np.array([((i / 255.0) ** gamma_roi) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced_roi = cv2.LUT(enhanced_roi, table_roi)
        
        # UPDATE #1: Reduced blur to preserve tight edges
        blurred_roi = cv2.medianBlur(enhanced_roi, 15)
        
        # =========================================================
        # --- STEP 3: DBSCAN CLUSTERING ---
        # =========================================================
        # UPDATE #2: Back to Otsu's thresholding to prevent total blackout
        _, thresh = cv2.threshold(blurred_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Extract the (y, x) coordinates of those bright pixels
        coords = np.column_stack(np.where(thresh > 0))
        
        roi_binary_mask = np.zeros_like(blurred_roi)
        
        if len(coords) > 0:
            # Run DBSCAN on the coordinates
            db = DBSCAN(eps=15, min_samples=40).fit(coords)
            labels = db.labels_
            
            # Ignore the noise label (-1) and find the largest cluster
            unique_labels = set(labels)
            unique_labels.discard(-1) 
            
            if unique_labels:
                # Find the label with the most coordinates attached to it
                largest_cluster_label = max(unique_labels, key=lambda l: np.sum(labels == l))
                
                # Get the coordinates that belong ONLY to the optic disc
                disc_coords = coords[labels == largest_cluster_label]
                
                # Draw those pixels onto our blank mask
                for y, x in disc_coords:
                    roi_binary_mask[y, x] = 255
                    
        # =========================================================
        # --- STEP 4: Mask Generation (Using fitEllipse) ---
        # =========================================================
        contours, _ = cv2.findContours(roi_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_final_mask = np.zeros_like(roi_binary_mask)
        full_final_mask = np.zeros_like(gray)
        
        if contours:
            valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
            
            if valid_contours:
                largest_contour_roi = max(valid_contours, key=cv2.contourArea)
                
                if len(largest_contour_roi) >= 5:
                    # UPDATE #3: The Convex Hull "rubber band" trick to fix vessel gaps
                    hull = cv2.convexHull(largest_contour_roi)
                    ellipse_roi = cv2.fitEllipse(hull)
                    
                    # Draw the perfectly smooth ellipse
                    cv2.ellipse(roi_final_mask, ellipse_roi, 255, thickness=-1, lineType=cv2.LINE_AA)
                    
                    # --- Map the Ellipse Back to the Full Image ---
                    (cx, cy), (w, h), angle = ellipse_roi
                    ellipse_full = ((cx + x1, cy + y1), (w, h), angle)
                    
                    cv2.ellipse(img_display, ellipse_full, (0, 255, 0), 4, lineType=cv2.LINE_AA)
                    
                    full_final_mask[y1:y2, x1:x2] = roi_final_mask

        # =========================================================
        # --- 5. Save the Outputs directly to folders ---
        # =========================================================
        mask_output_path = os.path.join(mask_folder, f"{filename}_mask.png")
        cv2.imwrite(mask_output_path, full_final_mask)
        
        processed_output_path = os.path.join(processed_folder, f"{filename}_processed.jpg")
        cv2.imwrite(processed_output_path, img_display)
        
    print("\nBatch processing is complete!")

if __name__ == "__main__":
    batch_ellipse_roi_dbscan()