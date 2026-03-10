import os
import cv2
import numpy as np
from keras import models

# Standard size (Must match what you trained your U-Net on)
IMG_SIZE = 192

class Segmentor:
    def __init__(self, disc_model_path):
        """Loads the trained Optic Disc U-Net model."""
        self.disc_model = models.load_model(disc_model_path, compile=False)
        print("✅ Optic Disc Model loaded successfully.")

    def predict_disc(self, image):
        """Predicts the Optic Disc mask."""
        img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
        # Ensure it is grayscale for the model input
        if len(img_resized.shape) == 3:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
        img_norm = img_resized / 255.0
        img_input = img_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        
        # Predict
        pred = self.disc_model.predict(img_input, verbose=0)[0, :, :, 0]
        
        # Convert prediction to binary mask (0 or 255)
        mask = (pred > 0.5).astype(np.uint8) * 255
        
        # Resize mask back to the original image dimensions
        mask_full = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask_full

# -------- BATCH PROCESSING --------
if __name__ == "__main__":
    # 1. SETUP RELATIVE PATHS
    ORIGINAL_DIR = "original"
    PROCESSED_DIR = "processed"
    
    # 🎯 YOUR NEW TARGET FOLDERS
    MASK_OUT_DIR = "binary_mask"
    RESULT_OUT_DIR = "result"
    
    DISC_MODEL_PATH = "optic_disc_unet.h5"

    # Automatically create the folders if they don't exist
    os.makedirs(MASK_OUT_DIR, exist_ok=True)
    os.makedirs(RESULT_OUT_DIR, exist_ok=True)

    # 2. INITIALIZE AI
    if not os.path.exists(DISC_MODEL_PATH):
        print(f"❌ Error: Could not find model at '{DISC_MODEL_PATH}'")
        exit()
        
    seg = Segmentor(DISC_MODEL_PATH)
    
    # 3. GET ALL ORIGINAL IMAGES
    image_files = [f for f in os.listdir(ORIGINAL_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"🚀 Found {len(image_files)} images. Starting predictions...\n")

    for filename in image_files:
        # Load the untouched original image to draw on
        orig_path = os.path.join(ORIGINAL_DIR, filename)
        original_img = cv2.imread(orig_path)
        
        # Load the exact same filename from the processed folder to feed to the AI
        processed_path = os.path.join(PROCESSED_DIR, filename)
        processed_img = cv2.imread(processed_path)
        
        if original_img is None:
            print(f"⚠️ Skipping {filename}: Could not find in '{ORIGINAL_DIR}'.")
            continue
        if processed_img is None:
            print(f"⚠️ Skipping {filename}: Could not find in '{PROCESSED_DIR}'.")
            continue

        # --- AI PREDICTION (Using the Processed Image) ---
        disc_mask = seg.predict_disc(processed_img)
        smooth_disc_mask = cv2.GaussianBlur(disc_mask, (15, 15), 0)

        # --- VISUALIZATION (Drawing on the Original Image) ---
        display_img = original_img.copy()
        
        # Find the boundary of the predicted mask
        d_contours, _ = cv2.findContours(smooth_disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(d_contours) > 0:
            largest_d = max(d_contours, key=cv2.contourArea)
            # An ellipse needs at least 5 points to be mathematically calculated
            if len(largest_d) >= 5:
                cv2.ellipse(display_img, cv2.fitEllipse(largest_d), (0, 255, 0), 2)

        # --- SAVE OUTPUTS TO SPECIFIC FOLDERS ---
        base_name = os.path.splitext(filename)[0]
        mask_save_path = os.path.join(MASK_OUT_DIR, f"{base_name}_mask.png")
        result_save_path = os.path.join(RESULT_OUT_DIR, f"{base_name}_result.jpg")
        
        cv2.imwrite(mask_save_path, disc_mask)
        cv2.imwrite(result_save_path, display_img)
        
        print(f"✅ Processed & Saved: {filename}")

    print(f"\n🎉 Finished! Check the '{MASK_OUT_DIR}' and '{RESULT_OUT_DIR}' folders.")