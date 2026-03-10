import os
import cv2
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from keras import layers, models, callbacks

# -------- SETTINGS --------
IMG_SIZE = 192
BATCH_SIZE = 8
EPOCHS = 50 

# -------- RELATIVE PATHS --------
# EXACT matches based on your screenshots
IMAGE_DIR = "final_processed" 
MASK_DIR = "masked"
MODEL_SAVE_PATH = "optic_disc_unet.h5" 

# -------- MODEL ARCHITECTURE (CNN-U-Net) --------
def build_unet():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 1))
    
    # Encoder
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D()(c1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D()(c2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    p3 = layers.MaxPooling2D()(c3)
    
    # Bottleneck
    c4 = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
    
    # Decoder
    u1 = layers.UpSampling2D()(c4)
    u1 = layers.concatenate([u1, c3])
    c5 = layers.Conv2D(128, 3, activation='relu', padding='same')(u1)
    
    u2 = layers.UpSampling2D()(c5)
    u2 = layers.concatenate([u2, c2])
    c6 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    
    u3 = layers.UpSampling2D()(c6)
    u3 = layers.concatenate([u3, c1])
    c7 = layers.Conv2D(32, 3, activation='relu', padding='same')(u3)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c7)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -------- MAIN TRAINING LOOP --------
images = []
# Safely searches for all common image types
for ext in ('*.jpg', '*.jpeg', '*.png'):
    images.extend(glob(os.path.join(IMAGE_DIR, ext)))

images = sorted(images)

print(f"Found {len(images)} images in '{IMAGE_DIR}'")

X, Y = [], []
for img_p in images:
    # 1. Get the base name (e.g., '001_hflip')
    base_name = os.path.splitext(os.path.basename(img_p))[0]
    
    # 2. Split it into the ID ('001') and the augmentation type ('hflip' or 'rot_neg15')
    parts = base_name.split("_", 1)
    
    # 3. Rebuild the string to match your exact mask folder naming
    if len(parts) == 2:
        mask_filename = f"{parts[0]}_mask_{parts[1]}.png"  # Creates '001_mask_hflip.png'
    else:
        mask_filename = f"{base_name}_mask.png"            # Fallback for plain '001.jpg'
        
    mask_p = os.path.join(MASK_DIR, mask_filename)

    if not os.path.exists(mask_p):
        continue 

    img = cv2.imread(img_p, 0) 
    mask = cv2.imread(mask_p, 0)
    
    if img is None or mask is None:
        continue
        
    # Resize and Normalize to 0.0 - 1.0
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)) / 255.0
    
    # Ensure mask is strictly binary (0 or 1)
    mask = (mask > 0.5).astype(np.float32)
    
    X.append(img)
    Y.append(mask)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = np.array(Y).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print(f"Training on {len(X)} matching image-mask pairs.")

if len(X) == 0:
    print("❌ Error: No matching pairs found. Check your folder paths.")
    exit()

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

model = build_unet()
saver = callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss', verbose=1)

print("🚀 Starting CNN-U-Net Training...")
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[saver])
print(f"✅ Training Complete. Model saved to {MODEL_SAVE_PATH}")