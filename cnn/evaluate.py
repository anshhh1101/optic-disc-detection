import cv2
import numpy as np
import os

def calculate_metrics(pred_mask, gt_mask):
    """Computes Dice, IoU, and Accuracy for binary masks."""
    # Ensure binary format (0 or 1)
    pred = (pred_mask > 127).astype(np.uint8)
    gt = (gt_mask > 127).astype(np.uint8)

    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))

    # Metrics calculation
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return dice, iou, accuracy

# --- REMOVED: max_limit parameter ---
def run_evaluation(pred_folder="binary_mask", gt_folder="disc_groundtruth"):
    dice_scores, iou_scores, acc_scores = [], [], []
    
    # Get your generated mask files
    pred_files = [f for f in os.listdir(pred_folder) if f.lower().endswith(('.png', '.jpg'))]
    
    # Sort the files sequentially so it goes from 001 to 520 in order
    pred_files.sort()
    
    # --- REMOVED: pred_files = pred_files[:max_limit] ---
    
    print(f"🚀 Evaluating all {len(pred_files)} masks from '{pred_folder}'...")
    print("-" * 60)

    for f in pred_files:
        # Load your prediction
        pred_path = os.path.join(pred_folder, f)
        pred_img = cv2.imread(pred_path, 0)

        # NAME MATCHING LOGIC
        image_id = f.split("_")[0] 
        
        # Look for the exact same name ('001_mask.png') in the ground truth folder
        gt_filename = f"{image_id}_mask.png" 
        gt_path = os.path.join(gt_folder, gt_filename)

        gt_img = cv2.imread(gt_path, 0)

        if gt_img is not None:
            # Resize if dimensions differ
            if pred_img.shape != gt_img.shape:
                pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))

            dice, iou, acc = calculate_metrics(pred_img, gt_img)
            
            dice_scores.append(dice)
            iou_scores.append(iou)
            acc_scores.append(acc)
            
            print(f"✅ ID: {image_id} | Dice: {dice:.4f} | IoU: {iou:.4f}")
        else:
            print(f"❌ Missing GT: Could not find '{gt_filename}' in '{gt_folder}'")

    # Final Summary
    if dice_scores:
        print("-" * 60)
        print(f"⭐ FINAL PERFORMANCE SUMMARY ⭐")
        print(f"Mean Dice Coefficient: {np.mean(dice_scores):.4f}")
        print(f"Mean IoU (Jaccard):    {np.mean(iou_scores):.4f}")
        print(f"Mean Pixel Accuracy:   {np.mean(acc_scores):.4f}")
        print("-" * 60)

if __name__ == "__main__":
    run_evaluation()