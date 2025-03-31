import warnings
import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from ultralytics import YOLO
from ultralytics.utils import ASSETS, ops
from ultralytics.engine.results import Results
from ultralytics.models.yolo.segment.predict import SegmentationPredictor

# -------------------------------------------------------------------------------
# Custom Predictor Class
# -------------------------------------------------------------------------------
class RawSegmentationPredictor(SegmentationPredictor):
    def construct_result(self, pred, img, orig_img, img_path, proto):
        if not len(pred):
            masks = None
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape)
        else:
            self.pred = pred
            self.proto_mask = proto
            masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

        if masks is not None:
            keep = masks.sum((-2, -1)) > 0
            pred, masks = pred[keep], masks[keep]
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)

# -------------------------------------------------------------------------------
# Image Processing Functions
# -------------------------------------------------------------------------------
def process_and_save_prototypes(image_path, raw_predictor, save_folder='visualizations'):
    """
    Process a single image to generate prototype masks, save the visualization
    with a name derived from the input image, and return a DataFrame containing
    the image name and its 32 mask coefficients.
    """
    # Ensure the save folder exists.
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    base_name = os.path.basename(image_path)
    name_without_ext, _ = os.path.splitext(base_name)
    print(f"Processing: {name_without_ext}")
    
    # Define the save path for the visualization.
    save_path = os.path.join(save_folder, f"{name_without_ext}.png")
    
    # Process the image with the raw_predictor.
    raw_predictor.predict_cli(source=image_path)
    
    # Extract the prototype masks and mask weights.
    proto_masks = raw_predictor.proto_mask.cpu().numpy()  # shape: (32, H, W)
    mask_weights = raw_predictor.pred[:, 6:].cpu().numpy()[0]  # shape: (32,)
    
    print(mask_weights)
    
    # Determine grid layout for 32 prototypes (4 columns by 8 rows).
    n_protos = proto_masks.shape[0]
    cols = 4
    rows = n_protos // cols if n_protos % cols == 0 else (n_protos // cols) + 1
    
    # Create subplots to display each mask with its corresponding weight.
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axs = axs.flatten()
    
    for i in range(n_protos):
        ax = axs[i]
        ax.imshow(proto_masks[i])
        ax.set_title(f'Weight: {mask_weights[i]:.2f}', fontsize=10)
        ax.axis('off')
    
    # Hide any extra subplots.
    for j in range(n_protos, len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
    # Build a DataFrame with the image name and mask coefficients.
    data = {"image_name": [name_without_ext], "type": ["All"]}
    for idx, weight in enumerate(mask_weights):
        data[f"mask_{idx+1}"] = [weight]
    df = pd.DataFrame(data)
    return mask_weights, df

def process_all_images_in_folder(images_folder, raw_predictor, save_folder='visualizations', csv_filename="mask_coefficients.csv"):
    """
    Process all images in a folder by generating prototype mask visualizations,
    and collect each image's mask coefficients into a consolidated CSV file.
    """
    if os.path.exists(csv_filename):
        df_existing = pd.read_csv(csv_filename)
        processed_images = set(df_existing["image_name"].tolist())
    else:
        df_existing = pd.DataFrame()
        processed_images = set()
    
    new_entries = []
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for filename in tqdm(image_files, desc="Processing images"):
        name_without_ext, _ = os.path.splitext(filename)
        if name_without_ext in processed_images:
            print(f"Skipping {name_without_ext} (already processed).")
            continue
        
        image_path = os.path.join(images_folder, filename)
        try:
            _, df = process_and_save_prototypes(image_path, raw_predictor, save_folder)
            new_entries.append(df)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    if new_entries:
        new_df = pd.concat(new_entries, ignore_index=True)
        final_df = pd.concat([df_existing, new_df], ignore_index=True) if not df_existing.empty else new_df
        final_df.to_csv(csv_filename, index=False)
    else:
        final_df = df_existing
    return final_df

# -------------------------------------------------------------------------------
# Model and Predictor Initialization
# -------------------------------------------------------------------------------
def setup_model_and_predictor(model_path, overrides):
    """
    Initialize the YOLO model and custom segmentation predictor.
    """
    predictor = RawSegmentationPredictor(overrides=overrides)
    predictor.setup_model(model=model_path, verbose=True)
    return predictor

# -------------------------------------------------------------------------------
# Main Routine
# -------------------------------------------------------------------------------
def main():
    start_time = time.time()
    
    # Define paths.
    model_path = r"C:\Users\nam.nguyen\Project\models\yolo11x-seg.pt"
    images_folder = os.path.join("data")
    save_folder = "visualizations"
    csv_filename = "mask_coefficients.csv"
    
    # Define predictor overrides.
    args = dict(model=model_path, source=ASSETS)
    raw_predictor = setup_model_and_predictor(model_path, overrides=args)
    
    # Process all images in the data folder.
    df_all = process_all_images_in_folder(images_folder, raw_predictor, save_folder, csv_filename)
    
    elapsed_time = time.time() - start_time
    print(f"Processing complete. Data saved to {csv_filename}")
    print(f"Total processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
