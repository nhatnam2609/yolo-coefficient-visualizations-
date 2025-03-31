import warnings
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse

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
        """
        Constructs the result object from the prediction.
        """
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
def process_and_save_coefficient(image_path, raw_predictor, coefficient_index, save_folder):
    """
    Process an image to extract the mask and weight for a given coefficient index,
    save the mask visualization, and return a DataFrame with the image name and the coefficient weight.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    base_name = os.path.basename(image_path)
    name_without_ext, _ = os.path.splitext(base_name)
    print(f"Processing: {name_without_ext}")
    
    # Process the image using the predictor
    raw_predictor.predict_cli(source=image_path)
    
    # Extract the prototype masks and mask weights.
    proto_masks = raw_predictor.proto_mask.cpu().numpy()  # shape: (32, H, W)
    mask_weights = raw_predictor.pred[:, 6:].cpu().numpy()[0]  # shape: (32,)
    
    # Validate the coefficient index.
    if coefficient_index < 0 or coefficient_index >= proto_masks.shape[0]:
        raise ValueError(f"Coefficient index must be between 0 and {proto_masks.shape[0]-1}, but got {coefficient_index}.")
    
    # Get the selected mask and its corresponding weight.
    mask_selected = proto_masks[coefficient_index]
    weight_selected = mask_weights[coefficient_index]
    
    # Save the selected mask visualization.
    save_path = os.path.join(save_folder, f"{name_without_ext}_mask{coefficient_index+1}.png")
    plt.figure(figsize=(4, 4))
    plt.imshow(mask_selected, cmap='viridis')
    plt.title(f'Mask {coefficient_index+1}\nWeight: {weight_selected:.2f}', fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Build a DataFrame with the image name and selected mask coefficient.
    data = {
        "image_name": [name_without_ext],
        f"mask{coefficient_index+1}_weight": [weight_selected]
    }
    df = pd.DataFrame(data)
    return weight_selected, df

def process_all_images_in_folder(images_folder, raw_predictor, coefficient_index, save_folder, csv_filename):
    """
    Process all images in a folder by extracting the selected mask coefficient,
    saving the visualizations, and compiling the coefficients into a CSV file.
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
            _, df = process_and_save_coefficient(image_path, raw_predictor, coefficient_index, save_folder)
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
    parser = argparse.ArgumentParser(
        description="Extract a specific mask coefficient from YOLO segmentation for all images in a folder."
    )
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO segmentation model file.")
    parser.add_argument("--images_folder", type=str, default="data", help="Folder containing input images.")
    parser.add_argument("--save_folder", type=str, default="visualizations", help="Folder to save mask visualizations.")
    parser.add_argument("--csv_filename", type=str, default="mask_coefficient.csv", help="CSV file to store the coefficient values.")
    parser.add_argument("--coefficient_index", type=int, required=True, help="Coefficient index to extract (0-based index).")
    
    args = parser.parse_args()
    
    start_time = time.time()
    raw_predictor = setup_model_and_predictor(args.model, overrides=dict(model=args.model, source=ASSETS))
    df_all = process_all_images_in_folder(args.images_folder, raw_predictor, args.coefficient_index, args.save_folder, args.csv_filename)
    elapsed_time = time.time() - start_time
    print(f"Processing complete. Data saved to {args.csv_filename}")
    print(f"Total processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
