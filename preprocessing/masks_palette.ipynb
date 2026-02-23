import os
import numpy as np
from PIL import Image

# Define class-to-ID mapping
CLASS_MAP = {
    "white_matter": 1,
    "lateral_ventricles": 2,
    "grey_matter": 3,
    "external_cerebrospinal_fluid": 4,
    "deep_grey_matter": 5,
    "cerebellum": 6,
    "brain_stem": 7
}

# Define the color palette (256 entries for 8-bit)
PALETTE = [
    0, 0, 0,       # Background (class 0 - Black)
    0, 0, 255,     # Class 1 (Blue - white_matter)
    255, 255, 0,   # Class 2 (Yellow - lateral_ventricles)
    255, 0, 255,   # Class 3 (Magenta - grey_matter)
    255, 165, 0,   # Class 4 (Orange - external_cerebrospinal_fluid)
    192, 192, 192, # Class 5 (Light Gray - deep_grey_matter)
    0, 0, 128,     # Class 6 (Dark Blue - cerebellum)
    128, 0, 128    # Class 7 (Purple - brain_stem)
]

# Fill the rest of the palette with zeros to ensure 256 entries
PALETTE.extend([0, 0, 0] * (256 - len(PALETTE) // 3))

def combine_masks(input_folder, output_folder):
    """
    Combines masks from class folders into a single palette-based mask.

    :param input_folder: Path to the folder containing class subfolders.
    :param output_folder: Path to save combined palette-based masks.
    """
    # Get all class folders
    class_folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

    # Process each image by filename
    image_files = set(os.listdir(os.path.join(input_folder, class_folders[0])))
    for image_file in image_files:
        # Create a blank mask
        combined_mask = None

        for class_name, class_id in CLASS_MAP.items():
            class_folder = os.path.join(input_folder, class_name)
            mask_path = os.path.join(class_folder, image_file)
            
            if os.path.exists(mask_path):
                # Load the mask as a binary array
                mask = Image.open(mask_path).convert("L")
                mask_array = np.array(mask, dtype=np.uint8)
                
                if combined_mask is None:
                    combined_mask = np.zeros_like(mask_array, dtype=np.uint8)
                
                # Add the class ID to the combined mask where the mask is non-zero
                combined_mask[mask_array > 0] = class_id

        # Save the combined mask as a palette-based image
        if combined_mask is not None:
            output_path = os.path.join(output_folder, image_file)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Create a palette-based image
            palette_image = Image.fromarray(combined_mask, mode="P")
            palette_image.putpalette(PALETTE)
            
            # Save the palette-based image as an 8-bit PNG
            palette_image.save(output_path, optimize=False)

# Example usage
input_folder = r"E:\Segmentations\Fetal Project\Dataset\png\masks"
output_folder = r"E:\Segmentations\Fetal Project\Dataset\png\masks\pe"
combine_masks(input_folder, output_folder)
