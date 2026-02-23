import os
import nibabel as nib
import numpy as np
from PIL import Image, ImageOps

def normalize_image(image_array):
    # Clip pixel values to Hounsfield Unit (HU) range
    min_hu = -100
    max_hu = 200
    image_array = np.clip(image_array, min_hu, max_hu)

    # Normalize pixel values to 0-255 (uint8)
    normalized_slice = (255 * (image_array - min_hu) / (max_hu - min_hu)).astype('uint8')
    return normalized_slice
def normalize_image2(image_array):
    # Find the minimum and maximum HU values in the entire volume
    min_hu = np.nanmin(image_array)  # Handle NaN values
    max_hu = np.nanmax(image_array)  # Handle NaN values

    # Check for division by zero cases
    if min_hu == max_hu:
        normalized_slice = np.zeros_like(image_array, dtype=np.uint8)
    else:
        # Normalize pixel values to 0-255 (uint8)
        normalized_slice = (255 * (image_array - min_hu) / (max_hu - min_hu)).astype(np.uint8)

    return normalized_slice



def intensity_scale_each_slice(raw_each_np_slice_local, min_intensity_local=0, max_intensity_local=1700):
    # Clip pixel values to the specified intensity range
    cliped_each_np_slice_local = np.clip(raw_each_np_slice_local, min_intensity_local, max_intensity_local)

    # Normalize pixel values to 0-255 (uint8)
    normalized_each_np_slice_local = ((cliped_each_np_slice_local - min_intensity_local) / (max_intensity_local - min_intensity_local) * 255).astype(np.uint8)

    # Create a PIL Image from the normalized NumPy array
    scaled_pil_formated_slice_local = Image.fromarray(normalized_each_np_slice_local)

    # Apply auto-contrast stretching to enhance visibility
    contrast_stretched_scaled_pil_formated_slice_local = ImageOps.autocontrast(scaled_pil_formated_slice_local)

    return contrast_stretched_scaled_pil_formated_slice_local

def nifti_to_png(input_nii_file, output_folder,name):
    # Read the NIfTI file using nibabel
    nii_image = nib.load(input_nii_file)
    image_array = nii_image.get_fdata()
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through the slices and save them as PNG images
    for idx, slice_array in enumerate(image_array.transpose(2, 0, 1)):  # Transpose to get slices along the third dimension
        # Apply intensity scaling to enhance contrast and visibility
        scaled_slice = normalize_image2(slice_array)
        # Save the image
        # output_path = os.path.join(output_folder, f"slice_{idx:03d}.png")
        # scaled_slice.save(output_path)

        image = Image.fromarray(scaled_slice)
        # angle = 270  # You can change this angle as needed
        # rotated_image = image.rotate(angle)

        flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        angle = 270  # You can change this angle as needed
        rotated_image = flipped_image.rotate(angle)
        # Save the image
        output_path = os.path.join(output_folder, f"{name}_{idx}.png")
        rotated_image.save(output_path)


dir_path = "path"
dest_path = "path"
for root,_,files in os.walk(dir_path):
    if files:
        for file in files:

            #  print(file)
            split_name = file[:-7]
            # print(split_name)
            input_file = os.path.join(root,file)
            nifti_to_png(input_file,dest_path,split_name)
