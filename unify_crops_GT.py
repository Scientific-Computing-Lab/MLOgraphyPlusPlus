import os
import cv2
from collections import defaultdict
from PIL import Image
import numpy as np
from cv_algorithms import guo_hall
import argparse

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def unify_crops(input_dir, output_dir, crop_size=256, unified_size=128):
    create_directory(output_dir)  # Ensure the output directory is created
    files = sorted(os.listdir(input_dir))
    for i in range(0, len(files), 4):
        unified_image = Image.new('RGB', (crop_size, crop_size), (0, 0, 0))
        for j in range(4):
            crop_image = Image.open(os.path.join(input_dir, files[i + j]))
            x = (j % 2) * unified_size
            y = (j // 2) * unified_size
            unified_image.paste(crop_image, (x, y))

        new_filename = '-'.join(files[i].split('-')[:3]) + '.png'
        unified_image.save(os.path.join(output_dir, new_filename))

def apply_guo_hall_thinning(image):
    image = np.array(image)
    binary_image = cv2.bitwise_not(image)
    thinned_image = guo_hall(binary_image, inplace=False)
    thinned_image = cv2.bitwise_not(thinned_image)
    return thinned_image

def convert_to_color(thinned_image, original_image):
    # Ensure both images are in the correct format
    thinned_image_rgb = cv2.cvtColor(thinned_image, cv2.COLOR_GRAY2RGB)
    combined_image = np.where(thinned_image_rgb == 0, original_image, thinned_image_rgb)
    return combined_image


# Process each image in the input directory and save the thinned images
def process_and_thin_images(input_dir, output_dir):
    create_directory(output_dir)  # Ensure the output directory is created
    for image_name in os.listdir(input_dir):
        if image_name.endswith(('.png', '.jpg', '.tif')):
            input_path = os.path.join(input_dir, image_name)
            print(f"Processing image: {input_path}")
            original_image = Image.open(input_path).convert('RGB')
            grayscale_image = original_image.convert('L')  # Convert to grayscale

            # Apply Guo-Hall thinning
            thinned_image = apply_guo_hall_thinning(grayscale_image)
            original_image_np = np.array(original_image)
            thinned_image_color = convert_to_color(thinned_image, original_image_np)
            final_image = Image.fromarray(thinned_image_color.astype(np.uint8))
            output_path = os.path.join(output_dir, image_name)
            final_image.save(output_path)
            print(f"Saved processed image: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process and unify image labels crops for metallography analysis.")
    parser.add_argument('--gt_path', type=str, required=True, 
                        help="Path to the directory containing the GT labels(128x128).")
    parser.add_argument('--gt_output_path', type=str, required=True, 
                        help="Output path for the unified ground truth crops(256x256).")

    args = parser.parse_args()

    # Unify the 128x128 GT crops into 256x256 images
    create_directory(args.gt_output_path)
    unify_crops(args.gt_path, args.gt_output_path)
     # Apply Guo-Hall thinning
    process_and_thin_images(args.gt_output_path, args.gt_output_path)

if __name__ == "__main__":
    main()






