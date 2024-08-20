import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import concurrent.futures
from multiprocessing import cpu_count
import argparse


# Define constants    
MARGIN = 2  # Margin for excluding borders when drawing horizontal lines

# Function to output messages
def output(msg):
    print(msg)

# Function to measure grain size
def grainsize(croppedlist, linenum, output_dir=None):
    grain_sizes = []
    output("\n-- Measure Grain Size --")

    for f in croppedlist:
        im = Image.open(f)
        width, height = im.size
        im_pixels = np.array(im)
        draw = ImageDraw.Draw(im)

        # Calculate interval excluding margins
        interval = (height - 2 * MARGIN) // (linenum + 1)
        sum_d = 0.0

        # Draw horizontal lines and find grain boundaries
        for l in range(linenum):
            y = MARGIN + interval * (l + 1)
            start, end = (0, y), (width, y)

            # Create an image to find cross points
            im2 = Image.new('RGB', (width, height))
            draw2 = ImageDraw.Draw(im2)
            draw2.line((start, end), fill=(255, 0, 0))
            im2_pixels = np.array(im2)
            im2_pixels = im2_pixels + im_pixels

            # Identify pixels on grain boundary
            pixelsonGB = [
                (i, j) for j in range(height) for i in range(width)
                if (im2_pixels[j, i] == [255, 0, 0]).all()
            ]

            # Calculate the distance between each pair of pixels on the grain boundary
            distances_between_pixels = [
                np.sqrt((coord2[0] - coord1[0]) ** 2 + (coord2[1] - coord1[1]) ** 2)
                for coord1, coord2 in zip(pixelsonGB[:-1], pixelsonGB[1:])
            ]

            if distances_between_pixels:
                d_grain = np.mean(distances_between_pixels)
                sum_d += d_grain
                output(f"D_horiz_{l} = {d_grain} [px]")
                grain_sizes.append(d_grain)

            # Draw line and grain boundaries
            draw.line((start, end), fill=(255, 0, 0))
            for coord in pixelsonGB:
                draw.ellipse((coord[0]-1, coord[1]-1, coord[0]+1, coord[1]+1), outline=(0, 0, 255))

        ave_d = sum_d / linenum
        output(f"D_ave = {ave_d} [px]\n")

        if output_dir:
            output_image_path = os.path.join(output_dir, f"heyn_{os.path.basename(f)}")
            im.save(output_image_path)
    
    return grain_sizes

# Function to calculate statistics for grain sizes
def calculate_statistics(grain_sizes):
    if not grain_sizes:
        return {}

    mean = np.mean(grain_sizes)
    variance = np.var(grain_sizes)

    return {
        'Mean': mean,
        'Variance': variance
    }

# Function to process images and measure grain sizes
def process_image(folder, model, target_filename, model_output_dir):
    image_path = os.path.join(folder, target_filename)
    if not os.path.exists(image_path):
        output(f"File not found: {image_path}")
        return None

    output(f"Processing image: {target_filename}")
    try:
        degem = extract_degem(target_filename)
        grain_sizes = grainsize([image_path], 20, model_output_dir)

        return {
            'Model': model,
            'Degem': degem,
            'Grain Sizes': grain_sizes
        }
    except Exception as e:
        output(f"Error processing {image_path}: {e}")
        return None

# Function to extract degem (identifier) from the filename
def extract_degem(filename):
    hyphen_pos = filename.find('-')
    if hyphen_pos >= 2:
        return filename[hyphen_pos - 2:hyphen_pos]
    else:
        output(f"Could not extract degem from {filename}")
        return "unknown"

# Function to analyze images from multiple directories
def analyze_images(image_dirs):
    model_degem_grain_sizes = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = []
        for folder, model in image_dirs:
            if not os.path.exists(folder):
                output(f"Folder not found: {folder}")
                continue

            target_filenames = [f for f in os.listdir(folder) if f.endswith((".tif", ".jpg", ".png"))]

            if not target_filenames:
                output(f"No TIFF, JPG, or PNG files found in {folder}")
                continue

            model_output_dir = os.path.join("results", model.replace(" ", "_"))
            os.makedirs(model_output_dir, exist_ok=True)

            for target_filename in target_filenames:
                futures.append(executor.submit(process_image, folder, model, target_filename, model_output_dir))

        for future in concurrent.futures.as_completed(futures):
            image_data = future.result()
            if image_data:
                model, degem, grain_sizes = image_data['Model'], image_data['Degem'], image_data['Grain Sizes']
                model_degem_grain_sizes.extend([[model, degem, size] for size in grain_sizes])

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(model_degem_grain_sizes, columns=['Model', 'Degem', 'Grain Size'])
    df.sort_values(by=['Model', 'Degem', 'Grain Size'], inplace=True)

    csv_path = os.path.join("results", "all_models_grain_sizes.csv")
    df.to_csv(csv_path, index=False)
    output(f"Saved all models' data to {csv_path}")

    return df

# Main function to set directories and start processing
def main():
    parser = argparse.ArgumentParser(description="Analyze images for grain size calculation.")
    parser.add_argument("--gt_path", required=True, help="Path to 256x256 crops of Ground Truth images")
    parser.add_argument("--mlography_path", required=True, help="Path to 256x256 crops of MLography predictions not overlapping with Ground Truth")
    parser.add_argument("--mlography_plus_plus_path", required=True, help="Path to 256x256 crops of MLOgraphy++ predictions not overlapping with Ground Truth")

    args = parser.parse_args()

    sub_model_folders = [
        (args.gt_path, "Ground_Truth"),
        (args.mlography_path, "MLOgraphy_Predictions"),
        (args.mlography_plus_plus_path, "MLOgraphy++_Predictions"),
    ]

    output("Starting analysis...")
    model_degem_grain_sizes = analyze_images(sub_model_folders)
    output("Analysis complete.")

if __name__ == "__main__":
    main()
