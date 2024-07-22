import pandas as pd
import numpy as np
import os
import random
import re
from PIL import Image, ImageDraw
import pandas as pd
# Constants
CUTOFF = 2 * np.sqrt(2) + 0.001     # cutting distance for grouping boundary
L_CONST = 1.13                      # D = L_CONST * l_ave


def output(message):
    print(message)

import numpy as np
from PIL import Image
import random

CUTOFF = 10  # Define a suitable cutoff distance for your problem
L_CONST = 1  # Define a suitable L_CONST for your problem

def output(msg):
    print(msg)


# Constants
CUTOFF = 2 * np.sqrt(2) + 0.001     # cutting distance for grouping boundary
L_CONST = 1.13                      # D = L_CONST * l_ave


def output(message):
    print(message)

import numpy as np
from PIL import Image
import random

CUTOFF = 10  # Define a suitable cutoff distance for your problem
L_CONST = 1  # Define a suitable L_CONST for your problem

def output(msg):
    print(msg)


def grainsize(croppedlist, linenum,output_dir="/Users/inbal/Desktop/Metallography_2/results_hyen_256_before"):

    output("\n-- Measure Grain Size --")
    grain_sizes = []
    for f in croppedlist:
        im = Image.open(f)
        width, height = im.size

        output(f)

        # value of pixels
        im_pixels = np.array([[im.getpixel((i, j)) for j in range(height)] for i in range(width)])

        draw = ImageDraw.Draw(im)

        # draw line
        # upper left is (0, 0)
        sum_d = 0.0
        for l in range(linenum):

            start = np.zeros(2, dtype=int)
            end = np.zeros(2, dtype=int)
            linelength = 0.0

            while True:
                # gradient
                grad = np.tan(random.uniform(-np.pi/2, np.pi/2))

                # point
                x = random.randrange(width)
                y = random.randrange(height)

                # y = ax + b
                b = y -grad * x
                if b < 0:
                    start[0] = -b / grad
                    start[1] = 0
                elif 0 <= b <= height:
                    start[0] = 0
                    start[1] = b
                elif height < b:
                    start[0] = (height - b) / grad
                    start[1] = height

                y_width = grad * width + b
                if y_width < 0:
                    end[0] = - b / grad
                    end[1] = 0
                elif 0 <= y_width <= height:
                    end[0] = width
                    end[1] = y_width
                elif height < y_width:
                    end[0] = (height - b) / grad
                    end[1] = height

                # check length (short line is not good)
                length2 = (start[0] - end[0])**2 + (start[1] - end[1])**2
                if length2 > width**2 and length2 > height**2:
                    #print(x, y, grad, start, end)
                    linelength = np.sqrt(length2)
                    break

            # find cross points
            # draw line
            im2 = Image.new('RGB', (width, height))
            draw2 = ImageDraw.Draw(im2)
            draw2.line(((start[0], start[1]), (end[0], end[1])), fill=(255, 0, 0))
            im2_pixels = np.array([[im2.getpixel((i, j)) for j in range(height)] for i in range(width)])
            for j in range(height):
                for i in range(width):
                    r, g, b = im2_pixels[i][j] + im_pixels[i][j]
                    im2.putpixel((i, j), (r, g, b))
            #im2.save(f+str(l)+".tif")

            # pick up pixels on grain boundary
            pixelsonGB = []
            im2_pixels = np.array([[im2.getpixel((i, j)) for j in range(height)] for i in range(width)])
            for j in range(height):
                for i in range(width):
                    r, g, b = im2_pixels[i][j]
                    if r == 255 and g == 0 and b == 0:
                        pixelsonGB.append((i, j))
            # print(pixelsonGB)

            # grouping neighbor pixels (distance <= CUTOFF)
            gbs = []
            p_coord = []
            i = 0
            for coord in pixelsonGB:
                if i == 0:
                    gbs.append([coord])

                else:
                    distance = np.sqrt((coord[0] - p_coord[0])**2 + (coord[1] - p_coord[1])**2)
                    if distance <= CUTOFF:
                        gbs[-1].append(coord)
                    else:
                        gbs.append([coord])

                p_coord = [coord[0], coord[1]]
                i += 1
            #print(gbs)

            # calc grain size
            d_grain = L_CONST * linelength / len(gbs)
            grain_sizes.append(d_grain)
            sum_d += d_grain
            output(" D_" + str(l) + " = " + str(d_grain) + " [px]")

            # draw line and G.B.
            draw.line(((start[0], start[1]), (end[0], end[1])), fill=(255, 0, 0))
            #draw.text((start[0] + 0.3 * (end[0] - start[0]), start[1] + 0.3 * (end[1] - start[1])), str(l), fill=(255, 0, 0))
            for gb in gbs:
                upperleft = [width, height]
                lowerright = [0, 0]
                for coord in gb:
                    if coord[0] < upperleft[0]:
                        upperleft[0] = coord[0]
                    if coord[1] < upperleft[1]:
                        upperleft[1] = coord[1]
                    if lowerright[0] < coord[0]:
                        lowerright[0] = coord[0]
                    if lowerright[1] < coord[1]:
                        lowerright[1] = coord[1]

                draw.ellipse((upperleft[0]-1, upperleft[1]-1, lowerright[0]+1, lowerright[1]+1), outline=(0, 0, 255))

        #im.save(f)
        ave_d = sum_d / linenum
        output(" D_ave = " + str(ave_d) + " [px]\n")
        if output_dir:
            base_filename = os.path.basename(f)
            output_image_path = os.path.join(output_dir, f"hyen_{base_filename}")
            im.save(output_image_path)
        return grain_sizes


def calculate_statistics(data):
    mean_grain_size = round(np.mean(data), 2)
    median_grain_size = round(np.median(data), 2)
    std_grain_size = round(np.std(data), 2)
    
    return {
        'Mean Grain Size': mean_grain_size,
        'Median Grain Size': median_grain_size,
        'Std Grain Size': std_grain_size
    }

def analyze_images(image_dirs):
    model_zone_data = []

    for folder, model in image_dirs:
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        target_filenames = [f for f in os.listdir(folder) if f.endswith((".tif", ".jpg", ".png"))]
        if not target_filenames:
            print(f"No TIFF, JPG, or PNG files found in {folder}")
            continue

        model_output_dir = os.path.join("/Users/inbal/Desktop/Metallography_2/results_hyen_256_", model.replace(" ", "_"))
        os.makedirs(model_output_dir, exist_ok=True)

        for target_filename in target_filenames:
            print(f"Processing image: {target_filename}")

            image_path = os.path.join(folder, target_filename)
            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                continue

            try:
                grain_sizes = grainsize([image_path], 10, model_output_dir)
                stats = calculate_statistics(grain_sizes)

                image_data = {
                    'Model': model,
                    'Filename': target_filename,
                    **stats
                }
                model_zone_data.append(image_data)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    return model_zone_data


def main(args):
    sub_model_folders = [
        (args.gt_crops_path, "GT"),
        ("/path/to/mlography_crops_256_no_gt", "MLOgraphy_256_Predictions"),
        ("/path/to/mlography++_crops_256_no_gt", "MLOgraphy++_256_Predictions"),
    ]

    output_base_dir = "/path/to/output_directory"
    output(f"Output directory: {output_base_dir}")
    
    # Process images and store results
    sub_model_data = analyze_images(sub_model_folders, output_base_dir)

    # Save results to CSV files
    sub_model_df = pd.DataFrame(sub_model_data)
    sub_model_csv_path = os.path.join(output_base_dir, "256_crops_results_before.csv")
    sub_model_df.to_csv(sub_model_csv_path, index=False)

    output(f"Results saved to: {sub_model_csv_path}")

if __name__ == "__main__":
    parser = ...
    main(parser.parse_args())
