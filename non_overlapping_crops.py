import os
import cv2
from collections import defaultdict
import argparse

# Constants
ZONE_SIZE = (256, 256)
STEP_Y = ZONE_SIZE[1] // 2
STEP_X = ZONE_SIZE[0] // 2

def extract_info_from_filename(filename):
    "Extract model name and coordinates from the filename."
    try:
        parts = filename.split('-')
        modelname = parts[0]
        y = int(parts[1])
        x = int(parts[2])
        dy = int(parts[3])
        dx = int(parts[4].split('.')[0])
        return modelname, y, x, dy, dx
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return None, None, None, None, None

def parse_gt_files(gt_image_dir):
    "Parse ground truth files to extract zones."
    gt_zones = defaultdict(set)
    gt_files = [f for f in os.listdir(gt_image_dir) if f.endswith('.png')]
    for f in gt_files:
        modelname, y, x, dy, dx = extract_info_from_filename(f)
        if None not in (modelname, y, x, dy, dx):
            gt_zones[modelname].add((y, x))
    return gt_zones

def get_box(x, y, delta):
    "Return a bounding box as (x_min, y_min, x_max, y_max)."
    return x, y, x + delta, y + delta

def get_intersection(box1, box2):
    "Calculate the intersection area of two bounding boxes."
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    xa = max(x1_min, x2_min)
    ya = max(y1_min, y2_min)
    xb = min(x1_max, x2_max)
    yb = min(y1_max, y2_max)

    w = max(0, xb - xa)
    h = max(0, yb - ya)
    return w * h

def x_y_in_gt(x, y, delta, gt_zones, modelname):
    "Check if the given coordinates overlap with ground truth zones."
    box1 = get_box(x, y, delta)
    for (y_gt, x_gt) in gt_zones[modelname]:
        box2 = get_box(x_gt, y_gt, delta)
        if get_intersection(box1, box2) > 0:
            return True
    return False

def crop_images(image_dir, output_dir, zone_size=ZONE_SIZE, gt_image_dir='/path/to/gt'):
    "Crop images into smaller zones with 50% overlap, avoiding specified ground truth zones."
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gt_zones = parse_gt_files(gt_image_dir)
    files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    print(f"Total files found: {len(files)}")

    crop_counts = defaultdict(int)
    for filename in files:
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        img_height, img_width = img.shape[:2]
        modelname = filename.split('.')[0]

        for y in range(0, img_height - STEP_Y, STEP_Y):
            for x in range(0, img_width - STEP_X, STEP_X):
                if x_y_in_gt(x, y, zone_size[0], gt_zones, modelname):
                    print(f"Skipping zone {modelname}-{y}-{x} as it overlaps with GT zones.")
                    continue

                crop_img = img[y:y + zone_size[1], x:x + zone_size[0]]
                if crop_img.shape[0] != zone_size[1] or crop_img.shape[1] != zone_size[0]:
                    continue

                output_filename = f"{modelname}-{y}-{x}.png"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, crop_img)
                crop_counts[modelname] += 1
                print(f"Saved {output_path}")

    print(f"Crop counts per model: {dict(crop_counts)}")



def parse_args():
    parser = argparse.ArgumentParser(description="Process image crops and save results.")
    
    parser.add_argument('--zone_size', type=int, nargs=2, default=(256, 256), 
                        help='Size of the cropping zone (width height)')
    parser.add_argument('--gt_image_dir', type=str, required=True, 
                        help='Ground truth crops(128x128) directory path')
    parser.add_argument('--image_dir1', type=str, required=True, 
                        help='Directory for MLOgraphy++ full predictions')
    parser.add_argument('--output_dir1', type=str, required=True, 
                        help='Output directory for MLOgraphy++ non verlapping crops(256x256) with GT')
    parser.add_argument('--image_dir2', type=str, required=True, 
                        help='Directory for MLOgraphy full predictions')
    parser.add_argument('--output_dir2', type=str, required=True, 
                        help='Output directory for or MLOgraphy non verlapping crops(256x256) with GT')
    
    return parser.parse_args()



def main():
    args = parse_args()
    
    dirs = [
        {"image_dir": args.image_dir1, "output_dir": args.output_dir1},
        {"image_dir": args.image_dir2, "output_dir": args.output_dir2}
    ]

    for dir_info in dirs:
        crop_images(
            image_dir=dir_info["image_dir"],
            output_dir=dir_info["output_dir"],
            zone_size=tuple(args.zone_size),
            gt_image_dir=args.gt_image_dir
        )

if __name__ == "__main__":
    main()