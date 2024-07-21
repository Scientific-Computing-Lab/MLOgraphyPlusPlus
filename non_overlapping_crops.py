import cv2
import os
from collections import defaultdict

def extract_info_from_filename(filename):
    try:
        parts = filename.split('-')
        modelname = parts[0]
        x = int(parts[1])
        y = int(parts[2])
        dx = int(parts[3])
        dy = int(parts[4].split('.')[0])  
        return modelname, x, y, dx, dy
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return None, None, None, None, None

def parse_gt_files(gt_image_dir):
    gt_zones = defaultdict(set)
    gt_files = [f for f in os.listdir(gt_image_dir) if f.endswith('.png')]
    for f in gt_files:
        modelname, x, y, dx, dy = extract_info_from_filename(f)
        if None not in (modelname, x, y, dx, dy):
            gt_zones[modelname].add((x, y))
    return gt_zones

def x_y_in_gt(x, y, delta, gt_zones, modelname):
    for (x_gt, y_gt) in gt_zones[modelname]:
        if (x <= x_gt <= x + delta) and (y <= y_gt <= y + delta):
            return True
    return False

def crop_images(image_dir, output_dir, zone_size=(256, 256), gt_image_dir='/Users/inbal/Desktop/Metallography_2/MLography/Segmentation/unet/data/squares_128/train/inv_label'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parse ground truth files
    gt_zones = parse_gt_files(gt_image_dir)

    # List all files in the directory
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

        for y in range(0, img_height, zone_size[1]):
            for x in range(0, img_width, zone_size[0]):
                if crop_counts[modelname] >= 4:
                    break  # Stop after getting 4 crops per model

                if x_y_in_gt(x, y, zone_size[0], gt_zones, modelname):
                    print(f"Skipping zone {modelname}-{x}-{y} as it overlaps with GT zones.")
                    continue  # Skip this zone if it overlaps with any GT zones

                crop_img = img[y:y + zone_size[1], x:x + zone_size[0]]
                if crop_img.shape[0] != zone_size[1] or crop_img.shape[1] != zone_size[0]:
                    continue  # Skip incomplete zones

                output_filename = f"{modelname}-{x}-{y}.png"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, crop_img)
                crop_counts[modelname] += 1
                print(f"Saved {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Crop images while avoiding specified ground truth zones.")
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images to be cropped.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save cropped images.')
    parser.add_argument('--zone_size', type=int, nargs=2, default=(256, 256), help='Size of the cropping zone (width, height).')
    parser.add_argument('--gt_image_dir', type=str, help='Directory containing ground truth images.')

    args = parser.parse_args()

    crop_images(args.image_dir, args.output_dir, zone_size=tuple(args.zone_size), gt_image_dir=args.gt_image_dir)

if __name__ == "__main__":
    main()
