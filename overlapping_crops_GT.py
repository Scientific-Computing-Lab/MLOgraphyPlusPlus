import os
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process and crop images.")
    parser.add_argument('--gt_directory', type=str, required=True, help="Path to the directory containing GT crops (256x256)")
    parser.add_argument('--image_directory', type=str, required=True, help="Path to the directory containing GT annotations_overlayed_on_full_images")
    parser.add_argument('--output_directory', type=str, required=True, help="Path to the directory to save output overlapping crops of GT (256x256)")
    return parser.parse_args()

def extract_model_and_coordinates(filename):
    name, _ = os.path.splitext(filename)
    model, y, x = name.split('-')
    return model, int(y), int(x)

def crop_and_save_image(model, y, x, image_directory, output_directory):
    img_path = os.path.join(image_directory, f"{model}.png")
    img = Image.open(img_path)
    crop = img.crop((x, y, x + 256, y + 256))
    crop.save(os.path.join(output_directory, f"{model}-{y}-{x}.png"))

def main():
    args = parse_args()

    os.makedirs(args.output_directory, exist_ok=True)

    original_crops = [extract_model_and_coordinates(f) for f in os.listdir(args.gt_directory) if f.endswith('.png')]
    original_crops_set = set((model, y, x) for model, y, x in original_crops)

    final_crops = []

    for model, y, x in original_crops:
        has_left_neighbor = (model, y, x - 256) in original_crops_set
        has_right_neighbor = (model, y, x + 256) in original_crops_set
        has_top_neighbor = (model, y - 256, x) in original_crops_set
        has_bottom_neighbor = (model, y + 256, x) in original_crops_set

        if not (has_left_neighbor or has_right_neighbor or has_top_neighbor or has_bottom_neighbor):
            final_crops.append((model, y, x))
        else:
            if has_right_neighbor:
                final_crops.append((model, y, x + 128))
            if has_left_neighbor:
                final_crops.append((model, y, x - 128))
            if has_bottom_neighbor:
                final_crops.append((model, y + 128, x))
            if has_top_neighbor:
                final_crops.append((model, y - 128, x))
            final_crops.append((model, y, x))

    for model, y, x in final_crops:
        crop_and_save_image(model, y, x, args.image_directory, args.output_directory)

if __name__ == "__main__":
    main()
