from PIL import Image
import numpy as np

# Load the image
image_path = '/home/inbal/Desktop/AutoSam2/AutoSAM/TBM/TestDataset/masks/11-0-0.png'
image = Image.open(image_path)

# Convert the image to a NumPy array
image_array = np.array(image)

# Check unique values in the image
unique_values, counts = np.unique(image_array, return_counts=True)

# Print unique values and their counts
for value, count in zip(unique_values, counts):
    print(f"Value: {value}, Count: {count}")