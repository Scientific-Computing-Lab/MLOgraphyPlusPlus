# MLOgraphy++

Analysis of metallography images comparing Melography and Clemex predictions to Ground Truth.

## Overview

This project involves processing and analyzing metallography images, comparing predictions from Melography and Clemex models to Ground Truth (GT).

### Key Steps

1. **Process human-tagged GT images into 256x256 squares** and analyze them using the Heyn intercept method.
2. **Compare predictions from Melography and Clemex models**, excluding overlapping sections with the GT used during training.
3. **Create 256x256 squares for consistency** and perform meta-statistical analysis to extract mean, median, and variance for each group.

### Key Findings

- **Mean and Median**: Both models show similar mean and median grain sizes compared to the GT.
- **Variance**: Melography exhibits higher variance in grain sizes, while Clemex predictions are closer to GT.
- **Statistical Comparison**: Melography's variance is 25% away from the GT, while Clemex is 16% away, indicating a 9% advantage for Clemex.

## Project Structure

- **Scripts**:
  - `grain_size.py`: Functions for calculating grain size.
  - `results.py`: Meta-statistical analysis and results presentation.
  - `crop_images_with_gt.py`: Functions for cropping images with GT consideration.
  - `crop_non_overlapping_crops.py`: Functions for cropping non-overlapping sections.
- **Data Directories**:
  - `data/`: Input images.
  - `results/`: Processed images and statistical results.
## Usage Instructions

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/Inbalc2/GetGrainSize.git

2. **Crop the GT images**:
   Run the script `crop_images_gt.py` in the following way:
   ```python
    python crop_images_gt.py --clemex_path <PATH TO CLEMEX PREDICITON> --zones_path <PATH TO 128 CROPS WITH THE NAME>
   ```

