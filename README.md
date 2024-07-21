# Introduction

In quantitative metallographic image analysis, accurately identifying grain boundaries within texture-oriented images poses significant challenges due to the intricate nature of texture boundary edges. Traditional methods, starting from simple image processing techniques to deep learning-based approaches like semantic segmentation using sliding windows, often fail to sustain effectiveness, especially when dealing with texture boundaries. Texture perception is subjective, context-sensitive, and difficult even for humans to define. On the other end, generalized models like the Segment Anything Model (SAM) struggle with purely texture-based images and often fail at accurate segmentation without clear object boundaries.

In this research, we introduce a novel approach that flips the conventional methodology by employing partial labels while maintaining the complete context of the images. This strategy enhances the model’s ability to discern grain boundaries more effectively. Our method, namely MLOgraphy++, utilizes a U-Net architecture trained with partial labels to segment metallographic images, prioritizing continuous boundary detection over complete grain contours.

We embrace the Heyn intercept method, a classical technique for measuring average grain size, as a valid alternative to the pixel-accuracy common evaluation metric. This is because the distribution of grain sizes is more critical than exact pixel prediction, which is inherently challenging to label accurately. Our method demonstrates the suitability of the Heyn intercept method as an evaluation metric.

We compare our approach against the previous state-of-the-art method MLOgraphy, which used complete labels with partial context, on the Texture Boundary in Metallography comprehensive dataset (TBM dataset). Our results show a significant improvement in segmentation accuracy and reliability, with quantitative analysis against ground truth annotations confirming the robustness and effectiveness of our approach.

## Overview

<div align="center">
  <img src="MLOgraphy++/Datasets/MLOgraphy/image/10-0-768-0-0.png" alt="Sample Image">
</div>

This project involves processing and analyzing metallography images, comparing predictions from Melography and Clemex models to Ground Truth (GT).
### Key Steps
sdgfdfg
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

