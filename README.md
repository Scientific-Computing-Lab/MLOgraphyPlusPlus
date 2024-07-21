# Introduction

In quantitative metallographic image analysis, accurately identifying grain boundaries within texture-oriented images poses significant challenges due to the intricate nature of texture boundary edges. Traditional methods, starting from simple image processing techniques to deep learning-based approaches like semantic segmentation using sliding windows, often fail to sustain effectiveness, especially when dealing with texture boundaries. Texture perception is subjective, context-sensitive, and difficult even for humans to define. On the other end, generalized models like the Segment Anything Model (SAM) struggle with purely texture-based images and often fail at accurate segmentation without clear object boundaries.

In this research, we introduce a novel approach that flips the conventional methodology by employing partial labels while maintaining the complete context of the images. This strategy enhances the model’s ability to discern grain boundaries more effectively. Our method, namely MLOgraphy++, utilizes a U-Net architecture trained with partial labels to segment metallographic images, prioritizing continuous boundary detection over complete grain contours.

We embrace the Heyn intercept method, a classical technique for measuring average grain size, as a valid alternative to the pixel-accuracy common evaluation metric. This is because the distribution of grain sizes is more critical than exact pixel prediction, which is inherently challenging to label accurately. Our method demonstrates the suitability of the Heyn intercept method as an evaluation metric.

We compare our approach against the previous state-of-the-art method MLOgraphy, which used complete labels with partial context, on the Texture Boundary in Metallography comprehensive dataset (TBM dataset). Our results show a significant improvement in segmentation accuracy and reliability, with quantitative analysis against ground truth annotations confirming the robustness and effectiveness of our approach.


# Instructions

## Installation Requirements
  Ensure you have the following dependencies installed:
  ```sh
  pip install pandas numpy opencv-python pillow argparse
  ```

### Key Steps

1. **Process human-tagged GT images into 256x256 squares** and analyze them using the Hyen intercept method.
2. **Compare predictions from Mlography and Clemex models**, excluding overlapping sections with the GT used during training.
3. **Create 256x256 squares for consistency** and perform meta-statistical analysis to extract mean, median, and variance for each group.

### Key Findings

- **Mean and Median**: Both models show similar mean and median grain sizes compared to the GT.
- **Variance**: Melography exhibits higher variance in grain sizes, while Clemex predictions are closer to GT.
- **Statistical Comparison**: Melography's variance is 25% away from the GT, while Clemex is 16% away, indicating a 9% advantage for Clemex.


**Evaluation**
There are several scripts:
  - `grain_size.py`: Functions for calculating grain size.
  - `results.py`: Meta-statistical analysis and results presentation.
  - `crop_images_gt.py`: Functions for cropping images into 256x256 with GT consideration.
  - `crop_non_overlapping_crops.py`:  Functions for cropping non-overlapping sections with GT into 256x256 squares.


## Usage Instructions

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/Inbalc2/GetGrainSize.git

2. **Crop the GT images**:
   Run the script `crop_images_gt.py` in the following way:
   ```python
    python crop_images_gt.py --clemex_path <PATH_TO_CLEMEX_PREDICTIONS> --zones_path <PATH_TO_ZONES> --output_path <PATH_TO_OUTPUT_DIRECTORY> --mlography_path     
    <PATH_TO_MLOGRAPHY_PREDICTIONS> --gt_path <PATH_TO_GROUND_TRUTH_IMAGES>
   ```
3. **Process Non-overlapping Crops**:
   Run the script crop_non_overlapping_crops.py to crop non-overlapping sections into 256x256 squares:
   ```python
   python non_overlapping_crops.py --image_dir <PATH TO IMAGE DIRECTORY> --output_dir <PATH TO OUTPUT DIRECTORY> --zone_size <WIDTH> <HEIGHT> --gt_image_dir <PATH TO GROUND TRUTH 
   IMAGE DIRECTORY>
   ```
4.  **Calculate Grain Size**:
    Run the script grain_size.py to calculate grain size:
    ```python
    python grain_size.py --gt_crops_path <PATH TO GT CROPS> --mlography_crops_path <PATH TO MLOGRAPHY CROPS> --clemex_crops_path <PATH TO CLEMEX CROPS> -- 
    output_dir <PATH TO OUTPUT DIRECTORY>
    ```
5.  **Analyze Results**:
    Run the script results.py to perform meta-statistical analysis and present the results:
    ```python
    python results.py --df_path <PATH TO CSV FILE CONTAINING RESULTS>
    ```


## Expected Output
- **Cropped Images**: 256x256 cropped images saved in the specified output directory.
- **Grain Size Calculation** : CSV file with calculated grain sizes.
- **Statistical Analysis**: Printed statistics including mean, median, mode, standard deviation, variance, minimum, maximum, and sum of grain sizes for each model.


