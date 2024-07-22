# Introduction
In quantitative metallographic image analysis, accurately identifying grain boundaries within texture-oriented images poses significant challenges due to the intricate nature of texture boundary edges. Traditional methods, starting from simple image processing techniques to deep learning-based approaches like semantic segmentation using sliding windows, often fail to sustain effectiveness, especially when dealing with texture boundaries. Texture perception is subjective, context-sensitive, and difficult even for humans to define. On the other end, generalized models like the Segment Anything Model (SAM) struggle with purely texture-based images and often fail at accurate segmentation without clear object boundaries.

In this research, we introduce a novel approach that flips the conventional methodology by employing partial labels while maintaining the complete context of the images. This strategy enhances the model’s ability to discern grain boundaries more effectively. Our method, namely MLOgraphy++, utilizes a U-Net architecture trained with partial labels to segment metallographic images, prioritizing continuous boundary detection over complete grain contours.

We embrace the Heyn intercept method, a classical technique for measuring average grain size, as a valid alternative to the pixel-accuracy common evaluation metric. This is because the distribution of grain sizes is more critical than exact pixel prediction, which is inherently challenging to label accurately. Our method demonstrates the suitability of the Heyn intercept method as an evaluation metric.

We compare our approach against the previous state-of-the-art method MLOgraphy, which used complete labels with partial context, on the Texture Boundary in Metallography comprehensive dataset (TBM dataset). Our results show a significant improvement in segmentation accuracy and reliability, with quantitative analysis against ground truth annotations confirming the robustness and effectiveness of our approach.

# Evaluation
In this GitHub repository, we provide the evaluation code to reproduce our results. We convert GT annotations and models' predictions into 256x256 pixel images by combining 4 adjacent 128x128 labels for statistical analysis, averaging 10 times per image. We perform analysis both with and without overlapping sections with GT used during training to ensure comprehensive evaluation. We use Guo-Hall thinning and the Heyn intercept method to extract statistical results, including variance and mean of grain sizes, comparing MLOgraphy and MLOGRAPHY++.
   
# Instructions

## Installation Requirements
  Ensure you have the following dependencies installed:
  ```sh
  pip install pandas numpy opencv-python pillow argparse
  ```

**Scripts**
There are several scripts:
  - `grain_size.py`: Functions for calculating grain size.
  - `results.py`: Meta-statistical analysis and results presentation.
  - `crop_images_gt.py`: Functions for cropping images into 256x256 with GT consideration.
  - `crop_non_overlapping_crops.py`:  Functions for cropping non-overlapping sections with GT into 256x256 squares.


## Usage Instructions

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/Inbalc2/MLOgraphy-.git

2. **Crop the GT images**:
   Run the script `crop_images_gt.py`  to crop the ground truth images into 256x256 squares:
   ```python
   python crop_images_gt.py --MLOGRAPHY++_path <PATH_TO_MLOGRAPHY++_PREDICTIONS> --zones_path <PATH_TO_ZONES> --output_path <PATH_TO_OUTPUT_DIRECTORY> --    
   mlography_path <PATH_TO_MLOGRAPHY_PREDICTIONS> --gt_path <PATH_TO_GROUND_TRUTH_IMAGES>
   ```
3. **Generate Non-overlapping Crops of Model Predictions:**:
   Run the script crop_non_overlapping_crops.py to crop non-overlapping sections of model predictions into 256x256 squares, ensuring they do not overlap with GT:
   ```python
   python crop_non_overlapping_crops.py --image_dir <PATH_TO_IMAGE_DIRECTORY> --output_dir <PATH_TO_OUTPUT_DIRECTORY> --zone_size <WIDTH> <HEIGHT> --    
   gt_image_dir <PATH_TO_GROUND_TRUTH_IMAGE_DIRECTORY>
   IMAGE DIRECTORY>
   ```
4.  **Calculate Grain Size Using the Heyn Intercept Method:**:
    Run the script grain_size.py for Heyn intercept method evaluation:
    ```python
    python grain_size.py --gt_crops_path <PATH_TO_GT_CROPS> --mlography_crops_path <PATH_TO_MLOGRAPHY_CROPS> --MLOGRAPHY++_crops_path 
    <PATH_TO_MLOGRAPHY++_CROPS> --output_dir <PATH_TO_OUTPUT_DIRECTORY>

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

## Data
The data that was used in the paper is:

**MLOgraphy**
- **Inference Data:**
  - Images: `/Datasets/MLOgraphy/Inference/without_impurities/`
  - Full Predictions: `/Datasets/MLOgraphy/Inference/mlography_full_predictions/`
- **Evaluation - 256x256 crops with the Heyn intercept method:**
   -  crops with no GT: `/Datasets/MLOgraphy++/Evaluation/squares_256_no_GT/`
  -   crops with ground truth: `/Datasets/MLOgraphy++/Evaluation/squares_256_with_GT/`


