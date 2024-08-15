

# MLOgraphy++
This project compares grain boundary detection using MLOgraphy and the enhanced model, MLOgraphy++, on the TBM dataset. MLOgraphy trains a U-Net on 128x128 cropped sub-images with complete annotations but limited context. In contrast, MLOgraphy++ uses 256x256 partial annotations (combining four adjacent 128x128 labels) over the full image, capturing broader context for edge segmentation. We assessed performance using a variation of the Heyn intercept method, analyzing 256x256 image crops (with and without 50% overlap) via Guo-Hall thinning. The results show MLOgraphy++ closely aligns with the GT, similar to MLOgraphy, but more efficiently and without post-processing.


<div align="center">
  <img src="/Datasets/GT/Evaluation Crops (256x256)/heyn_10-0-768.png" alt="Sample Image">
</div>



## Key Steps
1. **Unify human-tagged 128x128 GT crops into 256x256 images**, create additional 256x256 crops with 50% overlap.
2.  **Crop non-overlapping 256x256 crops from MLOgraphy and MLOgraphy++ predictions**, with some sections having 50% overlap.
2. **Use the crops from all models** to compare their grain sizes(Ground Truth, MLOgraphy, and MLOgraphy++) using a variation of the Heyn intercept method.


## Scripts
   - **unify_crops_GT.py**: Unifing 128x128 GT crops into 256x256 images .
   - **non_overlapping_crops.py**: Cropping non-overlapping 256x256 crops from MLOgraphy and MLOgraphy++ predictions. 
   - **overlapping_crops_GT.py**: Cropping overlapping 256x256 GT crops having 50% overlap.
   - **grain_size.py**: Functions for calculating grain size from images using a variation of the Heyn intercept method. It processes images, detects grain boundaries, calculates grain sizes, and optionally saves the processed images.


## Usage Instructions
1. **Unify the GT crops**:
   Run the script `unify_crops_GT.py` in the following way:
   ```python
    python unify_crops_GT.py --gt_path <path to GT_128_LABELS> --gt_output_path <path to GT_256_CROPS>
   ```
  
2. **Cropping non-overlapping 256x256 crops**:
   Run the script `non_overlapping_crops.py` in the following way:
   ```python
   python non_overlapping_crops.py --zone_size 256 256 --gt_image_dir <path to GT crops(128x128)> --image_dir1 <path to MLOgraphy++ full predictions> --output_dir1 <path to MLOgraphy++ non-overlapping 
   crops(256x256) with GT> --image_dir2 <path to MLOgraphy full predictions> --output_dir2 <path to MLOgraphy non-overlapping crops(256x256) with GT>
   ```
3. **Cropping overlapping 256x256 GT crops having 50% overlap**:
   Run the script `overlapping_crops_GT.py` in the following way:
   ```python
   python overlapping_crops_GT.py --gt_directory <path to GT crops(256x256)> --image_directory <path to GT annotations_overlayed_on_full_images> --output_directory <path to output overlapping crops of GT (256x256)>
   ```
4. **Calculating grain sizes**:
   Run the script `grain_size.py` in the following way:
   ```python
   python grain_size.py --gt_path <PATH_TO_256X256_GT_CROPS> --mlography_path <PATH_TO_256X256_MLOGRAPHY_CROPS> --mlography_plus_plus_path <PATH_TO_256X256_MLOGRAPHY_PP_CROPS>
   ```

## Data
  The data that was used in the paper is from the [TBM Dataset](https://zenodo.org/records/8386997). 
  The specific data used for the evaluation can be found in the /Datasets/ directory.

## Results
  The results, including the grain sizes measured for all models (Ground Truth, MLOgraphy, and MLOgraphy++), are saved in the Results_grain_sizes.csv file.


  

  
  

 

 
 

