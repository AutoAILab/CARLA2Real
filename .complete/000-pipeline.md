# Pipeline

## Description
Using the references, construct a pipeline to generate realistic CARLA data.

## I/O
Input directory: `/home/df/data/datasets/SkyScenes/proof_of_concept`
Output directory: `/home/df/data/datasets/SkyScenes/realistic`

Results in output directory should maintain the same directory structure as the input directory.

## References
- [SkyScenes](/home/df/data/jflinte/SkyScenes)
- [Paper](https://arxiv.org/pdf/2508.17061)
- [Github](https://github.com/AutoAILab/CARLA2Real)

## Implementation Plan

### 1. Data Parsing andtxt File Generation
- Walk through the `Input directory` (`/home/df/data/datasets/SkyScenes/proof_of_concept`).
- The directory structure contains `Height_Pitch` -> `Weather` (e.g., `ClearNoon`) -> `Town` (e.g., `Town10HD`).
- For each condition, collect matching frames across the available sensor directories:
  - `Images`: Source RGB frames.
  - `GBuffer`: `*.npz` G-buffer data files.
  - `CarlaSegment`: Semantic segmentation `*.npz` files (used as `SemanticSegmentation`).
- Generate a `test.txt` file (as required by `EPEExperiment.py`) where each line contains the comma-separated paths: `[Image_Path],[RobustImage_Path],[GBuffer_Path],[SemanticSegmentation_Path]`.
  - *Note: Since `RobustImages` are not available in this dataset, we can pass the `Images` path twice if the pipeline allows, or modify the data loading step to bypass `RobustImages` requirements during inference.*

### 2. Execution of the CARLA2Real Translation
- Use `test_pfd2cs.yaml` as the base configuration.
- Programmatically update the `fake_dataset.test_filelist` configuration pointing to the newly generated `test.txt` file.
- Run the inference command:
  ```bash
  python code/epe/EPEExperiment.py test <path-to-updated-yaml> --log=info
  ```
- Alternatively, modify the python script to pass the file list directly.

### 3. Output Reconstruction
- `EPEExperiment.py` outputs all frames into a flat directory defined by `dbg_dir` / `weight_save`.
- Create a post-processing step to read the generated images, map them back to their original `Height_Pitch/Weather/Town/Images` hierarchy, and save them into the `Output directory` (`/home/df/data/datasets/SkyScenes/realistic`).
- Ensure all other associated files (metadata, depth, instance, etc.) are either copied over or just the translated realistic `Images` folder is placed correctly side-by-side with the reference directories in the designated `Output directory`.

### Acceptance Criteria
- [ ] A python script or bash pipeline `run_pipeline.py/sh` is created.
- [ ] The pipeline seamlessly traverses the nested folders in the input directory.
- [ ] The pipeline successfully translates the images using `EPEExperiment.py` without errors regarding missing `RobustImages` or misaligned G-Buffer formats.
- [ ] The resulting images reside in `/home/df/data/datasets/SkyScenes/realistic` adhering strictly to the source tree architecture.