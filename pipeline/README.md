# CARLA2Real Pipeline

This directory contains the orchestrated pipeline for generating realistic CARLA data.

## Contents
- `run_pipeline.py`: The main orchestrator script for dataset translation.

## Usage
The pipeline supports both **EPE** (slow, high quality) and **REGEN** (fast) methods.

To run the pipeline:
1. Ensure your dependencies are installed via `uv`.
2. Configure your desired method and checkpoints in `code/config/carla_config.yaml`.
3. Execute the orchestrator from the project root:
   ```bash
   uv run python pipeline/run_pipeline.py --input <input_dir> --output <output_dir>
   ```

### Arguments
- `--input`: Root directory of the synthetic dataset (e.g., `simulated`).
- `--output`: Root directory where the realistic dataset will be saved.
- `--config`: (Optional) Path to a custom `carla_config.yaml`. Defaults to `code/config/carla_config.yaml`.
- `--overwrite`: (Optional) Flag to overwrite and clear out existing images in the output directory. If omitted, existing images will be skipped.

### Example
```bash
uv run pipeline/run_pipeline.py --input /home/df/data/datasets/SkyScenes/simulated --output /home/df/data/datasets/SkyScenes/realistic --overwrite
```

## Data Preparation
The script expects the input directory to contain nested folders with the following structure:
- `Images/*.png`
- `GBuffer/*_gbuffer.npz`
- `CarlaSegment/*_semsegCarla.png`

The pipeline automatically handles preprocessing (GBuffer stacking and Semantic category mapping) before passing data to the models.
