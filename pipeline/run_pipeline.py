import os
import argparse
import numpy as np
import imageio.v2 as imageio
import cv2
from pathlib import Path
import yaml
import subprocess
from tqdm import tqdm
import shutil
import sys
import torch

# Ensure 'code' is in path so we can import epe modules if needed
PIPELINE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = PIPELINE_DIR.parent.resolve()
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT))

# Preprocessing constants from CARLA2Real
SPECIFIC_CLASSES = [11, 1, 2, 24, 25, 27, 14, 15, 16, 17, 18, 19, 10, 9, 12, 13, 6, 7, 8, 21, 23, 20, 3, 4, 5, 26, 28, 0, 22]

def material_from_gt_label(gt_labelmap):
    if len(gt_labelmap.shape) == 3:
        labels = gt_labelmap[:, :, 2] # Red channel in cv2.imread (BGR)
    else:
        labels = gt_labelmap

    r = (labels[:, :, np.newaxis] == np.array(SPECIFIC_CLASSES).reshape(1, 1, -1))

    class_sky = r[:, :, 0][:, :, np.newaxis]
    class_road = np.any(r[:, :, [1, 2, 3, 4, 5]], axis=2)[:, :, np.newaxis]
    class_vehicle = np.any(r[:, :, [6, 7, 8, 9, 10, 11]], axis=2)[:, :, np.newaxis]
    class_terrain = r[:, :, 12][:, :, np.newaxis]
    class_vegetation = r[:, :, 13][:, :, np.newaxis]
    class_person = np.any(r[:, :, [14, 15]], axis=2)[:, :, np.newaxis]
    class_infa = r[:, :, 16][:, :, np.newaxis]
    class_traffic_light = r[:, :, 17][:, :, np.newaxis]
    class_traffic_sign = r[:, :, 18][:, :, np.newaxis]
    class_ego = np.any(r[:, :, [19, 20]], axis=2)[:, :, np.newaxis]
    class_building = np.any(r[:, :, [21, 22, 23, 24, 25, 26]], axis=2)[:, :, np.newaxis]
    class_unlabeled = np.any(r[:, :, [27, 28]], axis=2)[:, :, np.newaxis]

    concatenated_array = np.concatenate((class_sky, class_road, class_vehicle, class_terrain, class_vegetation,
                                         class_person, class_infa, class_traffic_light, class_traffic_sign, class_ego,
                                         class_building, class_unlabeled), axis=2)
    return concatenated_array.astype(np.float32)

def preprocess_gbuffer(gbuf_path, img_path):
    gbuff = np.load(gbuf_path)
    # RGB Image as SceneColor baseline
    img = imageio.imread(img_path)
    if img.shape[2] == 4:
        scene_color = img[:, :, :3]
    else:
        scene_color = img
        
    # Get shape from scene_color
    h, w = scene_color.shape[:2]
    
    # SceneDepth - take first channel
    scene_depth = gbuff['SceneDepth'][:, :, 0:1]
    
    # CustomStencil - use SceneStencil as fallback
    if 'CustomStencil' in gbuff:
        custom_stencil = gbuff['CustomStencil'][:, :, 0:1]
    elif 'SceneStencil' in gbuff:
        custom_stencil = gbuff['SceneStencil'][:, :, 0:1]
    else:
        custom_stencil = np.zeros((h, w, 1), dtype=np.float32)
        
    gbA = gbuff.get('GBufferA', np.zeros((h, w, 3), dtype=np.float32))
    gbB = gbuff.get('GBufferB', np.zeros((h, w, 3), dtype=np.float32))
    gbC = gbuff.get('GBufferC', np.zeros((h, w, 3), dtype=np.float32))
    
    # Missing buffers in some generations: GBufferD (3), GBufferSSAO (1)
    gbD = gbuff.get('GBufferD', np.zeros((h, w, 3), dtype=np.float32))
    gbSSAO = gbuff.get('GBufferSSAO', np.zeros((h, w, 1), dtype=np.float32))
    if len(gbSSAO.shape) == 3:
        gbSSAO = gbSSAO[:, :, 0:1]
    
    gbuffers_list = [
        scene_color,
        scene_depth,
        gbA,
        gbB,
        gbC,
        gbD,
        gbSSAO,
        custom_stencil
    ]
    
    processed_list = []
    for buf in gbuffers_list:
        processed_list.append(buf.astype(np.float32))
        
    stacked_image = np.concatenate(processed_list, axis=2)
    return stacked_image

def run_regen_inference(town_dir, target_images_dir, regen_cfg, checkpoint_path, device, image_dir=None, overwrite=False):
    from epe.REGEN import regen_generator
    
    # Initialize Generator
    generator = regen_generator.define_G(
        input_nc=int(regen_cfg['input_nc']),
        output_nc=int(regen_cfg['output_nc']),
        ngf=int(regen_cfg['ngf']),
        netG=str(regen_cfg['netG']),
        norm=str(regen_cfg['norm']),
        n_downsample_global=int(regen_cfg['n_downsample_global']),
        n_blocks_global=int(regen_cfg['n_blocks_global']),
        n_local_enhancers=int(regen_cfg['n_local_enhancers'])
    ).to(device)

    print(f"Loading REGEN checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint)
    generator.eval()

    if image_dir is None:
        image_dir = town_dir / "Images"
    image_files = sorted(list(image_dir.glob("*.png")))
    
    target_images_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for img_path in tqdm(image_files, desc=f"REGEN Inference {town_dir.name}"):
            out_img_path = target_images_dir / img_path.name
            if out_img_path.exists() and not overwrite:
                continue

            img = imageio.imread(img_path)
            if img.shape[2] == 4:
                img = img[:, :, :3]
            # Normalize to [-1, 1] as per REGEN requirements in BaseExperiment.py
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # REGEN Inference
            output = generator(img_tensor)
            
            # Denormalize
            out_img = output[0].cpu().permute(1, 2, 0).numpy()
            out_img = ((out_img + 1) * 127.5).clip(0, 255).astype(np.uint8)
            
            # Save
            imageio.imwrite(out_img_path, out_img)

def main():
    parser = argparse.ArgumentParser(description="CARLA2Real Pipeline Orchestrator (EPE & REGEN)")
    parser.add_argument('--input', type=str, nargs='+', required=True, help='Input directory or directories (e.g. simulated)')
    parser.add_argument('--output', type=str, required=True, help='Output directory (e.g. realistic)')
    parser.add_argument('--config', type=str, default='code/config/carla_config.yaml', help='Base carla_config.yaml')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing processed images (default: skip)')
    args = parser.parse_args()

    input_roots = [Path(inp).resolve() for inp in args.input]
    output_root = Path(args.output).resolve()
    
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        # Try relative to PIPELINE_DIR if not found relative to PROJECT_ROOT
        config_path = PIPELINE_DIR / args.config
    
    with open(config_path, 'r') as f:
        carla_config = yaml.safe_load(f)

    method = carla_config['general'].get('method', 'EPE')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using method: {method} on {device}")

    town_dirs = [] # list of (root, town_path) tuples
    for input_root in input_roots:
        if not input_root.exists():
            print(f"Warning: Input root {input_root} does not exist. Skipping.")
            continue
            
        for root, dirs, files in os.walk(input_root):
            # Case-insensitive check for required directories
            dir_names = [d.lower() for d in dirs]
            if 'images' in dir_names and 'gbuffer' in dir_names and 'carlasegment' in dir_names:
                town_dirs.append((input_root, Path(root)))

    if not town_dirs:
        print("No valid sensor directories found (needs Images, GBuffer, CarlaSegment).")
        return

    print(f"Found {len(town_dirs)} condition directories to process.")

    for input_root, town_dir in town_dirs:
        rel_path = town_dir.relative_to(input_root)
        print(f"\n>>> Processing {rel_path} from {input_root.name}...")
        
        target_town_dir = output_root / rel_path
        target_images_dir = target_town_dir / "Images"
        
        if args.overwrite and target_images_dir.exists():
            print(f"Clearing old images in {target_images_dir.name} due to --overwrite...")
            for f in target_images_dir.glob("*.png"):
                f.unlink()
                
        target_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Helper to find case-insensitive subdirectory
        def get_subdir(parent, name):
            for d in parent.iterdir():
                if d.is_dir() and d.name.lower() == name.lower():
                    return d
            return parent / name # Fallback

        if method == "REGEN":
            checkpoint_name = carla_config['REGEN_settings']['checkpoint_name']
            checkpoint_path = CODE_ROOT / "checkpoints" / "REGEN" / checkpoint_name
            if not checkpoint_path.exists():
                print(f"Error: REGEN checkpoint not found at {checkpoint_path}")
                print("Please download it and place it in the correct directory.")
                continue
            
            # Find Images dir case-insensitively
            image_dir = get_subdir(town_dir, "Images")
            run_regen_inference(town_dir, target_images_dir, carla_config['REGEN_settings'], checkpoint_path, device, image_dir=image_dir, overwrite=args.overwrite)
            
        elif method == "EPE":
            # EPE logic requires test.txt manifest
            tmp_dir = target_town_dir / "tmp_preprocess"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            gbuf_tmp = tmp_dir / "GBuffers"
            seg_tmp = tmp_dir / "SemanticSegmentation"
            gbuf_tmp.mkdir(exist_ok=True)
            seg_tmp.mkdir(exist_ok=True)
            
            image_dir = get_subdir(town_dir, "Images")
            gbuffer_dir = get_subdir(town_dir, "GBuffer")
            segment_dir = get_subdir(town_dir, "CarlaSegment")
            
            image_files = sorted(list(image_dir.glob("*.png")))
            manifest_entries = []
            
            for img_path in tqdm(image_files, desc=f"Preprocessing EPE {rel_path.name}"):
                out_img_path = target_images_dir / img_path.name
                if out_img_path.exists() and not args.overwrite:
                    continue

                frame_id = img_path.stem
                gbuf_in = gbuffer_dir / f"{frame_id}_gbuffer.npz"
                gbuf_out = gbuf_tmp / f"{frame_id}.npz"
                if not gbuf_out.exists():
                    np.savez_compressed(gbuf_out, preprocess_gbuffer(gbuf_in, img_path))
                    
                seg_in = segment_dir / f"{frame_id}_semsegCarla.png"
                seg_out = seg_tmp / f"{frame_id}.npz"
                if not seg_out.exists():
                    seg_img = cv2.imread(str(seg_in))
                    if seg_img is not None:
                        np.savez_compressed(seg_out, material_from_gt_label(seg_img))
                
                manifest_entries.append(f"{img_path.absolute()},{img_path.absolute()},{gbuf_out.absolute()},{seg_out.absolute()}")
                
            if not manifest_entries:
                print(f"No new EPE images to process for {rel_path}")
                continue

            manifest_path = tmp_dir / "test.txt"
            with open(manifest_path, "w") as f:
                f.write("\n".join(manifest_entries))
            
            # Load Base Config for EPE (usually test_pfd2cs.yaml)
            epe_base_config = CODE_ROOT / "config" / "test_pfd2cs.yaml"
            with open(epe_base_config, 'r') as f:
                epe_cfg = yaml.safe_load(f)
            
            epe_cfg['fake_dataset']['test_filelist'] = str(manifest_path.absolute())
            epe_cfg['weight_dir'] = str((CODE_ROOT / "checkpoints").absolute())
            
            tmp_config_path = tmp_dir / "run_config_epe.yaml"
            with open(tmp_config_path, 'w') as f:
                yaml.dump(epe_cfg, f)
            
            cmd = ["uv", "run", "python", "epe/EPEExperiment.py", "test", str(tmp_config_path.absolute()), "--dbg_dir", str(target_town_dir.absolute()), "--log=info"]
            print(f"Running EPE translation...")
            
            # Ensure CODE_ROOT is in PYTHONPATH for the subprocess
            env = os.environ.copy()
            env["PYTHONPATH"] = str(CODE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            
            try:
                subprocess.run(cmd, check=True, cwd=str(CODE_ROOT), env=env)
                # Move EPE results
                out_images = target_town_dir / "translated"
                for f in out_images.glob("*.png"):
                    shutil.move(str(f), str(target_images_dir / f.name))
                shutil.rmtree(out_images)
            except subprocess.CalledProcessError as e:
                print(f"Error running EPE for {rel_path}: {e}")
                continue

        # Post-copy other data
        for d in ['Depth', 'Instance', 'metaData']:
            src_d = town_dir / d
            if src_d.exists():
                dst_d = target_town_dir / d
                if dst_d.exists(): shutil.rmtree(dst_d)
                shutil.copytree(src_d, dst_d)

    print("\nPipeline execution completed.")

if __name__ == "__main__":
    main()
