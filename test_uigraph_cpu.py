# Test script for ShowUI uigraph generation on CPU
import os

# Set environment variable to force CPU usage if CUDA is available but user wants CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from PIL import Image
import numpy as np

# Add the directory containing 'model' to sys.path so we can import ShowUI modules
sys.path.append(r"D:\Magma_chaythu\Show_ui\ShowUI")
import transformers.image_utils
if not hasattr(transformers.image_utils, 'VideoInput'):
    transformers.image_utils.VideoInput = object

# Print sys.path for debugging
print(f"Added to path: {sys.path[-1]}")

try:
    # Try importing as if we are in the root of 'ShowUI'
    from model.showui.image_processing_showui import ShowUIImageProcessor
except ImportError as e:
    print(f"\nImport Error Detail: {e}")
    # Fallback: try importing if 'ShowUI' is the package
    try:
        sys.path.append(r"D:\Magma_chaythu\Show_ui")
        from Show_ui.ShowUI.model.showui.image_processing_showui import ShowUIImageProcessor
        print("Success: Imported from alternative path")
    except ImportError as e2:
        print(f"Alternative Import Error: {e2}")
        print("Error importing ShowUIImageProcessor. Make sure you have installed: torch, torchvision, transformers, scikit-image, scikit-learn")
        sys.exit(1)

def test_uigraph():
    # 1. Define image path
    # Using one of the examples found in ShowUI
    image_path = r"D:\Magma_chaythu\image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        return

    print(f"Loading image from {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    # 2. Initialize the processor
    # Parameters inferred from code reading or defaults
    processor = ShowUIImageProcessor(
        min_pixels=256*28*28,
        max_pixels=1344*28*28,
        patch_size=14,
        merge_size=2
    )

    # 3. Define uigraph parameters
    uigraph_diff = 5.0  # Threshold for pixel difference
    
    print(f"\nRunning preprocessing with uigraph_use=True, uigraph_diff={uigraph_diff}")
    
    # Create output directory for visualization
    vis_dir = "vis_output"
    os.makedirs(vis_dir, exist_ok=True)

    # 4. Run preprocessing
    # The 'preprocess' method returns a BatchFeature
    # internally calls _preprocess which calls _build_uigraph
    # We pass vis_dir to enable visualization saving inside the method (based on code reading)
    
    # NOTE: The preprocess method in image_processing_showui.py handles saving if vis_dir is provided
    batch_feature = processor.preprocess(
        images=image,
        uigraph_use=True,
        uigraph_diff=uigraph_diff,
        vis_dir=vis_dir,
        return_tensors="np" # Numpy is fine for CPU
    )
    
    # 5. Analyze results
    print("\n--- Results ---")
    
    # patch_assign contains the flat list of component IDs for ALL images concatenated
    # Since we only processed 1 image, the whole array corresponds to that image
    patch_assign = batch_feature['patch_assign']
    
    # If it is wrapped in another array level, flatten it
    if hasattr(patch_assign, 'flatten'):
        patch_assign = patch_assign.flatten()
    
    total_patches = len(patch_assign)
    num_components = len(np.unique(patch_assign))
    
    print(f"Total visual tokens: {total_patches}")
    print(f"Number of connected components (UI elements): {num_components}")
    print(f"Reduction Ratio (Components / Tokens): {num_components / total_patches:.4f}")
    
    print(f"\nVisualization saved to: {os.path.abspath(vis_dir)}/demo.png")

if __name__ == "__main__":
    test_uigraph()
