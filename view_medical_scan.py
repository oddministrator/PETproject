import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION SECTION ---
# Define both scan configurations
scans = [
    {
        "name": "PET",
        "filename": "raws/IL46_1_D150x166x199_V1.0,1.0,1.0_S-76.3,-160.5,261.5.raw",
        "dims": (199, 166, 150),  # (Z, Y, X)
        "dtype": np.float32
    },
    {
        "name": "CT",
        "filename": "raws/IL46_2_D400x400x399_V0.5,0.5,0.5_S-99.75,-184.75,264.35.raw",
        "dims": (399, 400, 400),  # (Z, Y, X)
        "dtype": np.float32
    }
]

# --- PROCESS ALL SCANS ---

for scan in scans:
    print(f"\n{'='*60}")
    print(f"Processing {scan['name']} scan...")
    print(f"{'='*60}")
    
    filename = scan["filename"]
    dims = scan["dims"]
    dtype = scan["dtype"]
    
    try:
        # 1. Load the raw binary data
        print(f"Loading {filename}...")
        with open(filename, 'rb') as f:
            data = np.fromfile(f, dtype=dtype)
        
        # 2. Check if the file size matches the expected dimensions
        expected_voxels = dims[0] * dims[1] * dims[2]
        if data.size != expected_voxels:
            print(f"Error: File contains {data.size} voxels, but dimensions {dims} require {expected_voxels}.")
            print("Skipping this scan.")
            continue
        
        # 3. Reshape the flat array into a 3D volume
        volume = data.reshape(dims)
        print(f"Volume shape: {volume.shape}")
        
        # 4. Create output directory
        output_dir = f"outputs/{scan['name']}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # 5. Process all slices
        num_slices = dims[0]
        print(f"Processing {num_slices} slices...")
        
        for slice_idx in range(num_slices):
            # Extract slice
            slice_img = volume[slice_idx, :, :]
            
            # Create figure
            plt.figure(figsize=(8, 8))
            plt.imshow(slice_img, cmap='gray', origin='lower')
            plt.title(f"{scan['name']} - Slice {slice_idx}/{num_slices-1}")
            plt.colorbar(label="Intensity (Float32)")
            plt.xlabel("X axis (pixels)")
            plt.ylabel("Y axis (pixels)")
            
            # Save the image
            output_filename = f"{output_dir}/slice_{slice_idx:04d}.png"
            plt.savefig(output_filename, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Progress indicator
            if (slice_idx + 1) % 20 == 0 or slice_idx == num_slices - 1:
                print(f"  Processed {slice_idx + 1}/{num_slices} slices...")
        
        print(f"✓ Success! All {num_slices} slices saved to {output_dir}/")

    except FileNotFoundError:
        print(f"✗ Could not find file: {filename}")
        print("Make sure the file exists in the raws/ directory.")
    except Exception as e:
        print(f"✗ An error occurred: {e}")

print(f"\n{'='*60}")
print("All scans processed!")
print(f"{'='*60}")

