import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Path to your results
results_dir = "/home/grannemann/PycharmProjects/EMVSNet/data/results/amini"

# List all npy files in the directory (change pattern if needed)
npy_files = sorted(glob.glob(os.path.join(results_dir, "*.npy")))

print(f"Found {len(npy_files)} .npy files.")

# Select a file to view (e.g., first one)
for fpath in npy_files:
    arr = np.load(fpath)
    print(f"Viewing {fpath}, shape={arr.shape}, dtype={arr.dtype}")
    plt.figure(figsize=(8, 6))
    plt.imshow(arr, cmap='jet')
    plt.colorbar()
    plt.title(os.path.basename(fpath))
    plt.show()

    # For quick browsing, uncomment the following to stop after one
    # break