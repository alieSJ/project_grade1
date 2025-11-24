# save_npy_to_csv.py
import numpy as np
import os

files = ["fc1_w.npy","fc1_b.npy","fc2_w.npy","fc2_b.npy","fc3_w.npy","fc3_b.npy"]
out_dir = "weights"
os.makedirs(out_dir, exist_ok=True)

for f in files:
    if not os.path.exists(f):
        print("Missing:", f)
        continue
    arr = np.load(f)
    out_path = os.path.join(out_dir, f.replace(".npy", ".csv"))
    # if 1D, save as single-row
    if arr.ndim == 1:
        arr2 = arr.reshape(1, -1)
    else:
        arr2 = arr
    np.savetxt(out_path, arr2, delimiter=",", fmt="%.8f")
    print("Saved", out_path, "shape", arr2.shape)

