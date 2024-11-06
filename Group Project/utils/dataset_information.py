import os
import pandas as pd
import h5py
import re
import numpy as np

def print_dataset_shapes(data_folder):
    for root, dirs, files in os.walk(data_folder):
        level = root.replace(data_folder, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.csv'):
                data = pd.read_csv(file_path)
                print(f"{subindent}{file} - Shape: {data.shape}")
            elif file.endswith('.h5') or file.endswith('.hdf5'):
                with h5py.File(file_path, 'r') as hdf:
                    keys = list(hdf.keys())
                    # Sort keys numerically
                    keys.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
                    print(f"{subindent}{file}/")
                    total_shape = []
                    for key in keys:
                        data = hdf[key]
                        # print(f"{subindent}    {key} - Shape: {data.shape}")
                        total_shape.append(data.shape)
                    # Print the shape of the sub-indent (summary of all shapes)
                    if total_shape:
                        total_shape_np = np.array(total_shape)
                        print(f"{subindent}Total Shape: {total_shape_np.shape}")

if __name__ == "__main__":
    data_folder = 'data'
    print_dataset_shapes(data_folder)