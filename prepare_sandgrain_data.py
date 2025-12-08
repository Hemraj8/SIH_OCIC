import pandas as pd
import numpy as np
import os

# Configuration
input_csv = 'SandGrain_Dataset/usace_1024_aug_dry_set1_2_3_4_5_aug2021.csv'
output_dir = 'SandGrain_Dataset'
train_output = os.path.join(output_dir, 'train.csv')
test_output = os.path.join(output_dir, 'test.csv')

# The CSV has paths like: sandsnap_images/Folder/File
# The dataset has paths like: SandGrain_Dataset/Folder/File
# However, "sandsnap_images" is NOT in SandGrain_Dataset.
# So we replace "sandsnap_images/" with "SandGrain_Dataset/"

path_prefix_to_replace = 'sandsnap_images/'
replacement_prefix = 'SandGrain_Dataset/'

def prepare_data():
    print(f"Reading {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: File {input_csv} not found.")
        return

    # Fix paths
    print("Fixing image paths...")
    df['files'] = df['files'].astype(str).str.replace(path_prefix_to_replace, replacement_prefix, regex=False)
    
    # Filter missing files
    print("Filtering missing files...")
    initial_count = len(df)
    
    # Efficient way to check existence
    # SediNet expects relative paths. I will verify if they exist relative to CWD.
    
    def file_exists(path):
        return os.path.exists(path)
    
    df['exists'] = df['files'].apply(file_exists)
    
    missing_df = df[~df['exists']]
    if len(missing_df) > 0:
        print(f"WARNING: {len(missing_df)} files not found.")
        print("Sample missing:", missing_df['files'].iloc[0])
    
    df = df[df['exists']].drop(columns=['exists'])
    final_count = len(df)
    
    print(f"Retained {final_count} / {initial_count} images.")

    if final_count == 0:
        print("ERROR: No images found. Check paths.")
        return
    
    # Shuffle
    print("Shuffling data...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split 80/20
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    print(f"Saving train ({len(train_df)} rows) and test ({len(test_df)} rows) CSVs...")
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    print("Done.")

if __name__ == "__main__":
    prepare_data()
