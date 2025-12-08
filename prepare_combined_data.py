import pandas as pd
import numpy as np
import os

# Configuration
# Dataset 1: SandGrain
sg_csv = 'SandGrain_Dataset/usace_1024_aug_dry_set1_2_3_4_5_aug2021.csv'
sg_prefix_replace = ('sandsnap_images/', 'SandGrain_Dataset/')

# Dataset 2: Global/Generic Sand (Superset of generic sand/gravel)
gen_csv = 'grain_size_global/global_all4.csv'
# images in 'images/' which matches the CSV entries, so no prefix change needed.

output_dir = 'combined_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

train_output = os.path.join(output_dir, 'train.csv')
test_output = os.path.join(output_dir, 'test.csv')

# Common columns to keep
common_vars = ['P10', 'P16', 'P25', 'P50', 'P75', 'P84', 'P90']
keep_columns = ['files'] + common_vars

from PIL import Image

def is_valid_image(path):
    try:
        if not os.path.exists(path):
            return False
        with Image.open(path) as img:
            img.verify() # Verify file integrity
        return True
    except:
        return False


def prepare_combined_data():
    print("--- Loading SandGrain Dataset ---")
    try:
        df_sg = pd.read_csv(sg_csv)
        # Fix paths
        df_sg['files'] = df_sg['files'].astype(str).str.replace(sg_prefix_replace[0], sg_prefix_replace[1], regex=False)
        # Filter columns
        # Check if all common vars exist
        missing_cols = [c for c in common_vars if c not in df_sg.columns]
        if missing_cols:
            print(f"Warning: SandGrain missing columns {missing_cols}")
        
        df_sg = df_sg[keep_columns].copy()
        
        # Filter missing files
        initial_count = len(df_sg)
        df_sg['valid'] = df_sg['files'].apply(is_valid_image)
        df_sg = df_sg[df_sg['valid']].drop(columns=['valid'])
        print(f"SandGrain: {len(df_sg)} / {initial_count} valid images.")
        
    except Exception as e:
        print(f"Error loading SandGrain: {e}")
        return

    print("\n--- Loading Generic Sand Dataset ---")
    try:
        df_gen = pd.read_csv(gen_csv)
        # Path is already 'images/IMG...', verifying existence relative to root
        
        # Filter columns
        missing_cols = [c for c in common_vars if c not in df_gen.columns]
        if missing_cols:
            print(f"Warning: Generic Sand missing columns {missing_cols}")
            
        df_gen = df_gen[keep_columns].copy()
        
        # Filter missing files
        initial_count = len(df_gen)
        df_gen['valid'] = df_gen['files'].apply(is_valid_image)
        
        # Debugging: check where it's looking for files
        # print("Sample generic path:", df_gen['files'].iloc[0])
        # print("Exists?", os.path.exists(df_gen['files'].iloc[0]))
        
        df_gen = df_gen[df_gen['valid']].drop(columns=['valid'])
        print(f"Generic Sand: {len(df_gen)} / {initial_count} valid images.")
        
    except Exception as e:
        print(f"Error loading Generic Sand: {e}")
        return

    print("\n--- Combining Datasets ---")
    combined_df = pd.concat([df_sg, df_gen], ignore_index=True)
    print(f"Total samples: {len(combined_df)}")
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split 80/20
    train_size = int(0.8 * len(combined_df))
    train_df = combined_df.iloc[:train_size]
    test_df = combined_df.iloc[train_size:]
    
    print(f"Saving combined train ({len(train_df)}) and test ({len(test_df)}) to {output_dir}")
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    print("Done.")

if __name__ == "__main__":
    prepare_combined_data()
