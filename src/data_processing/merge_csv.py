import numpy as np
import pandas as pd


def calculate_curvature(row):
    """Calculate curvature from joystick velocity and angular velocity."""
    joystick = eval(row['joystick'])
    velocity = joystick[0]
    commanded_av = joystick[1]
    
    if velocity == 0.0:
        return 0.0
    else:
        return commanded_av / velocity


def merge_training_data():
    """
    Merge multiple CSV files containing training data.
    Filters out zero-curvature data points and saves to ikddata2.csv.
    """
    print("[INFO] Loading CSV files...")
    
    # Load all data files
    data_files = [
        pd.read_csv(f"./bag-files-and-data/processed-data/{i}.csv") 
        for i in range(1, 9)
    ]
    
    # Concatenate all dataframes
    data = pd.concat(data_files, axis=0, ignore_index=True)
    
    # Drop any unnamed index columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    
    # Calculate curvature for filtering
    print("[INFO] Calculating curvatures...")
    data['curvature'] = data.apply(calculate_curvature, axis=1)
    
    # Filter out zero curvature data
    print(f"[INFO] Removing {(data['curvature'] == 0).sum()} zero-curvature samples...")
    data = data[data['curvature'] != 0].copy()
    
    # Drop curvature column (was only needed for filtering)
    data = data.drop('curvature', axis=1)
    
    # Shuffle the data
    print("[INFO] Shuffling data...")
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save merged and cleaned data
    output_path = "./dataset/ikddata2.csv"
    data.to_csv(output_path, index=False)
    print(f"[INFO] Merged data saved to {output_path}")
    print(f"[INFO] Total samples: {len(data)}")


if __name__ == '__main__':
    merge_training_data()
