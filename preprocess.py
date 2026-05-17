#!/usr/bin/env python3
"""
SUPER SIMPLE PREPROCESSING FOR BEER IMAGES
Only 3 steps: Load → Standardize → Save
"""

import os
import numpy as np
from PIL import Image
import pickle

def preprocess_beer_images():
    """Simple preprocessing: Load → Standardize → Save"""
    
    print("🍺 PREPROCESSING BEER IMAGES")
    print("=" * 30)
    
    # Step 1: Load images
    print("Step 1: Loading images...")
    
    train_images = []
    train_labels = []
    
    # Load from data/train folder
    for folder_name in sorted(os.listdir("data/train")):
        if os.path.isdir(os.path.join("data/train", folder_name)):
            class_number = int(folder_name[0])  # 0, 1, 2, 3, 4
            
            folder_path = os.path.join("data/train", folder_name)
            image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
            
            for image_file in image_files:
                image_path = os.path.join(folder_path, image_file)
                
                # Load and resize image
                img = Image.open(image_path).convert('RGB').resize((160, 160))
                img_array = np.array(img) / 255.0  # 0-1 range
                
                train_images.append(img_array)
                train_labels.append(class_number)
    
    # Convert to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    
    print(f"Loaded {len(train_images)} images")
    
    # Step 2: Standardize (Z-score)
    print("Step 2: Standardizing...")
    
    global_mean = np.mean(train_images)
    global_std = np.std(train_images)
    
    # Apply: (x - mean) / std
    train_images_std = (train_images - global_mean) / global_std
    
    print(f"Mean: {global_mean:.3f}, Std: {global_std:.3f}")
    
    # Step 3: Save
    print("Step 3: Saving...")
    
    data = {
        'X_train': train_images_std,
        'y_train': train_labels,
        'mean': global_mean,
        'std': global_std
    }
    
    with open('beer_data_processed.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print("✅ Done! Saved as 'beer_data_processed.pkl'")
    return data

def load_processed_data():
    """Load the processed data"""
    with open('beer_data_processed.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Run preprocessing
    data = preprocess_beer_images()
    
    # Test loading
    print("\nTesting...")
    loaded = load_processed_data()
    print(f"X_train shape: {loaded['X_train'].shape}")
    print(f"y_train shape: {loaded['y_train'].shape}")
