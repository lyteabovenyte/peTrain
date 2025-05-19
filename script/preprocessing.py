"""
    CPU-friendly point cloud preprocessing and dataset creation
"""
import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import random
import glob
from tqdm import tqdm
from multiprocessing import cpu_count

def load_intrinsics(width=300, height=300, fov=90.0):
    # Compute focal length from FOV
    f = 0.5 * width / np.tan(0.5 * np.deg2rad(fov))
    cx = width / 2
    cy = height / 2
    return f, cx, cy

def depth_to_point_cloud(depth, f, cx, cy):
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (x - cx) * z / f
    y = (y - cy) * z / f
    points = np.stack((x, y, z), axis=-1)
    return points.reshape(-1, 3)

def instance_to_labels(instance_img):
    # Flatten to match point cloud shape
    h, w, _ = instance_img.shape
    instance_flat = instance_img.reshape(-1, 3)
    color_strings = [tuple(c) for c in instance_flat]
    color_to_id = {c: i for i, c in enumerate(sorted(set(color_strings)))}
    labels = np.array([color_to_id[c] for c in color_strings], dtype=np.int32)
    return labels

def preprocess_all(data_dir="../data/light_test", out_dir="../data/processed", num_points=1024):
    """
    Preprocess all data with memory-efficient processing
    """
    os.makedirs(out_dir, exist_ok=True)
    f, cx, cy = load_intrinsics()

    depth_files = sorted(glob.glob(os.path.join(data_dir, "depth_*.npy")))
    
    for i, depth_path in enumerate(tqdm(depth_files)):
        base = os.path.basename(depth_path).split(".")[0].split("_")[1]
        rgb_path = os.path.join(data_dir, f"rgb_{base}.png")
        inst_path = os.path.join(data_dir, f"instance_{base}.png")

        # Load and process depth
        depth = np.load(depth_path)
        instance = cv2.imread(inst_path)[:, :, ::-1]  # BGR to RGB

        # Process in chunks to save memory
        chunk_size = 1000000  # Process 1M points at a time
        points = depth_to_point_cloud(depth, f, cx, cy)
        labels = instance_to_labels(instance)

        # Filter invalid points
        valid = (depth.reshape(-1) > 0.1) & (depth.reshape(-1) < 5.0)
        points = points[valid]
        labels = labels[valid]

        # Normalize points
        centroid = np.mean(points, axis=0)
        points = points - centroid
        m = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points = points / m

        # Sample points if needed
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
            labels = labels[indices]

        # Save processed data
        out_file = os.path.join(out_dir, f"cloud_{base}.npz")
        np.savez_compressed(out_file, xyz=points, label=labels)

    print(f"✅ Done preprocessing. Saved point clouds in {out_dir}")


class PointCloudDataset(Dataset):
    def __init__(self, root_dir, num_points=1024, partition='train', transform=None):
        """
        Memory-efficient point cloud dataset
        """
        self.root_dir = root_dir
        self.num_points = num_points
        self.partition = partition
        self.transform = transform
        
        # Store file paths instead of loading all data
        self.files = sorted(glob.glob(os.path.join(root_dir, '*.npz')))
        
        # Split into train/test
        if partition == 'train':
            self.files = self.files[:int(0.8 * len(self.files))]
        else:
            self.files = self.files[int(0.8 * len(self.files)):]
    
    def _normalize_points(self, points):
        """Normalize point cloud to unit sphere"""
        centroid = np.mean(points, axis=0)
        points = points - centroid
        m = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points = points / m
        return points
    
    def _random_sample(self, points):
        """Randomly sample points from point cloud"""
        if len(points) >= self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
        else:
            indices = np.random.choice(len(points), self.num_points, replace=True)
        return points[indices]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load data on demand
        data = np.load(self.files[idx])
        points = data['xyz']
        label = data['label']
        
        # Process points
        points = self._normalize_points(points)
        points = self._random_sample(points)
        
        # Convert to torch tensor
        points = torch.FloatTensor(points)
        label = torch.LongTensor([label])[0]
        
        if self.transform:
            points = self.transform(points)
        
        return points, label

class PointCloudTransform:
    def __init__(self, augment=True):
        self.augment = augment
    
    def __call__(self, points):
        if self.augment:
            # Random rotation around z-axis
            theta = np.random.uniform(0, 2*np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            points = points @ rotation_matrix
            
            # Random jitter (reduced magnitude for CPU)
            jitter = np.random.normal(0, 0.01, size=points.shape)
            points = points + jitter
            
            # Random scale (reduced range for CPU)
            scale = np.random.uniform(0.9, 1.1)
            points = points * scale
        
        return points

def get_dataloaders(processed_dir, batch_size=16, num_points=1024):
    """
    Create CPU-friendly dataloaders
    """
    # Use fewer workers for CPU
    num_workers = min(4, cpu_count())
    
    # Create datasets
    train_dataset = PointCloudDataset(
        root_dir=processed_dir,
        num_points=num_points,
        partition='train',
        transform=PointCloudTransform(augment=True)
    )
    
    test_dataset = PointCloudDataset(
        root_dir=processed_dir,
        num_points=num_points,
        partition='test',
        transform=PointCloudTransform(augment=False)
    )
    
    # Create dataloaders with CPU-friendly settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Disable pin_memory for CPU
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # Disable pin_memory for CPU
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, test_loader

if __name__ == '__main__':
    raw_dir = '../data/light_test'
    processed_dir = '../data/processed'
    
    # Preprocess data
    preprocess_all(raw_dir, processed_dir, num_points=1024)
    
    # Get dataloaders with CPU-friendly settings
    train_loader, test_loader = get_dataloaders(
        processed_dir,
        batch_size=16,  # Smaller batch size for CPU
        num_points=1024
    )
    
    # Test dataloaders
    for points, labels in train_loader:
        print(f'Batch shape: {points.shape}')
        print(f'Labels shape: {labels.shape}')
        break
