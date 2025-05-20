"""
    CPU-friendly point cloud preprocessing and dataset creation.
    for our test we have only 1260 datapoints, so we need aggressive data augmentation for better generalization.
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
import open3d as o3d

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

def preprocess_all(data_dir="../data/raw", out_dir="../data/processed", num_points=1024):
    """
    Preprocess all data with memory-efficient processing and save as PLY files
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

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Add colors from RGB image if available
        if os.path.exists(rgb_path):
            rgb = cv2.imread(rgb_path)[:, :, ::-1]  # BGR to RGB
            rgb_flat = rgb.reshape(-1, 3)[valid]
            if len(points) > num_points:
                rgb_flat = rgb_flat[indices]
            pcd.colors = o3d.utility.Vector3dVector(rgb_flat / 255.0)

        # Save as PLY file
        out_file = os.path.join(out_dir, f"cloud_{base}.ply")
        o3d.io.write_point_cloud(out_file, pcd, write_ascii=True)
        
        # Save labels separately
        label_file = os.path.join(out_dir, f"labels_{base}.npy")
        np.save(label_file, labels)

    print(f"✅ Done preprocessing. Saved point clouds in {out_dir}")


class PointCloudDataset(Dataset):
    def __init__(self, root_dir, num_points=1024, partition='train', transform=None):
        """
        Memory-efficient point cloud dataset for PLY files
        """
        self.root_dir = root_dir
        self.num_points = num_points
        self.partition = partition
        self.transform = transform
        
        # Store file paths instead of loading all data
        self.ply_files = sorted(glob.glob(os.path.join(root_dir, 'cloud_*.ply')))
        self.label_files = sorted(glob.glob(os.path.join(root_dir, 'labels_*.npy')))
        
        # Split into train/test
        if partition == 'train':
            self.ply_files = self.ply_files[:int(0.8 * len(self.ply_files))]
            self.label_files = self.label_files[:int(0.8 * len(self.label_files))]
        else:
            self.ply_files = self.ply_files[int(0.8 * len(self.ply_files)):]
            self.label_files = self.label_files[int(0.8 * len(self.label_files)):]
    
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
        return len(self.ply_files)
    
    def __getitem__(self, idx):
        # Load data on demand
        pcd = o3d.io.read_point_cloud(self.ply_files[idx])
        points = np.asarray(pcd.points)
        label = np.load(self.label_files[idx])
        
        # Process points
        points = self._normalize_points(points)
        points = self._random_sample(points)
        
        # Convert to torch tensor - fix the slow conversion warning
        points = torch.from_numpy(points).float()
        label = torch.from_numpy(np.array(label)).long()
        
        if self.transform:
            points = self.transform(points)
        
        return points, label

class PointCloudTransform:
    def __init__(self, augment=True):
        self.augment = augment
    
    def __call__(self, points):
        # Create a copy of the points tensor to avoid memory sharing issues
        points = points.clone()
        
        if self.augment:
            # Random rotation around all axes
            theta_x = np.random.uniform(0, 2*np.pi)
            theta_y = np.random.uniform(0, 2*np.pi)
            theta_z = np.random.uniform(0, 2*np.pi)
            
            # Create rotation matrices
            rotation_x = torch.tensor([
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)]
            ], dtype=points.dtype, device=points.device)
            
            rotation_y = torch.tensor([
                [np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)]
            ], dtype=points.dtype, device=points.device)
            
            rotation_z = torch.tensor([
                [np.cos(theta_z), -np.sin(theta_z), 0],
                [np.sin(theta_z), np.cos(theta_z), 0],
                [0, 0, 1]
            ], dtype=points.dtype, device=points.device)
            
            # Apply rotations
            rotation_matrix = torch.matmul(torch.matmul(rotation_z, rotation_y), rotation_x)
            points = torch.matmul(points, rotation_matrix)
            
            # Random jitter
            jitter = torch.randn_like(points) * 0.02
            points = points + jitter
            
            # Random scale
            scale = torch.tensor(np.random.uniform(0.8, 1.2), dtype=points.dtype, device=points.device)
            points = points * scale
            
            # Random flips
            if np.random.random() > 0.5:
                points = points.clone()
                points[:, 0] = -points[:, 0]
            
            if np.random.random() > 0.5:
                points = points.clone()
                points[:, 1] = -points[:, 1]
            
            # Random translation
            translation = torch.tensor(np.random.uniform(-0.1, 0.1, size=(3,)), 
                                    dtype=points.dtype, device=points.device)
            points = points + translation
            
            # Random dropout - fixed memory sharing issue
            if np.random.random() > 0.5:
                drop_ratio = np.random.uniform(0, 0.1)
                drop_idx = torch.rand(points.shape[0], device=points.device) <= drop_ratio
                if drop_idx.any():
                    # Create a new tensor for the result
                    new_points = points.clone()
                    new_points[drop_idx] = points[0].clone()
                    points = new_points
            
            # Random noise
            if np.random.random() > 0.5:
                noise = torch.randn_like(points) * 0.01
                points = points + noise
        
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
    raw_dir = '../data/raw'
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
