# from models import MODEL_REGISTRY
import os
import numpy as np


# def get_model(name, cfg):
#     assert name in MODEL_REGISTRY, f"Unknown model: {name}"
#     return MODEL_REGISTRY[name](cfg)


def compute_num_classes(label_dir):
    unique_labels = set()
    
    for fname in os.listdir(label_dir):
        if fname.endswith('.npy'):
            labels = np.load(os.path.join(label_dir, fname))
            unique_labels.update(np.unique(labels))

    return len(unique_labels), sorted(unique_labels)


label_dir = '../data/processed/label_dir'
num_classes, class_ids = compute_num_classes(label_dir)
print(f"Detected {num_classes} classes.")
print(f"Class IDs: {class_ids}")