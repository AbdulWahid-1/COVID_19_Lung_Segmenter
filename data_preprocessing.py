# DESCRIPTION: Handles data curation, class imbalance mitigation, image resizing,
#              and robust spatial data augmentations using Albumentations.
import os
import cv2
import numpy as np
import albumentations as A
from sklearn.model_selection import train_test_split

class LungDatasetCurator:
    def __init__(self, data_dir, img_size=256):
        self.data_dir = data_dir
        self.img_size = img_size
        self.classes = ['Normal', 'COVID', 'Lung_Opacity', 'Viral_Pneumonia']
        
    def get_augmentation_pipeline(self, phase='train'):
       #Define geometric and intensity transformations to prevent overfitting
        if phase == 'train':
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

    def audit_class_distribution(self):
        """Scan directories to analyze class imbalances across categories."""
        print("[INFO] Auditing dataset class distributions...")
        distribution = {}
        for cls in self.classes:
            cls_path = os.path.join(self.data_dir, cls)
            if os.path.exists(cls_path):
                count = len(os.listdir(os.path.join(cls_path, 'images'))) if os.path.exists(os.path.join(cls_path, 'images')) else len(os.listdir(cls_path))
                distribution[cls] = count
            else:
                distribution[cls] = 0
        print(f"[METRICS] Class breakdown: {distribution}")
        return distribution

if __name__ == "__main__":
    curator = LungDatasetCurator(data_dir="./data/raw/")
    curator.audit_class_distribution()