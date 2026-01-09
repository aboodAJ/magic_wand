import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2

class COCOSegmentationDataset(Dataset):
    def __init__(self, root_dir, ann_file, transforms=None):
        self.root_dir = root_dir
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        # Filter out images without annotations if necessary
        self.ids = [img_id for img_id in self.ids if len(self.coco.getAnnIds(imgIds=img_id)) > 0]
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        path = coco.loadImgs(img_id)[0]['file_name']
        image_path = os.path.join(self.root_dir, path)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, coco.annToMask(ann))
        
        # Augmentations
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Ensure mask is a tensor with channel dimension if not already handled by ToTensorV2
        # ToTensorV2 converts image to (C, H, W) and mask to (H, W) usually, or (H, W) for mask.
        # U-Net expects (N, 1, H, W) for mask usually.
        
        if isinstance(mask, torch.Tensor):
             if mask.ndim == 2:
                mask = mask.unsqueeze(0) # Add channel dim: (1, H, W)
        
        return image, mask.float()

    def __len__(self):
        return len(self.ids)

def get_training_augmentations(height=512, width=512):
    return A.Compose([
        A.Resize(height=height, width=width),
        
        # Geometric (Geometric)
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        
        # Photometric (Photometric)
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.GaussianBlur(p=0.3),
        A.CLAHE(p=0.2),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_validation_augmentations(height=512, width=512):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
