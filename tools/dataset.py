import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torchvision.transforms import functional as F

# VOC classes
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
    'bus', 'car', 'cat', 'chair', 'cow', 
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class PascalVOCDataset(Dataset):
    def __init__(self, root, split='train', transforms=None):
        self.root = root
        self.transforms = transforms
        self.split = split

        # 加载官方划分的文件列表
        split_file = os.path.join(root, "ImageSets", "Main", f"{split}.txt")
        with open(split_file) as f:
            file_names = [x.strip() for x in f.readlines()]
        
        # 只保留存在于Annotations和JPEGImages中的文件
        self.file_names = []
        for name in file_names:
            img_path = os.path.join(root, "JPEGImages", f"{name}.jpg")
            anno_path = os.path.join(root, "Annotations", f"{name}.xml")
            if os.path.exists(img_path) and os.path.exists(anno_path):
                self.file_names.append(name)
        
        
    def __getitem__(self, idx):
        name = self.file_names[idx]
        img_path = os.path.join(self.root, "JPEGImages", f"{name}.jpg")
        anno_path = os.path.join(self.root, "Annotations", f"{name}.xml")
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Load annotations
        tree = ET.parse(anno_path)
        root = tree.getroot()
        
        # Get image dimensions
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        
        boxes = []
        labels = []
        
        for obj in root.iter('object'):
            # Skip difficult objects
            is_difficult = int(obj.find('difficult').text)
            if is_difficult:
                continue
                
            # Get bounding box coordinates
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            
            # Handle boxes that go outside image boundaries
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(width, xmax)
            ymax = min(height, ymax)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(VOC_CLASSES.index(obj.find('name').text))
        
        # Convert to numpy array for Albumentations
        img_np = np.array(img,  dtype=np.uint8)
        boxes_np = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        
        # Apply transforms if any
        if self.transforms is not None:
            transformed = self.transforms(
                image=img_np,
                bboxes=boxes_np,
                labels=labels
            )
            img = transformed['image']  # This will be a tensor if using ToTensorV2
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        # Convert to tensors
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "area": (torch.as_tensor(boxes)[:, 3] - torch.as_tensor(boxes)[:, 1]) * 
                   (torch.as_tensor(boxes)[:, 2] - torch.as_tensor(boxes)[:, 0]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64)
        }
        
        return img, target
    
    def __len__(self):
        return len(self.file_names)

# class PascalVOCDataset(Dataset):
#     def __init__(self, root, transforms=None):
#         self.root = root
#         self.transforms = transforms
#         self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
#         self.annotations = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
        
#     def __getitem__(self, idx):
#         # Load image
#         img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
#         img = Image.open(img_path).convert("RGB")
        
#         # Load annotations
#         anno_path = os.path.join(self.root, "Annotations", self.annotations[idx])
#         tree = ET.parse(anno_path)
#         root = tree.getroot()
        
#         # Get image dimensions
#         width = int(root.find('size').find('width').text)
#         height = int(root.find('size').find('height').text)
        
#         boxes = []
#         labels = []
#         difficult = []
        
#         for obj in root.iter('object'):
#             # Skip difficult objects
#             is_difficult = int(obj.find('difficult').text)
#             if is_difficult:
#                 difficult.append(True)
#                 continue
#             difficult.append(False)
            
#             # Get bounding box coordinates
#             bndbox = obj.find('bndbox')
#             xmin = int(bndbox.find('xmin').text)
#             ymin = int(bndbox.find('ymin').text)
#             xmax = int(bndbox.find('xmax').text)
#             ymax = int(bndbox.find('ymax').text)
            
#             # Handle boxes that go outside image boundaries
#             xmin = max(0, xmin)
#             ymin = max(0, ymin)
#             xmax = min(width, xmax)
#             ymax = min(height, ymax)
            
#             boxes.append([xmin, ymin, xmax, ymax])
            
#             # Get label
#             name = obj.find('name').text
#             labels.append(VOC_CLASSES.index(name))
        
#         # Convert to tensors
#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         labels = torch.as_tensor(labels, dtype=torch.int64)
#         image_id = torch.tensor([idx])
#         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        
#         # Create target dictionary
#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#         target["image_id"] = image_id
#         target["area"] = area
#         target["iscrowd"] = iscrowd
        
#         # Apply transforms if any
#         if self.transforms is not None:
#             img, target = self.transforms(img, target)
        
#         return img, target
    
#     def __len__(self):
#         return len(self.imgs)
