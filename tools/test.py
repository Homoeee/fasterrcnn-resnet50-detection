import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader
import numpy as np
import argparse
from dataset import PascalVOCDataset, VOC_CLASSES
from albumentations.pytorch import ToTensorV2
import albumentations as A
from utils import calculate_ap, visualize_results, visualize_predictions
from collections import defaultdict
import matplotlib

# Set matplotlib backend to avoid GUI issues
matplotlib.use('Agg')

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN model on Pascal VOC')
    parser.add_argument('--data-dir', default='data/VOCdevkit/VOC2012', help='Path to dataset')
    parser.add_argument('--output-dir', default='fasterrcnn-resnet50/result/2-best-losses', help='Directory to save output files')
    parser.add_argument('--batch-size', default=8, type=int, help='Batch size for testing')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers for data loading')
    parser.add_argument('--model-path', default='fasterrcnn-resnet50/checkpoints/2/best_losses_model.pth', help='Path to trained model checkpoint')
    parser.add_argument('--device', default='cuda', help='Device to use for testing (cpu or cuda)')
    parser.add_argument('--iou-threshold', default=0.5, type=float, help='IOU threshold for mAP calculation')
    parser.add_argument('--num-visualize', default=20, type=int, help='Number of images to visualize')
    parser.add_argument('--confidence-threshold', default=0.3, type=float, help='Confidence threshold for predictions')
    
    return parser.parse_args()

def get_transform():
    """Return test transformations"""
    transform = A.Compose([
        A.Resize(height=800, width=800),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels']
    ))
    return transform

def get_model(num_classes):
    """Create Faster R-CNN model with ResNet50 backbone"""
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    backbone = torchvision.models.resnet50(weights=weights)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    backbone.out_channels = 2048
    
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def evaluate_model(model, data_loader, device, iou_threshold=0.5):
    """Evaluate model on test set and calculate mAP"""
    model.eval()
    
    # Storage for evaluation metrics
    all_aps = []  # 存储每张图片的AP
    all_ap_dict = defaultdict(list)  # 存储每个类别的AP
    all_ap_len = defaultdict(int)  # 存储每个类别的AP数量
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images)
            
            for target, output in zip(targets, outputs):
                ap_dict = calculate_ap(
                    target['boxes'], target['labels'],
                    output['boxes'], output['scores'], output['labels'],
                    iou_threshold=iou_threshold
                )
                # print(f"AP for image: {ap_dict}")
                for class_id, ap in ap_dict.items():
                    # print(f"Class {class_id}: AP = {ap}")
                    all_ap_dict[VOC_CLASSES[class_id]] = all_ap_dict.get(VOC_CLASSES[class_id], 0.0) + ap
                    all_ap_len[VOC_CLASSES[class_id]] += 1
                
    
    # 计算整体mAP
    mean_ap = np.mean(all_ap_dict.values()) if all_aps else 0.0
    for item, value in all_ap_dict.items():
        if all_ap_len[item] > 0:
            all_ap_dict[item] = value / all_ap_len[item]
        else:
            all_ap_dict[item] = 0.0

    return mean_ap, all_ap_dict  # 返回兼容格式

def main():
    args = get_args()
    
    # Create test dataset and data loader
    dataset_test = PascalVOCDataset(args.data_dir, split='val', transforms=get_transform())
    
    data_loader_test = DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, collate_fn=collate_fn
    )
    
    # Create model and load trained weights
    model = get_model(len(VOC_CLASSES) + 1)  # +1 for background
    model.to(args.device)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.model_path}")
    
    # Evaluate model
    overall_mAP, mean_aps = evaluate_model(
        model, data_loader_test, args.device, args.iou_threshold
    )
    
    # Print results
    print(f"\nOverall mAP@{args.iou_threshold}: {overall_mAP:.4f}")
    print("\nPer-class AP:")
    for class_id, ap in sorted(mean_aps.items(), key=lambda x: x[1], reverse=True):
        print(f"{class_id}: {ap:.4f}") 
    
    visualize_results(mean_aps, args.output_dir)
    
    # Visualize predictions on sample images
    print(f"\nVisualizing predictions on {args.num_visualize} random images...")
    visualize_predictions(model, dataset_test, args.device, args.output_dir, args.num_visualize, 
            iou_threshold=args.iou_threshold, confidence_threshold=args.confidence_threshold)
    
    print(f"\nAll results saved to {args.output_dir}")

if __name__ == "__main__":
    main()