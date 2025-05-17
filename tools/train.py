import os
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
from dataset import PascalVOCDataset, VOC_CLASSES
from albumentations.pytorch import ToTensorV2
import albumentations as A
from utils import MetricLogger, SmoothedValue, calculate_ap
import math
from collections import defaultdict
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import utils

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN model on Pascal VOC')
    parser.add_argument('--data-dir', default='data/VOCdevkit/VOC2012', help='Path to dataset')
    parser.add_argument('--batch-size', default=12, type=int, help='Batch size for training')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers for data loading')
    parser.add_argument('--num-epochs', default=20, type=int, help='Number of epochs to train')
    parser.add_argument('--learning-rate', default=0.005, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight-decay', default=0.0005, type=float, help='Weight decay')
    parser.add_argument('--step-size', default=3, type=int, help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma for learning rate scheduler')
    parser.add_argument('--output-dir', default='fasterrcnn-resnet50/checkpoints/2', help='Directory to save output files')
    parser.add_argument('--device', default='cuda', help='Device to use for training (cpu or cuda)')

    return parser.parse_args()

# 数据增强
def get_transform(train):
    """返回支持图像和标注同步变换的 Albumentations 增强管道"""
    if train:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Resize(height=800, width=800),
            A.RandomCrop(height=640, width=640, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(  # 添加归一化
                mean=[0.485, 0.456, 0.406],  # ImageNet均值
                std=[0.229, 0.224, 0.225],   # ImageNet标准差
                max_pixel_value=255.0  # 明确指定输入范围
            ),
            ToTensorV2(),  # 这会自动将图像转换为float32
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_area=16,
            min_visibility=0.1
        ))
    else:
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
    # 使用最新权重加载方式
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    backbone = torchvision.models.resnet50(weights=weights)
    
    # 移除最后两层
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    backbone.out_channels = 2048
    
    # Anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    # Feature map to perform RoI cropping
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    # Put the pieces together in the Faster R-CNN model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

# 训练一个epoch
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    logger = MetricLogger()
    logger.add_meter('lr', SmoothedValue(fmt='{value:.6f}'))
    logger.add_meter('loss', SmoothedValue(fmt='{value:.4f}'))
    header = f'Epoch: [{epoch}/{get_args().num_epochs}]'
    
    total_loss = 0.0  # 新增：累积总损失
    num_batches = 0   # 新增：批次计数
    
    for images, targets in logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Reduce losses over all GPUs for logging
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # 更新累积量
        total_loss += loss_value
        num_batches += 1
        
        logger.update(loss=losses_reduced, **loss_dict_reduced)
        logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # 返回整个epoch的平均损失（总损失/批次数量）
    return total_loss / num_batches if num_batches > 0 else 0.0

# 评估模型
def evaluate(model, data_loader, epoch, device, iou_threshold=0.5):
    """Evaluate model on test set and calculate mAP"""
    model.eval()
    
    # Storage for evaluation metrics
    all_aps = []  # 存储所有有效类别的AP值
    class_aps = defaultdict(list)  # 按类别存储AP值
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = model(images)
            
            # Collect predictions and ground truth for mAP calculation
            for target, output in zip(targets, outputs):
                ap_dict = calculate_ap(
                    target['boxes'], target['labels'],
                    output['boxes'], output['scores'], output['labels'],
                    iou_threshold=iou_threshold
                )
                
                # 只处理有值的类别
                for class_id, ap in ap_dict.items():
                    if ap > 0:  # 只添加AP大于0的类别
                        class_aps[class_id].append(ap)
                        all_aps.append(ap)
    
    # Compute mean AP for each class
    mean_aps = {}
    for label, aps in class_aps.items():
        mean_aps[label] = np.mean(aps) if aps else 0.0
    
    # Compute overall mAP
    overall_mAP = np.mean(all_aps) if all_aps else 0.0
    
    return overall_mAP, mean_aps
    

def main():
    args = get_args()
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)    

    # Create dataset and data loader
    dataset = PascalVOCDataset(args.data_dir, split='trainval', transforms=get_transform(train=True))
    dataset_test = PascalVOCDataset(args.data_dir,  split='val', transforms=get_transform(train=False))
    
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    data_loader_test = DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Create model
    model = get_model(len(VOC_CLASSES) + 1)  # +1 for background class
    model.to(args.device)
    
    # Construct optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = SGD(params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    # lr_scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # 使用 AdamW 优化器（比 Adam 更稳定，适合目标检测）
    optimizer = AdamW(
        params, 
        lr=args.learning_rate,      # 初始学习率（通常比 SGD 小，如 1e-4 或 3e-4）
        betas=(0.9, 0.999),        # 动量参数
        weight_decay=args.weight_decay,  # 权重衰减（防过拟合）
        eps=1e-8                   # 数值稳定性
    )

    # 动态学习率调度（根据验证集损失自动调整）
    lr_scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',                # 监控验证集损失（越小越好）
        factor=0.1,               # 学习率衰减因子（默认 0.1）
        patience=3,               # 连续 3 个 epoch 损失不下降则降低学习率
        # verbose=False              # 打印学习率更新日志
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    best_val_map = float('inf')
    for epoch in range(args.num_epochs):
        # Train for one epoch
        train_loss = train_one_epoch(model, optimizer, data_loader, args.device, epoch, print_freq=10)
        # lr_scheduler.step()         # Update learning rate
        val_map, aps_dict = evaluate(model, data_loader_test, epoch=epoch, device=args.device)
        train_losses.append(train_loss)
        # 输出日志
        log_line1 = f"Epoch [{epoch+1}/{args.num_epochs}]:   Train Loss: {train_loss:.4f}, Validation mAP: {val_map:.4f}"
        log_line2 = f"APs: {aps_dict}"

        print(log_line1)
        print(log_line2)
        # 保存日志到文件
        with open(os.path.join(args.output_dir,'training_log.txt'), 'a') as f:
            f.write(log_line1 + '\n')
            f.write(log_line2 + '\n\n')

        if train_loss < best_val_loss:
            best_val_loss = train_loss
            # 保存模型检查点
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict()
                },
                os.path.join(args.output_dir, f'best_losses_model.pth')
            )
            print(f"best losses Model saved at epoch {epoch} with loss {train_loss:.4f}")
        if val_map < best_val_map:
            best_val_map = val_map
            # 保存模型检查点
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict()
                },
                os.path.join(args.output_dir, f'best_map_model.pth')
            )
            print(f"best map Model saved at epoch {epoch} with map {val_map:.4f}")

    # 绘制损失曲线并保存为图片
    plt.figure(figsize=(10, 6))
    plt.plot(range(args.num_epochs), train_losses, label='Train Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))  # 保存为图片
    plt.close()  # 关闭图形，释放内存
    
    print("Training complete!")

if __name__ == "__main__":
    main()