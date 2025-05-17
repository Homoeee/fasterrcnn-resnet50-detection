import time
from collections import defaultdict, deque
import torch
import torch.distributed as dist
import numpy as np
from collections import defaultdict
from torchvision.ops import box_iou
from dataset import VOC_CLASSES
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.start_time = time.time()
    
    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)
    
    def log_every(self, iterable, print_freq, header=None):
        """日志记录迭代器"""
        i = 0
        if not header:
            header = ''
        
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            
            i += 1
            if i % print_freq == 0 or i == len(iterable):
                elapsed = time.time() - start_time
                eta = elapsed / i * (len(iterable) - elapsed)
                
                log_msg = [
                    header,
                    f'[{i}/{len(iterable)}]',
                    f'eta: {time.strftime("%H:%M:%S", time.gmtime(eta))}',
                    f'iter: {iter_time}',
                    # f'data: {data_time}',
                ]
                for name, meter in self.meters.items():
                    log_msg.append(f'{name}: {str(meter)}')
                
                log_msg.append(f'time: {time.strftime("%H:%M:%S", time.gmtime(elapsed))}')
                print(self.delimiter.join(log_msg))
            
            end = time.time()
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a window."""
    def __init__(self, window_size=20, fmt=None):
        self.window_size = window_size
        self.fmt = fmt or "{median:.4f}"
        self.reset()
    
    def reset(self):
        self.deque = deque(maxlen=self.window_size)
        self.total = 0.0
        self.count = 0
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.total += value * n
        self.count += n
    
    @property
    def median(self):
        if not self.deque:
            return 0
        return np.median(list(self.deque))
    
    @property
    def avg(self):
        if not self.count:
            return 0
        return self.total / self.count
    
    @property
    def global_avg(self):
        if not self.count:
            return 0
        return self.total / self.count
    
    @property
    def value(self):
        return self.deque[-1] if self.deque else 0
    
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            value=self.value
        )



def reduce_dict(input_dict, average=True):
    """
    对字典中的所有张量进行跨进程归约操作
    参数:
        input_dict (dict): 所有值都是张量的字典
        average (bool): 是否计算平均值而不是求和
    返回:
        归约后的字典
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    
    return reduced_dict

def calculate_ap(gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels, iou_threshold=0.5):
    """
    计算单张图片的每个类别的AP (Average Precision)
    
    返回:
        字典 {class_id: ap_value}
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return defaultdict(float)  # 返回空字典
    
    # 获取所有存在的类别
    all_classes = set(gt_labels.cpu().numpy()).union(set(pred_labels.cpu().numpy()))
    # print(f"all_classes: {all_classes}")
    
    ap_dict = {}
    for cls in all_classes:
        # 筛选当前类别的GT和预测
        cls_gt_indices = [i for i, label in enumerate(gt_labels) if label == cls]
        cls_pred_indices = [i for i, label in enumerate(pred_labels) if label == cls]
        
        if not cls_gt_indices:  # 如果没有该类的GT，跳过
            continue
            
        cls_gt_boxes = gt_boxes[cls_gt_indices]
        cls_gt_labels = gt_labels[cls_gt_indices]
        
        if cls_pred_indices:
            cls_pred_boxes = pred_boxes[cls_pred_indices]
            cls_pred_scores = pred_scores[cls_pred_indices]
            cls_pred_labels = pred_labels[cls_pred_indices]
        else:
            cls_pred_boxes = torch.zeros((0, 4), device=pred_boxes.device)
            cls_pred_scores = torch.zeros(0, device=pred_scores.device)
            cls_pred_labels = torch.zeros(0, device=pred_labels.device)
        
        # 初始化匹配标记
        gt_matched = np.zeros(len(cls_gt_boxes), dtype=bool)
        
        # 按置信度降序排序预测框
        sorted_indices = np.argsort(-cls_pred_scores.cpu().numpy())
        cls_pred_boxes = cls_pred_boxes[sorted_indices]
        cls_pred_labels = cls_pred_labels[sorted_indices]
        
        # 初始化TP和FP数组
        tp = np.zeros(len(cls_pred_boxes))
        fp = np.zeros(len(cls_pred_boxes))
        
        # 对每个预测框匹配真实框
        for i, pred_box in enumerate(cls_pred_boxes):
            # 计算IoU
            ious = box_iou(pred_box.unsqueeze(0), cls_gt_boxes)
            best_iou = ious.max().item()
            best_gt_idx = ious.argmax().item()
            
            if best_iou >= iou_threshold:
                if not gt_matched[best_gt_idx]:
                    tp[i] = 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        # 计算精度和召回率曲线
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / len(cls_gt_boxes)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        # print(f"recalls: {recalls}, precisions: {precisions}")

        # 计算AP（使用11点插值法）
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            mask = recalls >= t
            if mask.any():
                ap += np.max(precisions[mask]) / 11
        
        ap_dict[cls.item()] = ap
    
    return ap_dict

def visualize_predictions(model, dataset, device, output_dir, num_images=20, iou_threshold=0.5, confidence_threshold=0.5):
    """Visualize predictions with confidence thresholding and NMS"""
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # Select random indices
    indices = random.sample(range(len(dataset)), min(num_images, len(dataset)))
    
    model.eval()
    for idx in indices:
        image, target = dataset[idx]
        img_tensor = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(img_tensor)[0]
        
        # Convert image to numpy for visualization
        img_np = (image.permute(1, 2, 0).cpu().numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
        
        # Plot image
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img_np)
        
        # # Plot ground truth (green)
        # for box, label in zip(target['boxes'].cpu().numpy(), target['labels'].cpu().numpy()):
        #     xmin, ymin, xmax, ymax = box
        #     rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='g', facecolor='none')
        #     ax.add_patch(rect)
        #     ax.text(xmin, ymin, VOC_CLASSES[label], color='white', fontsize=10, bbox={'facecolor': 'green', 'alpha': 0.7})
        
        # Process predictions
        if len(prediction['boxes']) > 0:
            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            
            # 1. Sort by confidence
            sorted_indices = np.argsort(-scores)
            boxes = boxes[sorted_indices]
            scores = scores[sorted_indices]
            labels = labels[sorted_indices]
            
            # 2. Apply confidence threshold (新增步骤)
            keep_mask = scores >= confidence_threshold
            boxes = boxes[keep_mask]
            scores = scores[keep_mask]
            labels = labels[keep_mask]
            
            # 3. NMS
            keep_indices = []
            for i in range(len(boxes)):
                keep = True
                for j in keep_indices:
                    # if labels[i] == labels[j] and calculate_iou(boxes[i], boxes[j]) > iou_threshold:
                    if scores[i] < scores[j] and box_iou(torch.tensor(boxes[i]).unsqueeze(0), torch.tensor(boxes[j]).unsqueeze(0)).item() > iou_threshold:
                        keep = False
                        break
                if keep:
                    keep_indices.append(i)
            
            # Plot kept predictions (red)
            for i in keep_indices:
                xmin, ymin, xmax, ymax = boxes[i]
                rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin, f"{VOC_CLASSES[labels[i]]}: {scores[i]:.2f}", 
                        color='white', fontsize=10, bbox={'facecolor': 'red', 'alpha': 0.7})
        
        # Save image
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'visualizations', f'pred_{idx}.png'), bbox_inches='tight', dpi=300)
        plt.close()

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    return iou

def visualize_results(mean_aps, output_dir):
    """Visualize per-class AP results"""
    # Sort classes by AP
    sorted_classes = sorted(mean_aps.items(), key=lambda x: x[1], reverse=True)
    class_names = [class_name for class_name, _ in sorted_classes] 
    ap_values = [ap for _, ap in sorted_classes]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(class_names, ap_values, color='skyblue')
    plt.xlabel('Average Precision (AP)')
    plt.title('Per-class AP@0.5')
    plt.xlim(0, 1.0)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}',
                ha='left', va='center')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'per_class_ap.png'))
    plt.close()
