import numpy as np
import torch
from skimage.morphology import skeletonize, binary_dilation, binary_erosion, remove_small_objects
from skimage.measure import label
from skimage.segmentation import find_boundaries
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve, auc

class ComprehensiveMetrics:
    def __init__(self):
        self.results = {}
    
    def calculate_cldice(self, pred, target, smooth=1e-6):
        pred_np = (pred.squeeze().cpu().numpy() > 0.5).astype(bool)
        target_np = (target.squeeze().cpu().numpy() > 0.5).astype(bool)
        
        pred_skeleton = skeletonize(pred_np)
        target_skeleton = skeletonize(target_np)
        
        if not target_skeleton.any():
            return 1.0 if not pred_skeleton.any() else 0.0
        
        pred_skel_in_target = np.sum(pred_skeleton * target_np)
        target_skel_in_pred = np.sum(target_skeleton * pred_np)
        
        pred_skel_sum = np.sum(pred_skeleton)
        target_skel_sum = np.sum(target_skeleton)
        target_sum = np.sum(target_np)
        pred_sum = np.sum(pred_np)
        
        if pred_skel_sum == 0 or target_sum == 0:
            term1 = 0.0
        else:
            term1 = 2 * pred_skel_in_target / (pred_skel_sum + target_sum + smooth)
            
        if target_skel_sum == 0 or pred_sum == 0:
            term2 = 0.0
        else:
            term2 = 2 * target_skel_in_pred / (target_skel_sum + pred_sum + smooth)
        
        cldice = (term1 + term2) / 2
        return cldice
    
    def calculate_dice_coefficient(self, pred, target, smooth=1e-6):
        pred_binary = (pred > 0.5).float()
        target_binary = (target > 0.5).float()
        
        intersection = (pred_binary * target_binary).sum()
        dice = (2.0 * intersection + smooth) / (pred_binary.sum() + target_binary.sum() + smooth)
        
        return dice.item()
    
    def calculate_iou_metrics(self, pred, target, num_classes=2):
        pred_class = (pred > 0.5).long()
        target_class = (target > 0.5).long()
        
        ious = []
        for class_id in range(num_classes):
            pred_mask = (pred_class == class_id)
            target_mask = (target_class == class_id)
            
            intersection = (pred_mask & target_mask).float().sum()
            union = (pred_mask | target_mask).float().sum()
            
            if union == 0:
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = (intersection / union).item()
            ious.append(iou)
        
        return {
            'background_iou': ious[0],
            'vessel_iou': ious[1],
            'mean_iou': np.mean(ious)
        }
    
    def calculate_boundary_f1(self, pred, target, tolerance=2):
        pred_np = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        target_np = (target.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        
        pred_boundary = find_boundaries(pred_np, mode='inner')
        target_boundary = find_boundaries(target_np, mode='inner')
        
        if not target_boundary.any():
            return 1.0 if not pred_boundary.any() else 0.0
        
        kernel = np.ones((tolerance*2+1, tolerance*2+1), np.uint8)
        target_dilated = binary_dilation(target_boundary, kernel)
        pred_dilated = binary_dilation(pred_boundary, kernel)
        
        boundary_tp = np.sum(pred_boundary & target_dilated)
        boundary_fp = np.sum(pred_boundary & ~target_dilated)
        boundary_fn = np.sum(target_boundary & ~pred_dilated)
        
        if boundary_tp + boundary_fp == 0:
            boundary_precision = 0.0
        else:
            boundary_precision = boundary_tp / (boundary_tp + boundary_fp)
            
        if boundary_tp + boundary_fn == 0:
            boundary_recall = 0.0
        else:
            boundary_recall = boundary_tp / (boundary_tp + boundary_fn)
        
        if boundary_precision + boundary_recall == 0:
            boundary_f1 = 0.0
        else:
            boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
        
        return boundary_f1
    
    def calculate_auprc(self, pred_probs, target):
        pred_flat = pred_probs.cpu().numpy().flatten()
        target_flat = (target > 0.5).cpu().numpy().flatten().astype(int)
        
        if np.sum(target_flat) == 0:
            return 1.0 if np.sum(pred_flat > 0.5) == 0 else 0.0
        
        precision, recall, _ = precision_recall_curve(target_flat, pred_flat)
        auprc = auc(recall, precision)
        
        return auprc
    
    def calculate_topology_metrics(self, pred, target):
        pred_np = (pred.squeeze().cpu().numpy() > 0.5).astype(bool)
        target_np = (target.squeeze().cpu().numpy() > 0.5).astype(bool)
        
        pred_clean = remove_small_objects(pred_np, min_size=10)
        target_clean = remove_small_objects(target_np, min_size=10)
        
        pred_skeleton = skeletonize(pred_clean)
        target_skeleton = skeletonize(target_clean)
        
        pred_components = label(pred_skeleton)
        target_components = label(target_skeleton)
        
        pred_num_components = pred_components.max()
        target_num_components = target_components.max()
        
        component_count_delta = abs(pred_num_components - target_num_components)
        
        dilated_target_skel = binary_dilation(target_skeleton, np.ones((3, 3)))
        skeleton_overlap = np.sum(pred_skeleton & dilated_target_skel)
        total_target_skeleton = np.sum(target_skeleton)
        
        if total_target_skeleton > 0:
            skeleton_recall = skeleton_overlap / total_target_skeleton
            breaks_per_image = max(0, 1 - skeleton_recall) * target_num_components
        else:
            skeleton_recall = 1.0 if not pred_skeleton.any() else 0.0
            breaks_per_image = 0
        
        dilated_pred_skel = binary_dilation(pred_skeleton, np.ones((3, 3)))
        spurious_skeleton = pred_skeleton & ~dilated_target_skel
        spurious_branches = np.sum(spurious_skeleton) / max(1, np.sum(pred_skeleton))
        
        return {
            'breaks_per_image': breaks_per_image,
            'spurious_branches': spurious_branches,
            'component_count_delta': component_count_delta,
            'skeleton_recall': skeleton_recall
        }
    
    def evaluate_single_sample(self, prediction, target, prediction_probs=None):
        pred = prediction
        target_tensor = target
        pred_prob = prediction_probs if prediction_probs is not None else pred
        
        pred_flat = (pred > 0.5).cpu().numpy().flatten().astype(int)
        target_flat = (target > 0.5).cpu().numpy().flatten().astype(int)
        
        metrics = {}
        metrics['cldice'] = self.calculate_cldice(pred, target_tensor)
        metrics['dice_coefficient'] = self.calculate_dice_coefficient(pred, target_tensor)
        iou_metrics = self.calculate_iou_metrics(pred, target_tensor)
        metrics.update(iou_metrics)
        metrics['sensitivity'] = recall_score(target_flat, pred_flat, zero_division=0)
        metrics['specificity'] = recall_score(1 - target_flat, 1 - pred_flat, zero_division=0)
        metrics['precision'] = precision_score(target_flat, pred_flat, zero_division=0)
        metrics['f1_score'] = f1_score(target_flat, pred_flat, zero_division=0)
        metrics['pixel_accuracy'] = accuracy_score(target_flat, pred_flat)
        metrics['boundary_f1'] = self.calculate_boundary_f1(pred, target_tensor)
        topology_metrics = self.calculate_topology_metrics(pred, target_tensor)
        metrics.update(topology_metrics)
        metrics['auprc'] = self.calculate_auprc(pred_prob, target_tensor)
        
        return metrics