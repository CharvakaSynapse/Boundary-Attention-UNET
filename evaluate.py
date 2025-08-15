import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from dataset import RetinalDataset, setup_dataset
from model import BoundaryAwareAttentionUNet
from metrics import ComprehensiveMetrics

class StandaloneEvaluator:
    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.metrics_calculator = ComprehensiveMetrics()
        print(f"Boundary-Aware Attention U-Net [E(x)] loaded successfully on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_model(self, model_path):
        model = BoundaryAwareAttentionUNet()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def evaluate_dataset(self, image_dir, mask_dir, batch_size=4, size=(512, 512)):
        dataset = RetinalDataset(image_dir, mask_dir, size=size, include_filename=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        all_metrics = []
        sample_results = []
        
        print(f"Evaluating {len(dataset)} samples with E(x) Boundary-Aware Attention U-Net...")
        
        with torch.no_grad():
            for batch_idx, (images, masks, filenames) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)
                predictions = torch.sigmoid(outputs)
                
                for i in range(images.size(0)):
                    pred = predictions[i]
                    target = masks[i]
                    filename = filenames[i]
                    sample_metrics = self.metrics_calculator.evaluate_single_sample(
                        prediction=pred,
                        target=target,
                        prediction_probs=pred
                    )
                    sample_metrics['filename'] = filename
                    all_metrics.append(sample_metrics)
                    sample_results.append({
                        'image': images[i].cpu(),
                        'ground_truth': target.cpu(),
                        'prediction': pred.cpu(),
                        'filename': filename,
                        'metrics': sample_metrics
                    })
        
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        return aggregated_metrics, sample_results
    
    def _aggregate_metrics(self, all_metrics):
        if not all_metrics:
            return {}
        aggregated = {}
        metric_keys = [key for key in all_metrics[0].keys() if key != 'filename']
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics if not np.isnan(m[key]) and not np.isinf(m[key])]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)
            else:
                aggregated[f'{key}_mean'] = 0.0
                aggregated[f'{key}_std'] = 0.0
                aggregated[f'{key}_min'] = 0.0
                aggregated[f'{key}_max'] = 0.0
        return aggregated
    
    def visualize_results(self, sample_results, num_samples=5, save_path=None):
        num_samples = min(num_samples, len(sample_results))
        sorted_samples = sorted(sample_results, key=lambda x: x['metrics']['cldice'], reverse=True)
        indices = [0, len(sorted_samples)//4, len(sorted_samples)//2, 3*len(sorted_samples)//4, len(sorted_samples)-1]
        selected_samples = [sorted_samples[i] for i in indices[:num_samples]]
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample in enumerate(selected_samples):
            image = sample['image'].permute(1, 2, 0).numpy()
            gt_mask = sample['ground_truth'].squeeze().numpy()
            pred_mask = (sample['prediction'].squeeze().numpy() > 0.5).astype(np.float32)
            image = np.clip(image, 0, 1)
            
            overlay = np.zeros((*gt_mask.shape, 3))
            overlay[gt_mask > 0.5] = [1, 0, 0]
            overlay[pred_mask > 0.5] = [0, 1, 0]
            overlap = (gt_mask > 0.5) & (pred_mask > 0.5)
            overlay[overlap] = [1, 1, 0]
            alpha = 0.3
            image_overlay = image * (1 - alpha) + overlay * alpha
            
            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f'Original\n{sample["filename"]}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(gt_mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title('Prediction [E(x)]')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(np.clip(image_overlay, 0, 1))
            axes[i, 3].set_title('Overlay\n(Red: GT, Green: Pred, Yellow: Match)')
            axes[i, 3].axis('off')
            
            metrics = sample['metrics']
            metrics_text = f"clDice: {metrics['cldice']:.3f}\n"
            metrics_text += f"Dice: {metrics['dice_coefficient']:.3f}\n"
            metrics_text += f"IoU: {metrics['vessel_iou']:.3f}\n"
            metrics_text += f"F1: {metrics['f1_score']:.3f}\n"
            metrics_text += f"AUPRC: {metrics['auprc']:.3f}"
            axes[i, 0].text(0.02, 0.98, metrics_text, transform=axes[i, 0].transAxes,
                           fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        plt.show()
    
    def generate_report(self, aggregated_metrics, save_path=None):
        report = f"""
BOUNDARY-AWARE ATTENTION U-NET - EVALUATION REPORT
=================================================
Model Configuration: E(x) - Edge detection on skip connections

PRIMARY METRICS (for early stopping):
-------------------------------------
clDice (centerline Dice):     {aggregated_metrics.get('cldice_mean', 0):.4f} ± {aggregated_metrics.get('cldice_std', 0):.4f}
Dice Coefficient (binary):    {aggregated_metrics.get('dice_coefficient_mean', 0):.4f} ± {aggregated_metrics.get('dice_coefficient_std', 0):.4f}

OVERLAP / REGION ACCURACY:
--------------------------
IoU (Jaccard) - Vessel:       {aggregated_metrics.get('vessel_iou_mean', 0):.4f} ± {aggregated_metrics.get('vessel_iou_std', 0):.4f}
mIoU (mean IoU):              {aggregated_metrics.get('mean_iou_mean', 0):.4f} ± {aggregated_metrics.get('mean_iou_std', 0):.4f}
IoU - Background:             {aggregated_metrics.get('background_iou_mean', 0):.4f} ± {aggregated_metrics.get('background_iou_std', 0):.4f}

CLASS-WISE VESSEL METRICS:
--------------------------
Precision_vessel:             {aggregated_metrics.get('precision_mean', 0):.4f} ± {aggregated_metrics.get('precision_std', 0):.4f}
Recall_vessel (Sensitivity):  {aggregated_metrics.get('sensitivity_mean', 0):.4f} ± {aggregated_metrics.get('sensitivity_std', 0):.4f}
F1_vessel:                    {aggregated_metrics.get('f1_score_mean', 0):.4f} ± {aggregated_metrics.get('f1_score_std', 0):.4f}
IoU_vessel:                   {aggregated_metrics.get('vessel_iou_mean', 0):.4f} ± {aggregated_metrics.get('vessel_iou_std', 0):.4f}

AGGREGATED CLASSIFICATION METRICS:
----------------------------------
Sensitivity (Recall):         {aggregated_metrics.get('sensitivity_mean', 0):.4f} ± {aggregated_metrics.get('sensitivity_std', 0):.4f}
Specificity:                  {aggregated_metrics.get('specificity_mean', 0):.4f} ± {aggregated_metrics.get('specificity_std', 0):.4f}
Precision:                    {aggregated_metrics.get('precision_mean', 0):.4f} ± {aggregated_metrics.get('precision_std', 0):.4f}
F1-Score:                     {aggregated_metrics.get('f1_score_mean', 0):.4f} ± {aggregated_metrics.get('f1_score_std', 0):.4f}
Pixel Accuracy:               {aggregated_metrics.get('pixel_accuracy_mean', 0):.4f} ± {aggregated_metrics.get('pixel_accuracy_std', 0):.4f}

BOUNDARY / DISTANCE METRICS:
----------------------------
Boundary F1 (BF-score):      {aggregated_metrics.get('boundary_f1_mean', 0):.4f} ± {aggregated_metrics.get('boundary_f1_std', 0):.4f}

TOPOLOGY / CONNECTIVITY:
------------------------
Skeleton/Centerline Recall:  {aggregated_metrics.get('skeleton_recall_mean', 0):.4f} ± {aggregated_metrics.get('skeleton_recall_std', 0):.4f}
Breaks per Image:             {aggregated_metrics.get('breaks_per_image_mean', 0):.2f} ± {aggregated_metrics.get('breaks_per_image_std', 0):.2f}
Spurious Branches:            {aggregated_metrics.get('spurious_branches_mean', 0):.4f} ± {aggregated_metrics.get('spurious_branches_std', 0):.4f}
Component Count Δ vs GT:     {aggregated_metrics.get('component_count_delta_mean', 0):.1f} ± {aggregated_metrics.get('component_count_delta_std', 0):.1f}

IMBALANCE-ROBUST, THRESHOLD-FREE:
---------------------------------
AUPRC (Precision-Recall AUC): {aggregated_metrics.get('auprc_mean', 0):.4f} ± {aggregated_metrics.get('auprc_std', 0):.4f}

PERFORMANCE SUMMARY:
-------------------
PRIMARY: clDice = {aggregated_metrics.get('cldice_mean', 0):.4f} (vessel topology preservation)
VESSEL:  IoU = {aggregated_metrics.get('vessel_iou_mean', 0):.4f}, F1 = {aggregated_metrics.get('f1_score_mean', 0):.4f}
ROBUST:  AUPRC = {aggregated_metrics.get('auprc_mean', 0):.4f} (threshold-independent)
CONFIG:  E(x) - Edge detection on high-resolution skip connections

METHODOLOGICAL NOTES:
---------------------
Uses high-resolution encoder features for edge detection
Sobel operators applied to skip connections (fine spatial details)
Boundary-aware attention gating with edge enhancement
Focal Tversky Loss for class imbalance handling
"""
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        print(report)
        return report

def predict_and_visualize(model, dataset, device, num_samples=4):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_samples):
            x, y_true = dataset[i]
            x_batch = x.unsqueeze(0).to(device)
            y_pred = model(x_batch)
            y_pred_sigmoid = torch.sigmoid(y_pred)
            y_pred_binary = (y_pred_sigmoid > 0.5).float()
            
            x_np = x.permute(1, 2, 0).cpu().numpy()
            y_true_np = y_true.squeeze().cpu().numpy()
            y_pred_np = y_pred_binary.squeeze().cpu().numpy()
            
            axes[i, 0].imshow(x_np)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(y_true_np, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(y_pred_np, cmap='gray')
            axes[i, 2].set_title('Prediction [E(x)]')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()