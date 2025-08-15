
import os
import torch
from dataset import RetinalDataset, setup_dataset
from evaluate import StandaloneEvaluator

def main():
    MODEL_PATH = "boundary_aware_attention_unet.pth"
    DATASET_PATH = "retinal_blood_vessel_icpr_seg"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        return
    
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset folder '{DATASET_PATH}' not found!")
        return
    
    evaluator = StandaloneEvaluator(MODEL_PATH)
    test_image_dir = os.path.join(DATASET_PATH, 'test_images')
    test_mask_dir = os.path.join(DATASET_PATH, 'test_labels')
    
    print("Starting comprehensive evaluation of E(x) model...")
    aggregated_metrics, sample_results = evaluator.evaluate_dataset(
        image_dir=test_image_dir,
        mask_dir=test_mask_dir,
        batch_size=4,
        size=(512, 512)
    )
    
    print("\nGenerating evaluation report...")
    evaluator.generate_report(aggregated_metrics, save_path="evaluation_report_ex.txt")
    
    print("\nCreating visualizations...")
    evaluator.visualize_results(sample_results, num_samples=5, save_path="evaluation_visualization_ex.png")
    
    print("\nE(x) evaluation completed successfully!")
    print(f"- Report saved to: evaluation_report_ex.txt")
    print(f"- Visualization saved to: evaluation_visualization_ex.png")
    
    print(f"\n{'='*60}")
    print("KEY PERFORMANCE METRICS [E(x)]")
    print(f"{'='*60}")
    print(f"clDice (Topology):     {aggregated_metrics.get('cldice_mean', 0):.4f}")
    print(f"Boundary F1:          {aggregated_metrics.get('boundary_f1_mean', 0):.4f}")
    print(f"Vessel IoU:           {aggregated_metrics.get('vessel_iou_mean', 0):.4f}")
    print(f"Dice Coefficient:     {aggregated_metrics.get('dice_coefficient_mean', 0):.4f}")
    print(f"F1-Score:             {aggregated_metrics.get('f1_score_mean', 0):.4f}")
    print(f"AUPRC:                {aggregated_metrics.get('auprc_mean', 0):.4f}")

if __name__ == "__main__":
    main()
