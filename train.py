
import os  
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from dataset import RetinalDataset, setup_dataset, seeding
from model import BoundaryAwareAttentionUNet
from loss import FocalTverskyLoss

def calculate_iou(pred, target, class_id):
    pred_mask = (pred == class_id)
    target_mask = (target == class_id)
    intersection = (pred_mask & target_mask).float().sum()
    union = (pred_mask | target_mask).float().sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()

def calculate_miou(pred, target, num_classes=2):
    ious = []
    for class_id in range(num_classes):
        iou = calculate_iou(pred, target, class_id)
        ious.append(iou)
    return np.mean(ious)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_epoch(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    epoch_miou = 0.0
    model.train()
    
    for x, y in tqdm(loader, desc="Training"):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        with torch.no_grad():
            y_pred_sigmoid = torch.sigmoid(y_pred)
            pred_binary = (y_pred_sigmoid > 0.5).float()
            target_binary = (y > 0.5).float()
            pred_class = pred_binary.long()
            target_class = target_binary.long()
            batch_miou = calculate_miou(pred_class, target_class)
            epoch_miou += batch_miou
    
    epoch_loss = epoch_loss / len(loader)
    epoch_miou = epoch_miou / len(loader)
    return epoch_loss, epoch_miou

def validate_epoch(model, loader, loss_fn, device):
    epoch_loss = 0.0
    epoch_miou = 0.0
    model.eval()
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation"):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()
            
            y_pred_sigmoid = torch.sigmoid(y_pred)
            pred_binary = (y_pred_sigmoid > 0.5).float()
            target_binary = (y > 0.5).float()
            pred_class = pred_binary.long()
            target_class = target_binary.long()
            batch_miou = calculate_miou(pred_class, target_class)
            epoch_miou += batch_miou

    epoch_loss = epoch_loss / len(loader)
    epoch_miou = epoch_miou / len(loader)
    return epoch_loss, epoch_miou

def train_model():
    seeding(42)
    H = 512
    W = 512
    size = (H, W)
    batch_size = 4
    lr = 1e-4
    num_epochs = 50
    checkpoint_path = "boundary_aware_attention_unet.pth"
    
    dataset_path = setup_dataset()
    train_dataset = RetinalDataset(
        image_dir=os.path.join(dataset_path, 'train_images'),
        mask_dir=os.path.join(dataset_path, 'train_labels'),
        size=size
    )
    val_dataset = RetinalDataset(
        image_dir=os.path.join(dataset_path, 'test_images'),
        mask_dir=os.path.join(dataset_path, 'test_labels'),
        size=size
    )
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BoundaryAwareAttentionUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    loss_fn = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
    
    print(f"Boundary-Aware Attention U-Net with E(x) edge detection")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    best_miou = 0.0
    train_losses, val_losses = [], []
    train_mious, val_mious = [], []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_miou = train_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_miou = validate_epoch(model, val_loader, loss_fn, device)
        scheduler.step(valid_loss)
        
        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        train_mious.append(train_miou)
        val_mious.append(valid_miou)
        
        if valid_miou > best_miou:
            best_miou = valid_miou
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved! mIoU: {best_miou:.4f}")
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f}')
        print(f'Val Loss: {valid_loss:.4f} | Val mIoU: {valid_miou:.4f}')
        print(f'Best mIoU: {best_miou:.4f}')
        print('-' * 60)
    
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Boundary-Aware Attention U-Net Loss [E(x)]')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, train_mious, 'b-', label='Training mIoU')
    ax2.plot(epochs, val_mious, 'r-', label='Validation mIoU')
    ax2.set_title('Mean IoU [E(x)]')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('mIoU')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model, best_miou
