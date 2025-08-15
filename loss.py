import torch
import torch.nn as nn

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        true_pos = (inputs * targets).sum()
        false_neg = (targets * (1 - inputs)).sum()
        false_pos = ((1 - targets) * inputs).sum()
        
        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky