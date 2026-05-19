import torch
import numpy as np

def compute_dice_score(pred, target):
    pred = pred.flatten()
    target = target.flatten()

    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    
    pred_fg = (pred == 1)
    target_fg = (target == 1)
    
    intersection = np.sum(pred_fg & target_fg)
    total = np.sum(pred_fg) + np.sum(target_fg)
    return (2.0 * intersection) / total if total > 0 else 1.0
