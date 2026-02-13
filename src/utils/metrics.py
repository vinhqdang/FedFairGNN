import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

def calculate_auc(y_true, y_pred_prob):
    """
    Calculate Area Under the ROC Curve.
    
    Args:
        y_true (np.array or torch.Tensor): True binary labels.
        y_pred_prob (np.array or torch.Tensor): Predicted probabilities (scores).
        
    Returns:
        float: AUC-ROC score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_prob, torch.Tensor):
        y_pred_prob = y_pred_prob.detach().cpu().numpy()
        
    try:
        return roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        return 0.5  # Handle cases with only one class present

def calculate_f1(y_true, y_pred_prob, threshold=0.5):
    """
    Calculate F1 Score.
    
    Args:
        y_true (np.array or torch.Tensor): True binary labels.
        y_pred_prob (np.array or torch.Tensor): Predicted probabilities.
        threshold (float): Classification threshold.
        
    Returns:
        float: F1 score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_prob, torch.Tensor):
        y_pred_prob = y_pred_prob.detach().cpu().numpy()
        
    y_pred = (y_pred_prob >= threshold).astype(int)
    return f1_score(y_true, y_pred)

def calculate_dpd(y_pred_prob, sensitive_attr):
    """
    Calculate Demographic Parity Difference (DPD).
    DPD = |P(Y_hat=1 | S=0) - P(Y_hat=1 | S=1)|
    Using soft probabilities (mean prediction) as a proxy for rate.
    
    Args:
        y_pred_prob (torch.Tensor): Predicted probabilities.
        sensitive_attr (torch.Tensor): Sensitive attributes (binary 0/1).
        
    Returns:
        float: DPD value.
    """
    if isinstance(y_pred_prob, np.ndarray):
        y_pred_prob = torch.tensor(y_pred_prob)
    if isinstance(sensitive_attr, np.ndarray):
        sensitive_attr = torch.tensor(sensitive_attr)
        
    # Ensure they are on the same device and type
    y_pred_prob = y_pred_prob.float().view(-1)
    sensitive_attr = sensitive_attr.view(-1)
    
    # Masks
    mask_0 = (sensitive_attr == 0)
    mask_1 = (sensitive_attr == 1)
    
    if mask_0.sum() == 0 or mask_1.sum() == 0:
        return 0.0  # Cannot compute if one group is missing
        
    mean_0 = y_pred_prob[mask_0].mean()
    mean_1 = y_pred_prob[mask_1].mean()
    
    dpd = torch.abs(mean_0 - mean_1)
    return dpd.item()
