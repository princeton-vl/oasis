import torch
import torch.nn as nn

class CosineAngularLoss(nn.Module):
  def __init__(self):
    super(CosineAngularLoss, self).__init__()

  def forward(self,preds,truths):
    # Calculate loss : average cosine value between predicted/actual normals at each pixel
    # theta = arccos((P dot Q) / (|P|*|Q|)) -> cos(theta) = (P dot Q) / (|P|*|Q|)
    # Both the predicted and ground truth normals normalized to be between -1 and 1
    preds_norm =  torch.nn.functional.normalize(preds, p=2, dim=1)
    truths_norm = torch.nn.functional.normalize(truths, p=2, dim=1)
    # make negative so function decreases (cos -> 1 if angles same)
    loss = torch.mean(-torch.sum(preds_norm * truths_norm, dim = 1))
    return loss
