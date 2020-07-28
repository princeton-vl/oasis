import torch
import torch.nn as nn

# Wrapper for nn.functional.interpolate to be used in nn.Sequential() module
class Interpolate(nn.Module):
  def __init__(self, scale_factor):
    super(Interpolate, self).__init__()
    self.scale_factor = scale_factor
    self.mode = 'nearest'
      
  def forward(self, x):
    x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
    return x