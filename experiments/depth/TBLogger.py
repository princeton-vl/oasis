'''
TensorBoard logger.
    https://pytorch.org/docs/stable/tensorboard.html
'''

import config
from torch.utils.tensorboard import SummaryWriter

class TBLogger(object):
    def __init__(self, folder, flush_secs=60):
        self.writer = SummaryWriter(log_dir = folder, flush_secs=flush_secs)
        
    def add_value(self, name, value, step):
        self.writer.add_scalar(tag = name, scalar_value = value, global_step=step)
    def add_image(self, name, value, step, dataformats):
        self.writer.add_image(tag = name, img_tensor = value, global_step=step, dataformats=dataformats)


class TBLoggerX(object):
    def __init__(self, folder, flush_secs=60):
        self.writer = SummaryWriter(log_dir = folder, flush_secs=flush_secs)
        
    def add_value(self, name, value, step):
        self.writer.add_scalar(tag = name, scalar_value = value, global_step=step)
    def add_image(self, name, value, step, dataformats):
        self.writer.add_image(tag = name, img_tensor = value, global_step=step, dataformats=dataformats)
