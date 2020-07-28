'''
TensorBoard logger.
'''

# import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter

class TBLogger(object):
    def __init__(self, folder, flush_secs=60):
        self.writer = SummaryWriter(folder, flush_secs=flush_secs)

    def add_value(self, name, value, step):
        self.writer.add_scalar(tag = name, scalar_value = value, global_step=step)