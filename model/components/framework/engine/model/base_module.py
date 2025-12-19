import torch.nn as nn

class BaseModule(nn.Module):
    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg

    def init_weights(self):
        pass  # hook cho compatibility mmengine
